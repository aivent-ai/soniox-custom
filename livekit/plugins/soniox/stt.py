# Copyright 2025 LiveKit, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import asyncio
import enum
import json
import os
import time
from dataclasses import dataclass
from typing import Optional

import aiohttp

from livekit import rtc
from livekit.agents import (
    APIConnectionError,
    APIConnectOptions,
    APIStatusError,
    APITimeoutError,
    stt,
    utils,
    vad,
)
from livekit.agents.stt import SpeechEventType
from livekit.agents.types import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    NotGivenOr,
)

from .log import logger

# Base URL for Soniox Speech-to-Text API.
BASE_URL = "wss://stt-rt.soniox.com/transcribe-websocket"

# WebSocket messages and tokens.
KEEPALIVE_MESSAGE = '{"type": "keepalive"}'
FINALIZE_MESSAGE = '{"type": "finalize"}'
END_TOKEN = "<end>"
FINALIZED_TOKEN = "<fin>"


class StreamState(enum.Enum):
    """State of an individual stream in the dual-stream system."""
    IDLE = "idle"  # Stream ist bereit, Audio zu empfangen
    ACTIVE = "active"  # Stream verarbeitet aktiv Audio
    RECONNECTING = "reconnecting"  # Stream wird neu verbunden
    CLOSED = "closed"  # Stream ist geschlossen


def is_end_token(token: dict) -> bool:
    """Return True if the given token marks an end or finalized event."""
    return token.get("text") in (END_TOKEN, FINALIZED_TOKEN)


@dataclass
class STTOptions:
    """Configuration options for Soniox Speech-to-Text service."""

    model: str | None = "stt-rt-preview"
    language_hints: list[str] | None = None
    context: str | None = None

    num_channels: int = 1
    sample_rate: int = 16000

    enable_language_identification: bool = True

    enable_non_final_tokens: bool = True
    max_non_final_tokens_duration_ms: int | None = None

    client_reference_id: str | None = None
    
    # Dual-stream configuration
    enable_dual_stream: bool = True  # Enable dual-stream mode to prevent context caching
    stream_switch_delay_ms: int = 100  # Delay when switching streams
    max_reconnect_attempts: int = 3  # Max reconnect attempts per stream


class STT(stt.STT):
    """Speech-to-Text service using Soniox Speech-to-Text API.

    This service connects to Soniox Speech-to-Text API for real-time transcription
    with support for multiple languages, custom context, speaker diarization,
    and more.

    For complete API documentation, see: https://soniox.com/docs/speech-to-text/api-reference/websocket-api
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str = BASE_URL,
        http_session: aiohttp.ClientSession | None = None,
        vad: vad.VAD | None = None,
        params: STTOptions | None = None,
    ):
        """Initialize instance of Soniox Speech-to-Text API service.

        Args:
            api_key: Soniox API key, if not provided, will look for SONIOX_API_KEY env variable.
            base_url: Base URL for Soniox Speech-to-Text API, default to BASE_URL defined in this
                module.
            http_session: Optional aiohttp.ClientSession to use for requests.
            vad: If passed, enable Voice Activity Detection (VAD) for audio frames.
            params: Additional configuration parameters, such as model, language hints, context and
                speaker diarization.
        """
        super().__init__(capabilities=stt.STTCapabilities(streaming=True, interim_results=True))

        self._api_key = api_key or os.getenv("SONIOX_API_KEY")
        self._base_url = base_url
        self._http_session = http_session
        self._vad_stream = vad.stream() if vad else None
        self._params = params or STTOptions()

    async def _recognize_impl(
        self,
        buffer: utils.AudioBuffer,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions,
    ) -> stt.SpeechEvent:
        """Raise error since single-frame recognition is not supported
        by Soniox Speech-to-Text API."""
        raise NotImplementedError(
            "Soniox Speech-to-Text API does not support single frame recognition"
        )

    def stream(
        self,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> SpeechStream:
        """Return a new LiveKit streaming speech-to-text session."""
        # Always use regular SpeechStream with reconnect support
        return SpeechStream(
            stt=self,
            conn_options=conn_options,
        )


class SpeechStream(stt.SpeechStream):
    def __init__(
        self,
        stt: STT,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> None:
        """Set up state and queues for a WebSocket-based transcription stream."""
        super().__init__(stt=stt, conn_options=conn_options, sample_rate=stt._params.sample_rate)
        self._stt = stt
        self._ws: aiohttp.ClientWebSocketResponse | None = None
        self._reconnect_event = asyncio.Event()

        self.audio_queue = asyncio.Queue()

        self._last_tokens_received: float | None = None
        
        # Dual-stream support
        self._utterance_count = 0
        self._should_reconnect_after_utterance = self._stt._params.enable_dual_stream

    def _ensure_session(self) -> aiohttp.ClientSession:
        """Get or create an aiohttp ClientSession for WebSocket connections."""
        if not self._stt._http_session:
            self._stt._http_session = utils.http_context.http_session()

        return self._stt._http_session

    async def _connect_ws(self):
        """Open a WebSocket connection to the Soniox Speech-to-Text API and send the
        initial configuration."""
        # If VAD was passed, disable endpoint detection, otherwise enable it.
        enable_endpoint_detection = not self._stt._vad_stream

        # Create initial config object.
        config = {
            "api_key": self._stt._api_key,
            "model": self._stt._params.model,
            "audio_format": "pcm_s16le",
            "num_channels": self._stt._params.num_channels or 1,
            "enable_endpoint_detection": enable_endpoint_detection,
            "sample_rate": self._stt._params.sample_rate,
            "language_hints": self._stt._params.language_hints,
            "context": self._stt._params.context,
            "enable_non_final_tokens": self._stt._params.enable_non_final_tokens,
            "max_non_final_tokens_duration_ms": self._stt._params.max_non_final_tokens_duration_ms,
            "enable_language_identification": self._stt._params.enable_language_identification,
            "client_reference_id": self._stt._params.client_reference_id,
        }
        # Connect to the Soniox Speech-to-Text API.
        ws = await asyncio.wait_for(
            self._ensure_session().ws_connect(self._stt._base_url),
            timeout=self._conn_options.timeout,
        )
        # Set initial configuration message.
        await ws.send_str(json.dumps(config))
        logger.debug("Soniox Speech-to-Text API connection established!")
        return ws

    async def _run(self) -> None:
        """Manage connection lifecycle, spawning tasks and handling reconnection."""
        while True:
            try:
                ws = await self._connect_ws()
                self._ws = ws
                # Create task for audio processing, voice turn detection and message handling.
                tasks = [
                    asyncio.create_task(self._prepare_audio_task()),
                    asyncio.create_task(self._handle_vad_task()),
                    asyncio.create_task(self._send_audio_task()),
                    asyncio.create_task(self._recv_messages_task()),
                    asyncio.create_task(self._keepalive_task()),
                ]
                wait_reconnect_task = asyncio.create_task(self._reconnect_event.wait())
                try:
                    done, _ = await asyncio.wait(
                        [asyncio.gather(*tasks), wait_reconnect_task],
                        return_when=asyncio.FIRST_COMPLETED,
                    )

                    for task in done:
                        if task != wait_reconnect_task:
                            task.result()

                    if wait_reconnect_task not in done:
                        break

                    self._reconnect_event.clear()
                finally:
                    await utils.aio.gracefully_cancel(*tasks, wait_reconnect_task)
            # Handle errors.
            except asyncio.TimeoutError as e:
                logger.error(
                    f"Timeout during Soniox Speech-to-Text API connection/initialization: {e}"
                )
                raise APITimeoutError(
                    "Timeout connecting to or initializing Soniox Speech-to-Text API session"
                ) from e

            except aiohttp.ClientResponseError as e:
                logger.error(
                    "Soniox Speech-to-Text API status error during session init:"
                    + f"{e.status} {e.message}"
                )
                raise APIStatusError(
                    message=e.message, status_code=e.status, request_id=None, body=None
                ) from e

            except aiohttp.ClientError as e:
                logger.error(f"Soniox Speech-to-Text API connection error: {e}")
                raise APIConnectionError(f"Soniox Speech-to-Text API connection error: {e}") from e

            except Exception as e:
                logger.exception(f"Unexpected error occurred: {e}")
                raise APIConnectionError(f"An unexpected error occurred: {e}") from e
            # Close the WebSocket connection on finish.
            finally:
                if self._ws is not None:
                    await self._ws.close()
                    self._ws = None

    async def _keepalive_task(self):
        """Periodically send keepalive messages (while no audio is being sent)
        to maintain the WebSocket connection."""
        try:
            while self._ws:
                await self._ws.send_str(KEEPALIVE_MESSAGE)
                await asyncio.sleep(5)
        except Exception as e:
            logger.error(f"Error while sending keep alive message: {e}")

    async def _prepare_audio_task(self):
        """Read audio frames, process VAD, and enqueue PCM data for sending."""
        if not self._ws:
            logger.error("WebSocket connection to Soniox Speech-to-Text API is not established")
            return

        async for data in self._input_ch:
            if self._stt._vad_stream:
                # If VAD is enabled, push the audio frame to the VAD stream.
                if isinstance(data, self._FlushSentinel):
                    self._stt._vad_stream.flush()
                else:
                    self._stt._vad_stream.push_frame(data)

            if isinstance(data, rtc.AudioFrame):
                # Get the raw bytes from the audio frame.
                pcm_data = data.data.tobytes()
                self.audio_queue.put_nowait(pcm_data)

    async def _send_audio_task(self):
        """Take queued audio data and transmit it over the WebSocket."""
        if not self._ws:
            logger.error("WebSocket connection to Soniox Speech-to-Text API is not established")
            return

        while self._ws:
            try:
                data = await self.audio_queue.get()

                if isinstance(data, bytes):
                    await self._ws.send_bytes(data)
                else:
                    await self._ws.send_str(data)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error while sending audio data: {e}")
                break

    async def _handle_vad_task(self):
        """Listen for VAD events to trigger finalize or keepalive messages."""
        if not self._stt._vad_stream:
            logger.debug("VAD stream is not enabled, skipping VAD task")
            return

        async for event in self._stt._vad_stream:
            if event.type == vad.VADEventType.END_OF_SPEECH:
                self.audio_queue.put_nowait(FINALIZE_MESSAGE)

    async def _recv_messages_task(self):
        """Receive transcription messages, handle tokens, errors, and dispatch events."""

        # Transcription frame will be only sent after we get the "endpoint" event.
        final_transcript_buffer = ""
        # Language code sent by Soniox if language detection is enabled (e.g. "en", "de", "fr")
        final_transcript_language: str = ""

        def send_endpoint_transcript():
            nonlocal final_transcript_buffer, final_transcript_language
            if final_transcript_buffer:
                event = stt.SpeechEvent(
                    type=SpeechEventType.FINAL_TRANSCRIPT,
                    alternatives=[
                        stt.SpeechData(
                            text=final_transcript_buffer, language=final_transcript_language
                        )
                    ],
                )
                self._event_ch.send_nowait(event)
                final_transcript_buffer = ""
                final_transcript_language = ""

        # Method handles receiving messages from the Soniox Speech-to-Text API.
        while self._ws:
            try:
                async for msg in self._ws:
                    if msg.type == aiohttp.WSMsgType.TEXT:
                        try:
                            content = json.loads(msg.data)
                            tokens = content["tokens"]

                            if tokens:
                                if len(tokens) == 1 and tokens[0]["text"] == FINALIZED_TOKEN:
                                    # Ignore finalized token, prevent auto finalize cycle.
                                    pass
                                else:
                                    # Got at least one token, reset the auto finalize delay.
                                    self._last_tokens_received = time.time()

                            # We will only send the final tokens after we get the "endpoint" event.
                            non_final_transcription = ""
                            non_final_transcription_language: str = ""

                            for token in tokens:
                                if token["is_final"]:
                                    if is_end_token(token):
                                        # Found an endpoint, tokens until here will be sent as
                                        # transcript, the rest will be sent as interim tokens
                                        # (even final tokens).
                                        send_endpoint_transcript()
                                    else:
                                        final_transcript_buffer += token["text"]

                                        # Soniox provides language for each token,
                                        # LiveKit requires only a single language for the entire transcription chunk.
                                        # Current heuristic is to take the first language we see.
                                        if token.get("language") and not final_transcript_language:
                                            final_transcript_language = token.get("language")
                                else:
                                    non_final_transcription += token["text"]
                                    if (
                                        token.get("language")
                                        and not non_final_transcription_language
                                    ):
                                        non_final_transcription_language = token.get("language")

                            if final_transcript_buffer or non_final_transcription:
                                event = stt.SpeechEvent(
                                    type=SpeechEventType.INTERIM_TRANSCRIPT,
                                    alternatives=[
                                        stt.SpeechData(
                                            text=final_transcript_buffer + non_final_transcription,
                                            language=final_transcript_language
                                            if final_transcript_language
                                            else non_final_transcription_language,
                                        )
                                    ],
                                )
                                self._event_ch.send_nowait(event)

                            error_code = content.get("error_code")
                            error_message = content.get("error_message")

                            if error_code or error_message:
                                # In case of error, still send the final transcript.
                                send_endpoint_transcript()
                                logger.error(f"WebSocket error: {error_code} - {error_message}")

                            finished = content.get("finished")

                            if finished:
                                # When finished, still send the final transcript.
                                send_endpoint_transcript()
                                logger.debug("Transcription finished")
                                
                                # Trigger reconnect for dual-stream mode
                                if self._should_reconnect_after_utterance:
                                    self._utterance_count += 1
                                    logger.info(f"Utterance #{self._utterance_count} completed, triggering reconnect for fresh context")
                                    self._reconnect_event.set()

                        except Exception as e:
                            logger.exception(f"Error processing message: {e}")
                    elif msg.type in (
                        aiohttp.WSMsgType.CLOSED,
                        aiohttp.WSMsgType.CLOSE,
                        aiohttp.WSMsgType.CLOSING,
                    ):
                        break
                    else:
                        logger.warning(
                            f"Unexpected message type from Soniox Speech-to-Text API: {msg.type}"
                        )
            except aiohttp.ClientError as e:
                logger.error(f"WebSocket error while receiving: {e}")
            except Exception as e:
                logger.error(f"Unexpected error while receiving messages: {e}")


class DualSpeechStream_DEPRECATED(stt.SpeechStream):
    """Dual-stream implementation to prevent Soniox context caching.
    
    Uses two parallel WebSocket connections and switches between them after
    each utterance to ensure fresh context for each transcription.
    """
    
    def __init__(
        self,
        stt: STT,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> None:
        """Initialize dual-stream manager."""
        super().__init__(stt=stt, conn_options=conn_options)
        self._stt = stt
        
        # Two parallel streams
        self._stream_a: Optional[SpeechStream] = None
        self._stream_b: Optional[SpeechStream] = None
        
        # Stream states
        self._stream_a_state = StreamState.IDLE
        self._stream_b_state = StreamState.IDLE
        
        # Currently active stream
        self._active_stream: Optional[SpeechStream] = None
        self._inactive_stream: Optional[SpeechStream] = None
        
        # Locks for thread safety
        self._switch_lock = asyncio.Lock()
        
        # Statistics
        self._switch_count = 0
        self._last_switch_time: Optional[float] = None
        
        # Control flags
        self._running = False
        self._switching = False

    async def _run(self) -> None:
        """Main run loop for dual-stream manager."""
        self._running = True
        
        try:
            # Initialize both streams
            logger.info("Initializing dual-stream mode...")
            
            # Create first stream
            self._stream_a = SpeechStream(self._stt, self._conn_options)
            self._stream_a_state = StreamState.IDLE
            
            # Create second stream  
            self._stream_b = SpeechStream(self._stt, self._conn_options)
            self._stream_b_state = StreamState.IDLE
            
            # Set stream A as initial active stream
            self._active_stream = self._stream_a
            self._inactive_stream = self._stream_b
            self._stream_a_state = StreamState.ACTIVE
            
            logger.info("Dual-stream mode initialized successfully")
            
            # Start tasks
            tasks = [
                asyncio.create_task(self._handle_audio_task()),
                asyncio.create_task(self._handle_vad_task()),
                asyncio.create_task(self._merge_events_task()),
                asyncio.create_task(self._monitor_streams_task()),
            ]
            
            # Start both stream's run loops
            stream_a_task = asyncio.create_task(self._stream_a._run())
            stream_b_task = asyncio.create_task(self._stream_b._run())
            
            tasks.extend([stream_a_task, stream_b_task])
            
            # Wait for all tasks
            done, pending = await asyncio.wait(
                tasks, 
                return_when=asyncio.FIRST_EXCEPTION
            )
            
            for task in done:
                if task.exception():
                    raise task.exception()
            
        except Exception as e:
            logger.error(f"Error in dual-stream manager: {e}")
            raise
        finally:
            self._running = False
            await self._cleanup_streams()

    async def _handle_endpoint_detection(self) -> None:
        """Handle endpoint detection for stream switching.
        
        Since we're not using VAD, we need to detect when Soniox sends
        an endpoint token to trigger stream switch.
        """
        while self._running:
            if self._active_stream:
                try:
                    # Monitor the active stream's event channel for END tokens
                    event = await asyncio.wait_for(
                        self._active_stream._event_ch.recv(),
                        timeout=0.5
                    )
                    
                    # Check if this is a final transcript (endpoint detected)
                    if event.type == SpeechEventType.FINAL_TRANSCRIPT:
                        logger.debug("Endpoint detected, switching streams...")
                        await self._switch_streams("endpoint_detection")
                    
                    # Forward the event to output
                    self._event_ch.send_nowait(event)
                    
                except asyncio.TimeoutError:
                    continue
                except Exception as e:
                    logger.error(f"Error in endpoint detection: {e}")
            else:
                await asyncio.sleep(0.1)

    async def _handle_audio_task(self) -> None:
        """Forward audio frames to the active stream."""
        async for data in self._input_ch:
            if not self._active_stream:
                continue
                
            # During switch, buffer audio briefly
            if self._switching:
                await asyncio.sleep(0.01)
                
            # Forward to active stream
            if self._active_stream:
                self._active_stream._input_ch.send_nowait(data)

    async def _handle_vad_task(self) -> None:
        """Handle VAD events and trigger stream switches."""
        # If VAD is disabled, use endpoint detection instead
        if not self._stt._vad_stream:
            await self._handle_endpoint_detection()
            return
            
        # VAD-based switching
        async for event in self._stt._vad_stream:
            if event.type == vad.VADEventType.END_OF_SPEECH:
                logger.debug("VAD detected end of speech, switching streams...")
                await self._switch_streams("vad_end_of_speech")

    async def _switch_streams(self, reason: str) -> None:
        """Switch between streams A and B.
        
        Args:
            reason: Reason for the switch (for logging)
        """
        async with self._switch_lock:
            if not self._active_stream or not self._inactive_stream:
                logger.error("Cannot switch: streams not properly initialized")
                return
                
            self._switching = True
            
            try:
                logger.debug(f"Switching streams (reason: {reason})")
                
                # Finalize current stream
                if self._active_stream._ws:
                    await self._active_stream._ws.send_str(FINALIZE_MESSAGE)
                
                # Wait briefly for smooth transition
                await asyncio.sleep(self._stt._params.stream_switch_delay_ms / 1000.0)
                
                # Swap active and inactive streams
                old_active = self._active_stream
                self._active_stream = self._inactive_stream
                self._inactive_stream = old_active
                
                # Update states
                if old_active == self._stream_a:
                    self._stream_a_state = StreamState.RECONNECTING
                    self._stream_b_state = StreamState.ACTIVE
                else:
                    self._stream_b_state = StreamState.RECONNECTING
                    self._stream_a_state = StreamState.ACTIVE
                
                # Reconnect the old active stream in background
                asyncio.create_task(self._reconnect_stream(
                    old_active,
                    "A" if old_active == self._stream_a else "B"
                ))
                
                # Update statistics
                self._switch_count += 1
                self._last_switch_time = time.time()
                
                logger.info(f"Stream switch #{self._switch_count} completed")
                
            finally:
                self._switching = False

    async def _reconnect_stream(self, stream: SpeechStream, name: str) -> None:
        """Reconnect a stream with fresh context.
        
        Args:
            stream: Stream to reconnect
            name: Stream name for logging
        """
        logger.debug(f"Reconnecting stream {name}...")
        
        for attempt in range(self._stt._params.max_reconnect_attempts):
            try:
                # Close existing connection
                if stream._ws:
                    await stream._ws.close()
                    stream._ws = None
                
                # Brief pause before reconnect
                await asyncio.sleep(0.2)
                
                # Create a new fresh stream instance
                new_stream = SpeechStream(self._stt, self._conn_options)
                
                # Replace the old stream
                if name == "A":
                    self._stream_a = new_stream
                    self._inactive_stream = new_stream
                    self._stream_a_state = StreamState.IDLE
                    # Start the new stream's run loop
                    asyncio.create_task(new_stream._run())
                else:
                    self._stream_b = new_stream
                    self._inactive_stream = new_stream
                    self._stream_b_state = StreamState.IDLE
                    # Start the new stream's run loop
                    asyncio.create_task(new_stream._run())
                
                logger.debug(f"Stream {name} reconnected successfully")
                return
                
            except Exception as e:
                logger.error(f"Failed to reconnect stream {name} "
                           f"(attempt {attempt + 1}/{self._stt._params.max_reconnect_attempts}): {e}")
                
                if attempt == self._stt._params.max_reconnect_attempts - 1:
                    # Mark as closed after final attempt
                    if name == "A":
                        self._stream_a_state = StreamState.CLOSED
                    else:
                        self._stream_b_state = StreamState.CLOSED

    async def _merge_events_task(self) -> None:
        """Merge events from both streams into single output channel."""
        # This task is handled by _handle_endpoint_detection when VAD is disabled
        # or by forwarding events directly when VAD is enabled
        if self._stt._vad_stream:
            # With VAD, we need to merge events differently
            while self._running:
                if self._active_stream:
                    try:
                        # Get event from active stream with timeout
                        event = await asyncio.wait_for(
                            self._active_stream._event_ch.recv(),
                            timeout=0.1
                        )
                        # Forward to output channel
                        self._event_ch.send_nowait(event)
                    except asyncio.TimeoutError:
                        continue
                    except Exception as e:
                        logger.error(f"Error merging events: {e}")
                else:
                    await asyncio.sleep(0.1)

    async def _monitor_streams_task(self) -> None:
        """Monitor stream health and log statistics."""
        last_log_time = time.time()
        
        while self._running:
            try:
                # Check stream health
                if self._stream_a_state == StreamState.CLOSED and \
                   self._stream_b_state == StreamState.CLOSED:
                    logger.error("Both streams closed, stopping...")
                    break
                
                # Log statistics periodically
                if time.time() - last_log_time > 30:
                    active_name = "A" if self._active_stream == self._stream_a else "B"
                    logger.info(
                        f"Dual-stream stats: switches={self._switch_count}, "
                        f"active={active_name}, states=[A={self._stream_a_state.value}, "
                        f"B={self._stream_b_state.value}]"
                    )
                    last_log_time = time.time()
                
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.error(f"Error in monitor task: {e}")

    async def _cleanup_streams(self) -> None:
        """Clean up both streams on shutdown."""
        logger.info("Cleaning up dual-stream connections...")
        
        if self._stream_a and self._stream_a._ws:
            await self._stream_a._ws.close()
            
        if self._stream_b and self._stream_b._ws:
            await self._stream_b._ws.close()
        
        logger.info("Dual-stream cleanup completed")