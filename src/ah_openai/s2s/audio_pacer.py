"""
Audio Pacer for OpenAI S2S

Handles buffering and pacing of audio chunks to match real-time playback speed.
"""
import asyncio
import time
import logging
from collections import deque

logger = logging.getLogger(__name__)

class AudioPacer:
    """Paces audio chunks to real-time speed with small buffer.
    
    Persistent version: Designed to be reused across multiple response turns
    to prevent audio overlap race conditions.
    """

    def __init__(self):
        self.buffer = deque()
        self.pacer_task = None
        self.on_audio_chunk = None
        self.context = None
        self._running = False
        
        # Absolute timing for precise pacing
        self.start_time = None
        self.bytes_sent = 0
        self.audio_start_time = None  # When first audio chunk of CURRENT response received
        self.playback_rate = 1.0  # Real-time playback (configurable)

    async def add_chunk(self, audio_bytes):
        """Add audio chunk to buffer."""
        if self._running:
            self.buffer.append(audio_bytes)
            
            # Track when first audio of this sequence arrives
            if self.audio_start_time is None:
                self.audio_start_time = time.perf_counter()

    async def clear(self):
        """Clear buffer and reset state for interruption.
        
        This empties the queue immediately and resets timing, making the
        pacer ready for the NEW response immediately without needing to
        stop/start tasks.
        """
        self.buffer.clear()
        self.audio_start_time = None
        self.bytes_sent = 0
        # Reset pacing clock to now so we don't try to catch up to the past
        self.start_time = time.perf_counter()
        logger.info("AudioPacer cleared and reset")

    async def start_pacing(self, on_audio_chunk, context):
        """Start real-time pacing task."""
        self.on_audio_chunk = on_audio_chunk
        self.context = context
        self._running = True
        
        # Wait for initial buffer to build up
        # This prevents underruns at the start
        # initial_buffer_time = 0.04  # 100ms initial buffer
        # await asyncio.sleep(initial_buffer_time)
        
        # Record absolute start time
        self.start_time = time.perf_counter()
        self.bytes_sent = 0
        
        self.pacer_task = asyncio.create_task(self._pace_loop())

    async def _pace_loop(self):
        """Send buffered chunks at real-time intervals using absolute timing.
        
        This prevents drift and maintains consistent playback speed by calculating
        the target time based on total bytes sent, not accumulated sleep durations.
        """
        while self._running:
            if len(self.buffer) > 0:
                chunk = self.buffer.popleft()
                
                # Send the chunk
                # Calculate timestamp for this chunk based on when it should play
                # Timestamp is relative to when first audio was received
                if self.audio_start_time:
                    # Calculate when this chunk should start playing
                    chunk_timestamp = self.audio_start_time + (self.bytes_sent / 8000.0)
                    # print(f"[AUDIOPACER] Calling on_audio_chunk with {len(chunk)} bytes, timestamp={chunk_timestamp}")
                    await self.on_audio_chunk(chunk, timestamp=chunk_timestamp, context=self.context)
                else:
                    # Fallback if no start time (shouldn't happen)
                    # print(f"[AUDIOPACER] Calling on_audio_chunk with {len(chunk)} bytes, NO timestamp")
                    await self.on_audio_chunk(chunk, context=self.context)
                
                # Update bytes sent counter
                self.bytes_sent += len(chunk)
                
                # Calculate target time based on total bytes sent
                # At 8000 Hz, each byte = 1/8000 seconds
                target_time = self.start_time + (self.bytes_sent / 8000.0) * self.playback_rate
                
                # Calculate how long to sleep to hit target time
                current_time = time.perf_counter()
                sleep_duration = target_time - current_time
                
                # Sleep if we're ahead of schedule
                if sleep_duration > 0:
                    await asyncio.sleep(sleep_duration)
                # If we're behind schedule (sleep_duration < 0), don't sleep - catch up
                
            else:
                # No data in buffer, short sleep
                await asyncio.sleep(0.005)

    async def stop(self):
        """Stop pacing and clear buffer."""
        self._running = False
        if self.pacer_task:
            self.pacer_task.cancel()
            try:
                await self.pacer_task
            except asyncio.CancelledError:
                pass
        self.buffer.clear()
