"""
Audio encoding utilities and optimization constants for S2S
"""
import base64
import struct
import time
import logging

logger = logging.getLogger(__name__)

# Pre-encoded JSON structure for audio chunks (optimization)
# Avoids repeated json.dumps() calls for high-frequency audio sends
JSON_PREFIX = '{"type":"input_audio_buffer.append","audio":"'
JSON_SUFFIX = '"}'


def float_to_16bit_pcm(float32_array):
    """Convert float32 audio samples to 16-bit PCM"""
    clipped = [max(-1.0, min(1.0, x)) for x in float32_array]
    pcm16 = b''.join(struct.pack('<h', int(x * 32767)) for x in clipped)
    return pcm16


def base64_encode_audio(float32_array):
    """Convert float32 audio to base64-encoded 16-bit PCM"""
    pcm_bytes = float_to_16bit_pcm(float32_array)
    encoded = base64.b64encode(pcm_bytes).decode('ascii')
    return encoded


class LatencyTracker:
    """Track send latency for performance monitoring"""
    
    def __init__(self, sample_size=100):
        self.samples = []
        self.sample_size = sample_size
        self.chunk_count = 0
        self.total_bytes = 0
    
    def record(self, latency_ms, byte_count):
        """Record a latency sample"""
        self.samples.append(latency_ms)
        self.chunk_count += 1
        self.total_bytes += byte_count
        
        if len(self.samples) >= self.sample_size:
            avg_latency = sum(self.samples) / len(self.samples)
            avg_bytes = self.total_bytes / self.chunk_count
            logger.info(
                f"S2S Performance: {self.chunk_count} chunks sent, "
                f"avg latency: {avg_latency:.2f}ms, "
                f"avg size: {avg_bytes:.0f} bytes"
            )
            self.samples = []
    
    def get_stats(self):
        """Get current statistics"""
        return {
            'chunk_count': self.chunk_count,
            'total_bytes': self.total_bytes,
            'avg_bytes': self.total_bytes / self.chunk_count if self.chunk_count > 0 else 0
        }
