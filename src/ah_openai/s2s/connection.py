"""
WebSocket connection management for OpenAI S2S with low-latency optimizations
"""
import json
import base64
import socket
import asyncio
import logging
import websockets
import nanoid
from .utils import JSON_PREFIX, JSON_SUFFIX, LatencyTracker
import time

logger = logging.getLogger(__name__)

# Global storage for WebSocket connections
_sockets = {}

# Latency trackers per session
_latency_trackers = {}


async def create_connection(url, api_key, buffer_size=4096):
    """
    Create an optimized WebSocket connection to OpenAI with minimal latency.
    
    Args:
        url: WebSocket URL
        api_key: OpenAI API key
        buffer_size: TCP send/receive buffer size (default 4096 for low latency)
                    Lower = lower latency but may drop on slow networks
                    Recommended: 4096 (4KB), Aggressive: 2048, Extreme: 1024
    
    Returns:
        WebSocket connection
    """
    logger.info(f"Creating S2S connection with {buffer_size} byte buffers")
    
    # Connect with websockets library
    ws = await websockets.connect(
        url,
        additional_headers={"Authorization": f"Bearer {api_key}"},
        ping_interval=None,  # Disable ping/pong for lower overhead
        max_size=10 * 1024 * 1024,  # 10MB max message size
    )
    
    # Set socket options after connection
    if hasattr(ws, 'transport') and ws.transport:
        sock = ws.transport.get_extra_info('socket')
        if sock:
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, buffer_size)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, buffer_size)
            try:
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_QUICKACK, 1)
            except (AttributeError, OSError):
                pass
    
    logger.info("S2S WebSocket connected with low-latency optimizations enabled")
    return ws


async def initialize_session(ws, system_prompt, voice):
    """
    Initialize the OpenAI session with configuration.
    
    Args:
        ws: WebSocket connection
        system_prompt: System instructions
        voice: Voice to use (alloy, echo, fable, onyx, nova, shimmer, marin)
    """
    session_update = {
        "type": "session.update",
        "session": {
            "type": "realtime",
            "instructions": system_prompt,
            "audio": {
                "input": {
                    "format": {
                        "type": "audio/pcmu"
                    },
                    "noise_reduction": {
                        "type": "near_field"
                    },
                    "turn_detection": {
                        "type": "semantic_vad",
                        "eagerness": "high",
                        "create_response": True,
                        "interrupt_response": True
                    }
                    #"transcription": {
                    #    "language": "en",
                    #    "model": "gpt-4o-transcribe"
                    #},
                    #"turn_detection": {
                    #    "type": "server_vad",
                    #    "threshold": 0.5,
                    #    "prefix_padding_ms": 300,
                    #    "silence_duration_ms": 350,
                    #    "create_response": True,
                    #    "interrupt_response": True
                    #}
                },
                "output": {
                    "voice": voice,
                    "format": {
                        "type": "audio/pcmu"
                    }
                }
            },
            "tools": [
                {
                    "type": "function",
                    "name": "output",
                    "description": "Call this function with JSON-encoded function calls if necessary.",
                    "parameters": {
                        "type": "object",
                        "strict": True,
                        "properties": {
                            "text": {
                                "type": "string",
                                "description": "Properly escaped JSON for the command and arguments"
                            }
                        }
                    }
                }
            ],
            "tool_choice": "auto"
        }
    }
    
    await ws.send(json.dumps(session_update))
    logger.info("S2S session initialized")


async def send_audio_chunk(ws, audio_bytes, context):
    """
    Send an audio chunk to OpenAI with optimized encoding.
    
    Args:
        ws: WebSocket connection
        audio_bytes: Audio data in ulaw format
        context: Context with log_id
    """
    # Get or create latency tracker for this session
    if context.log_id not in _latency_trackers:
        _latency_trackers[context.log_id] = LatencyTracker()
    
    tracker = _latency_trackers[context.log_id]
    start_time = time.perf_counter()
    
    # Optimized: Use pre-encoded JSON structure instead of json.dumps()
    base64_chunk = base64.b64encode(audio_bytes).decode('ascii')
    json_str = JSON_PREFIX + base64_chunk + JSON_SUFFIX
    
    await ws.send(json_str)
    
    # Track latency
    latency_ms = (time.perf_counter() - start_time) * 1000
    tracker.record(latency_ms, len(audio_bytes))


async def send_message(ws, message, context):
    """
    Send a text message to OpenAI.
    
    Args:
        ws: WebSocket connection
        message: Message dict with 'role' and 'content'
        context: Context with log_id
    """
    parts = []
    for item in message['content']:
        part = {}
        if item['type'] == 'text':
            part['type'] = 'input_text'
            part['text'] = item['text']
            parts.append(part)
        else:
            raise Exception(f"Unimplemented content type: {item['type']}")
    
    event = {
        "type": "conversation.item.create",
        "item": {
            "type": "message",
            "role": "user",
            "content": parts
        },
        "event_id": nanoid.generate()
    }
    
    await ws.send(json.dumps(event))
    await ws.send(json.dumps({"type": "response.create"}))
    logger.debug(f"Sent message to OpenAI S2S")


def store_socket(log_id, ws):
    """Store a WebSocket connection for later retrieval"""
    _sockets[log_id] = ws


def get_socket(log_id):
    """Retrieve a stored WebSocket connection"""
    ws = _sockets.get(log_id)
    if not ws:
        raise Exception(f"No active OpenAI socket for log_id {log_id}")
    return ws


def remove_socket(log_id):
    """Remove a WebSocket connection from storage"""
    if log_id in _sockets:
        del _sockets[log_id]
    if log_id in _latency_trackers:
        del _latency_trackers[log_id]


async def close_connection(log_id):
    """Close a WebSocket connection"""
    ws = _sockets.get(log_id)
    if ws:
        await ws.close()
        remove_socket(log_id)
        logger.info(f"Closed S2S connection for {log_id}")
