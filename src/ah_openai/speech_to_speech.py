"""
OpenAI Speech-to-Speech (S2S) Realtime API Integration

Optimized for low-latency audio streaming with:
- TCP_NODELAY for immediate packet sends
- Minimal buffer sizes (configurable)
- Pre-encoded JSON for audio chunks
- Native asyncio websockets (no threading)
- Latency monitoring
"""
import os
import asyncio
import logging
from lib.providers.services import service
from .s2s import connection, handlers, utils
logger = logging.getLogger(__name__)

@service()
async def start_s2s(model=None, system_prompt='', on_command=None, on_audio_chunk=None, on_transcript=None, on_interrupt=None, voice='marin', play_local=False, context=None, buffer_size=4096, **kwargs):
    """
    Start a speech-to-speech OpenAI realtime websocket session.
    Session will be identified by context.log_id

    Arguments:
        model: Model name (default: 'gpt-realtime')
        system_prompt: System instructions for the AI
        on_command: Async callback for function calls from the server
                   Signature: async def on_command(cmd_dict, context)
        on_audio_chunk: Async callback for audio chunks from the server
                       Signature: async def on_audio_chunk(audio_bytes, context)
                       Audio format: int16 PCM at 24000 Hz sample rate
        voice: OpenAI voice (alloy, echo, fable, onyx, nova, shimmer, marin)
        play_local: Whether to play audio locally (default True)
        buffer_size: TCP buffer size in bytes (default 4096 for low latency)
                    Lower = lower latency but may drop on slow networks
                    Recommended: 4096 (4KB), Aggressive: 2048, Extreme: 1024
        context: MindRoot context object with log_id

    Returns:
        None (connection stored globally by log_id)
    """
    if model is None:
        model = 'gpt-realtime'
    OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
    if not OPENAI_API_KEY:
        raise Exception('OPENAI_API_KEY environment variable not set')
    url = f'wss://api.openai.com/v1/realtime?model={model}'
    logger.info(f'Starting S2S session {context.log_id} with {buffer_size}B buffers')
    ws = await connection.create_connection(url, OPENAI_API_KEY, buffer_size)
    await connection.initialize_session(ws, system_prompt, voice)
    connection.store_socket(context.log_id, ws)
    asyncio.create_task(handlers.message_handler_loop(ws, on_command, on_audio_chunk, on_transcript, on_interrupt, play_local, context))
    logger.info(f'S2S session {context.log_id} started successfully')

@service()
async def send_s2s_message(message, context=None):
    """
    Send a text message to the OpenAI S2S session.
    
    Args:
        message: Dict with 'role' and 'content' (list of content items)
                Example: {"role": "user", "content": [{"type": "text", "text": "Hello"}]}
        context: MindRoot context with log_id
    """
    ws = connection.get_socket(context.log_id)
    await connection.send_message(ws, message, context)

@service()
async def send_s2s_audio_chunk(audio_bytes, context=None):
    """
    Send an audio chunk to OpenAI for processing.
    
    Args:
        audio_bytes: Audio data in ulaw format (from PySIP)
        context: MindRoot context with log_id
    """
    ws = connection.get_socket(context.log_id)
    await connection.send_audio_chunk(ws, audio_bytes, context)

@service()
async def close_s2s_session(context=None):
    """
    Close an S2S session and clean up resources.
    
    Args:
        context: MindRoot context with log_id
    """
    await connection.close_connection(context.log_id)
    logger.info(f'S2S session {context.log_id} closed')