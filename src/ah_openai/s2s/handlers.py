"""
Message and audio handlers for OpenAI S2S WebSocket
"""
import json
import base64
import traceback
import asyncio
import logging
import websockets
from collections import deque
import time
logger = logging.getLogger(__name__)
_audio_pacers = {}

class AudioPacer:
    """Paces audio chunks to real-time speed with small buffer."""

    def __init__(self):
        self.buffer = deque()
        self.pacer_task = None
        self.on_audio_chunk = None
        self.context = None
        self._running = False

    async def add_chunk(self, audio_bytes):
        """Add audio chunk to buffer with backpressure."""
        if self._running:
            self.buffer.append(audio_bytes)
            await asyncio.sleep(0.0002)

    async def start_pacing(self, on_audio_chunk, context):
        """Start real-time pacing task."""
        self.on_audio_chunk = on_audio_chunk
        self.context = context
        self._running = True
        # a little more time for the second chunk to arrive
        await asyncio.sleep(0.12)
        self.pacer_task = asyncio.create_task(self._pace_loop())

    async def _pace_loop(self):
        """Send buffered chunks at real-time intervals."""
        while self._running:
            if len(self.buffer) > 0:
                chunk = self.buffer.popleft()
                duration = len(chunk) / 8000.0
                duration *= 0.9999  # Slightly faster than real-time
                await self.on_audio_chunk(chunk, context=self.context)
                await asyncio.sleep(duration)
            else:
                await asyncio.sleep(0.05)

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

async def handle_audio_delta(server_event, on_audio_chunk, play_local, context):
    """Handle incoming audio delta from OpenAI"""
    try:
        audio_bytes = base64.b64decode(server_event['delta'])
        logger.debug(f'Audio chunk: {len(audio_bytes)} bytes')
        if play_local:
            import numpy as np
            import sounddevice as sd
            audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
            sd.play(audio_array, 24000, blocking=True)
        if on_audio_chunk and context:
            session_id = context.log_id
            if session_id not in _audio_pacers:
                pacer = AudioPacer()
                await pacer.start_pacing(on_audio_chunk, context)
                _audio_pacers[session_id] = pacer
                logger.info(f'Started audio pacer for session {session_id}')
            await _audio_pacers[session_id].add_chunk(audio_bytes)
    except Exception as e:
        logger.error(f'Error handling audio delta: {e}')
        traceback.print_exc()

async def handle_function_call(item, on_command, context):
    """Handle function call from OpenAI"""
    try:
        arguments = json.loads(item['arguments'])
        if item['name'] != 'output':
            cmd_name = item['name']
            args = json.loads(item['arguments'])
            cmd = {cmd_name: args}
            logger.info(f'Invoking command: {cmd}')
            await on_command(cmd, context=context)
        else:
            cmd = json.loads(arguments['text'])
            if isinstance(cmd, list):
                for single_cmd in cmd:
                    logger.info(f'Invoking command: {single_cmd}')
                    await on_command(single_cmd, context=context)
            else:
                logger.info(f'Invoking command: {cmd}')
                await on_command(cmd, context=context)
    except Exception as e:
        logger.error(f'Error in function call handler: {e}')
        traceback.print_exc()
        try:
            from . import connection
            trace = traceback.format_exc()
            error_msg = {'role': 'user', 'content': [{'type': 'text', 'text': f'[SYSTEM: Error executing command: {str(e)}\n{trace}]'}]}
            ws = connection.get_socket(context.log_id)
            await connection.send_message(ws, error_msg, context)
        except Exception as send_error:
            logger.error(f'Failed to send error message: {send_error}')

async def handle_transcript(server_event, on_transcript, context):
    """Handle transcript from conversation events"""
    try:
        if 'transcript' in server_event:
            transcript = server_event['transcript']
            await on_transcript('user', transcript, context=context)
            return
        item = server_event['item']
        role = item['role']
        transcript = None
        for content_item in item.get('content', []):
            if role == 'assistant' and content_item.get('type') == 'output_audio':
                if 'transcript' in content_item:
                    transcript = content_item['transcript']
                    # I think this means done outputting AI response audio
                    break
            elif role == 'user':
                if 'transcript' in content_item:
                    transcript = content_item['transcript']
                    break
                elif content_item.get('type') == 'input_text' and 'text' in content_item:
                    transcript = content_item['text']
                    break
        if transcript and on_transcript:
            logger.info(f'Transcript from {role}: {transcript}')
            await on_transcript(role, transcript, context=context)
    except Exception as e:
        logger.error(f'Error handling transcript: {e}')
        traceback.print_exc()

async def handle_message(server_event, on_command, on_audio_chunk, on_transcript, on_interrupt, play_local, context):
    """Handle a single message from OpenAI"""
    try:
        event_type = server_event['type']
        if event_type == 'response.output_audio.delta':
            await handle_audio_delta(server_event, on_audio_chunk, play_local, context)
        elif event_type == 'conversation.item.input_audio_transcription.completed':
            await handle_transcript(server_event, on_transcript, context)
        elif event_type == 'input_audio_buffer.speech_started':
            if context and context.log_id in _audio_pacers:
                logger.info(f'Interrupt detected - stopping audio pacer for {context.log_id}')
                await _audio_pacers[context.log_id].stop()
                del _audio_pacers[context.log_id]
            await on_interrupt(server_event)
        elif event_type == 'conversation.item.done':
            item = server_event['item']
            if item['type'] == 'function_call':
                await handle_function_call(item, on_command, context)
            elif item['type'] == 'message':
                await handle_transcript(server_event, on_transcript, context)
        else:
            logger.debug(f'Received message: {json.dumps(server_event, indent=2)}')
    except Exception as e:
        logger.error(f'Error handling message: {e}')
        traceback.print_exc()

async def message_handler_loop(ws, on_command, on_audio_chunk, on_transcript, on_interrupt, play_local, context):
    """Background task to handle incoming WebSocket messages"""
    try:
        session_id = context.log_id
        async for message in ws:
            server_event = json.loads(message)
            await handle_message(server_event, on_command, on_audio_chunk, on_transcript, on_interrupt, play_local, context)
            await asyncio.sleep(0.000025)

    except websockets.exceptions.ConnectionClosed:
        logger.info(f'WebSocket connection closed for {context.log_id}')
    except Exception as e:
        logger.error(f'Error in message handler loop: {e}')
        traceback.print_exc()
    finally:
        if context and context.log_id in _audio_pacers:
            logger.info(f'Cleaning up audio pacer for {context.log_id}')
            await _audio_pacers[context.log_id].stop()
            del _audio_pacers[context.log_id]
