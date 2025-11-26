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
from .audio_pacer import AudioPacer
logger = logging.getLogger(__name__)

# Timing instrumentation for S2S latency analysis
_timing_log_file = None
_session_timings = {}  # Track per-session timing data
_first_audio_delta_logged = {}  # Track if we've logged first audio delta per response

_last_ai_audio_done_time = {}  # Track when AI audio finished per session

def _log_timing(event_type, context, extra=""):
    """Log timing event to instrumentation file."""
    global _timing_log_file
    try:
        if _timing_log_file is None:
            _timing_log_file = open("data/s2s_instr.log", "a")
        timestamp = time.perf_counter()
        wall_time = time.strftime("%H:%M:%S")
        session_id = context.log_id if context else "unknown"
        _timing_log_file.write(f"{wall_time} | {timestamp:.6f} | {session_id} | {event_type} | {extra}\n")
        _timing_log_file.flush()
    except Exception as e:
        logger.error(f"Failed to log timing: {e}")

_audio_pacers = {}

async def handle_audio_delta(server_event, on_audio_chunk, play_local, context):
    """Handle incoming audio delta from OpenAI"""
    global _first_audio_delta_logged
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
            # Log first audio delta of each response
            if session_id not in _first_audio_delta_logged or not _first_audio_delta_logged[session_id]:
                _log_timing("AI_FIRST_AUDIO_DELTA", context, f"bytes={len(audio_bytes)}")
                _first_audio_delta_logged[session_id] = True
            
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
        elif event_type == 'response.output_audio.done':
            # Reset first audio delta flag for next response
            if context and context.log_id in _first_audio_delta_logged:
                _first_audio_delta_logged[context.log_id] = False
            # Track when AI audio finished
            _last_ai_audio_done_time[context.log_id] = time.perf_counter()
            _log_timing("AI_AUDIO_DONE", context)
            if context and context.log_id in _audio_pacers:
                # Mark response done so pacer resets timing for next response
                # This fixes latency drift between normal turns vs interruptions
                _audio_pacers[context.log_id].mark_response_done()
                logger.info(f'Audio generation done for {context.log_id}, marked for timing reset')
        elif event_type == 'conversation.item.input_audio_transcription.completed':
            transcript = server_event.get('transcript', '')[:50]
            _log_timing("USER_TRANSCRIPT_COMPLETE", context, f"transcript={transcript}...")
            await handle_transcript(server_event, on_transcript, context)
        elif event_type == 'input_audio_buffer.speech_started':
            # Determine if this is a true interrupt or normal turn
            session_id = context.log_id if context else None
            pacer_has_audio = False
            time_since_ai_done = None
            if session_id and session_id in _audio_pacers:
                pacer_has_audio = len(_audio_pacers[session_id].buffer) > 0
            if session_id and session_id in _last_ai_audio_done_time:
                time_since_ai_done = time.perf_counter() - _last_ai_audio_done_time[session_id]
            
            turn_type = "INTERRUPT" if pacer_has_audio else "NORMAL"
            extra = f"type={turn_type}, pacer_has_audio={pacer_has_audio}, time_since_ai_done={time_since_ai_done:.3f}s" if time_since_ai_done else f"type={turn_type}, pacer_has_audio={pacer_has_audio}, time_since_ai_done=None"
            _log_timing("USER_SPEECH_STARTED", context, extra)
            if context and context.log_id in _audio_pacers:
                logger.info(f'Interrupt detected - clearing audio pacer for {context.log_id}')
                # Clear the buffer but keep the pacer alive
                await _audio_pacers[context.log_id].clear()
            await on_interrupt(server_event)
        elif event_type == 'input_audio_buffer.speech_stopped':
            _log_timing("USER_SPEECH_STOPPED", context)
        elif event_type == 'response.created':
            _log_timing("AI_RESPONSE_CREATED", context)
        elif event_type == 'response.output_audio.started':
            _log_timing("AI_AUDIO_STARTED", context)
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
            #await asyncio.sleep(0.000025)

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
