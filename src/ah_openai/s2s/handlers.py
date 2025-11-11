"""
Message and audio handlers for OpenAI S2S WebSocket
"""
import json
import base64
import traceback
import asyncio
import logging
import websockets

logger = logging.getLogger(__name__)


async def handle_audio_delta(server_event, on_audio_chunk, play_local, context):
    """Handle incoming audio delta from OpenAI"""
    try:
        audio_bytes = base64.b64decode(server_event['delta'])
        logger.debug(f"Audio chunk: {len(audio_bytes)} bytes")
        
        # Play locally if requested
        if play_local:
            import numpy as np
            import sounddevice as sd
            audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
            sd.play(audio_array, 24000, blocking=True)
        
        # Call the audio chunk callback if provided
        if on_audio_chunk:
            await on_audio_chunk(audio_bytes, context=context)
    except Exception as e:
        logger.error(f"Error handling audio delta: {e}")
        traceback.print_exc()


async def handle_function_call(item, on_command, context):
    """Handle function call from OpenAI"""
    try:
        arguments = json.loads(item['arguments'])
        
        if item['name'] != 'output':
            # Direct function call
            cmd_name = item['name']
            args = json.loads(item['arguments'])
            cmd = {cmd_name: args}
            logger.info(f"Invoking command: {cmd}")
            await on_command(cmd, context=context)
        else:
            # Output function with JSON commands
            cmd = json.loads(arguments['text'])
            
            # Handle both single command and list of commands
            if isinstance(cmd, list):
                for single_cmd in cmd:
                    logger.info(f"Invoking command: {single_cmd}")
                    await on_command(single_cmd, context=context)
            else:
                logger.info(f"Invoking command: {cmd}")
                await on_command(cmd, context=context)
                
    except Exception as e:
        logger.error(f"Error in function call handler: {e}")
        traceback.print_exc()
        
        # Send error back to OpenAI
        try:
            from . import connection
            trace = traceback.format_exc()
            error_msg = {
                "role": "user",
                "content": [{
                    "type": "text",
                    "text": f"[SYSTEM: Error executing command: {str(e)}\n{trace}]"
                }]
            }
            ws = connection.get_socket(context.log_id)
            await connection.send_message(ws, error_msg, context)
        except Exception as send_error:
            logger.error(f"Failed to send error message: {send_error}")


async def handle_transcript(server_event, on_transcript, context):
    """Handle transcript from conversation.item.done events"""
    try:
        item = server_event['item']
        role = item['role']
        transcript = None
        
        # Extract transcript from content array for both roles
        for content_item in item.get('content', []):
            if role == 'assistant' and content_item.get('type') == 'output_audio':
                if 'transcript' in content_item:
                    transcript = content_item['transcript']
                    break
            elif role == 'user':
                # User content might have transcript property or be input_text type
                if 'transcript' in content_item:
                    transcript = content_item['transcript']
                    break
                elif content_item.get('type') == 'input_text' and 'text' in content_item:
                    transcript = content_item['text']
                    break
        
        if transcript and on_transcript:
            logger.info(f"Transcript from {role}: {transcript}")
            await on_transcript(role, transcript, context=context)
    except Exception as e:
        logger.error(f"Error handling transcript: {e}")
        traceback.print_exc()


async def handle_message(server_event, on_command, on_audio_chunk, on_transcript, play_local, context):
    """Handle a single message from OpenAI"""
    try:
        event_type = server_event['type']
        logger.debug(f"Received server event: {event_type}")
        
        if event_type == "response.output_audio.delta":
            await handle_audio_delta(server_event, on_audio_chunk, play_local, context)
            
        elif event_type == "conversation.item.done":
            item = server_event['item']
            if item['type'] == "function_call":
                await handle_function_call(item, on_command, context)
            elif item['type'] == "message":
                await handle_transcript(server_event, on_transcript, context)
        else:
            # Log other message types for debugging
            logger.debug(f"Received message: {json.dumps(server_event, indent=2)}")
            
    except Exception as e:
        logger.error(f"Error handling message: {e}")
        traceback.print_exc()


async def message_handler_loop(ws, on_command, on_audio_chunk, on_transcript, play_local, context):
    """Background task to handle incoming WebSocket messages"""
    try:
        async for message in ws:
            server_event = json.loads(message)
            await handle_message(server_event, on_command, on_audio_chunk, on_transcript, play_local, context)
            
    except websockets.exceptions.ConnectionClosed:
        logger.info(f"WebSocket connection closed for {context.log_id}")
    except Exception as e:
        logger.error(f"Error in message handler loop: {e}")
        traceback.print_exc()
