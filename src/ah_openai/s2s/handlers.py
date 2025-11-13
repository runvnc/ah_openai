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

# Global state for real-time audio pacing per session
_audio_pacers = {}  # session_id -> AudioPacer

class AudioPacer:
    """Paces audio output to real-time speed with small buffer."""
    def __init__(self, frame_size=320, frame_duration_ms=20):
        self.frame_size = frame_size  # bytes per frame (320 bytes = 20ms at 24kHz int16)
        self.frame_duration = frame_duration_ms / 1000.0  # seconds
        self.buffer = deque(maxlen=3)  # 2-3 frame buffer
        self.pacer_task = None
        self.on_audio_chunk = None
        self.context = None
        self._running = False
        
    async def add_chunk(self, audio_bytes):
        """Add audio chunk, split into frames and buffer."""
        # Split chunk into frames
        for i in range(0, len(audio_bytes), self.frame_size):
            frame = audio_bytes[i:i+self.frame_size]
            if len(frame) == self.frame_size:  # Only queue complete frames
                self.buffer.append(frame)
    
    async def start_pacing(self, on_audio_chunk, context):
        """Start real-time pacing task."""
        self.on_audio_chunk = on_audio_chunk
        self.context = context
        self._running = True
        self.pacer_task = asyncio.create_task(self._pace_loop())
    
    async def _pace_loop(self):
        """Send buffered frames at real-time intervals."""
        while self._running:
            if len(self.buffer) >= 2:  # Wait for buffer to have at least 2 frames
                frame = self.buffer.popleft()
                await self.on_audio_chunk(frame, context=self.context)
                await asyncio.sleep(self.frame_duration)  # Real-time pacing
            else:
                await asyncio.sleep(0.005)  # Check buffer frequently
    
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
        logger.debug(f"Audio chunk: {len(audio_bytes)} bytes")
        
        # Play locally if requested
        if play_local:
            import numpy as np
            import sounddevice as sd
            audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
            sd.play(audio_array, 24000, blocking=True)
        
        # Use real-time pacer for SIP output
        if on_audio_chunk and context:
            session_id = context.log_id
            #8000 bytes = 1 second
            duration_seconds = len(audio_bytes) / 8000
            await on_audio_chunk(audio_bytes, context)
            await asyncio.sleep(duration_seconds * 0.92)
            # Create pacer if it doesn't exist
            #if session_id not in _audio_pacers:
            #    pacer = AudioPacer(frame_size=160, frame_duration_ms=20)
            #    await pacer.start_pacing(on_audio_chunk, context)
            #    _audio_pacers[session_id] = pacer
            #    logger.info(f"Started audio pacer for session {session_id}")
            
            # Add chunk to pacer (will be sent at real-time speed)
            #await _audio_pacers[session_id].add_chunk(audio_bytes)
            
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
    """Handle transcript from conversation events"""
    try:
        if 'transcript' in server_event:
            transcript = server_event['transcript']
            print("found uer transcript")
            await on_transcript('user', transcript, context=context)
            return
        item = server_event['item']

        role = item['role']
        transcript = None
        print("handle transcript") 
        # Extract transcript from content array for both roles

        for content_item in item.get('content', []):
            print("content_item", str(content_item))
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


async def handle_message(server_event, on_command, on_audio_chunk, on_transcript, on_interrupt, play_local, context):
    """Handle a single message from OpenAI"""
    try:
        event_type = server_event['type']
        logger.debug(f"Received server event: {event_type}")
        
        if event_type == "response.output_audio.delta":
            await handle_audio_delta(server_event, on_audio_chunk, play_local, context)
        elif event_type == "conversation.item.input_audio_transcription.completed":
            await handle_transcript(server_event, on_transcript, context)
        elif event_type == "input_audio_buffer.speech_started":
            # User interrupted - stop audio pacer immediately
            if context and context.log_id in _audio_pacers:
                logger.info(f"Interrupt detected - stopping audio pacer for {context.log_id}")
                await _audio_pacers[context.log_id].stop()
                del _audio_pacers[context.log_id]
            
            await on_interrupt(server_event)
        elif event_type == "conversation.item.done":
            item = server_event['item']
            if item['type'] == "function_call":
                await handle_function_call(item, on_command, context)
            elif item['type'] == "message":
                print("handling transcript:")
                print(str(item))
                await handle_transcript(server_event, on_transcript, context)
        else:
            # Log other message types for debugging
            logger.debug(f"Received message: {json.dumps(server_event, indent=2)}")
            
    except Exception as e:
        logger.error(f"Error handling message: {e}")
        traceback.print_exc()

async def message_handler_loop(ws, on_command, on_audio_chunk, on_transcript, on_interrupt, play_local, context):
    """Background task to handle incoming WebSocket messages"""
    try:
        async for message in ws:
            server_event = json.loads(message)
            await handle_message(server_event, on_command, on_audio_chunk, on_transcript, on_interrupt, play_local, context)
            
    except websockets.exceptions.ConnectionClosed:
        logger.info(f"WebSocket connection closed for {context.log_id}")
    except Exception as e:
        logger.error(f"Error in message handler loop: {e}")
        traceback.print_exc()
    finally:
        # Clean up audio pacer on session end
        if context and context.log_id in _audio_pacers:
            logger.info(f"Cleaning up audio pacer for {context.log_id}")
            await _audio_pacers[context.log_id].stop()
            del _audio_pacers[context.log_id]
