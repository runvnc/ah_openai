import base64
import json
import struct
import soundfile as sf
import traceback
# for playing local audio
import time
import asyncio
import os
import websocket
import sounddevice as sd
import nanoid
from lib.providers.services import service


import logging
openai_sockets = {}
# Store event loops for each session
openai_loops = {}

def float_to_16bit_pcm(float32_array):
    clipped = [max(-1.0, min(1.0, x)) for x in float32_array]
    pcm16 = b''.join(struct.pack('<h', int(x * 32767)) for x in clipped)
    return pcm16

def base64_encode_audio(float32_array):
    pcm_bytes = float_to_16bit_pcm(float32_array)
    encoded = base64.b64encode(pcm_bytes).decode('ascii')
    return encoded

logger = logging.getLogger(__name__)

files = [
    #'/files/upd6/mr_verification_dashboard/audio/voicemailpadded.wav',
    '/files/upd6/mr_verification_dashboard/audio/voicemailpadded2_24000_pcm.wav'
]

def send_wavs(ws):
    try:
        print("Top of send_wavs")
        for filename in files:
            data, samplerate = sf.read(filename, dtype='float32')
            channel_data = data[:, 0] if data.ndim > 1 else data
            #base64_chunk = base64.b64encode(channel_data.tobytes()).decode('ascii')
            base64_chunk = base64_encode_audio(channel_data)

            # Send the client event
            event = {
                "type": "input_audio_buffer.append",
                "audio": base64_chunk
            }
            print("Sending audio data")
            ws.send(json.dumps(event))
            print('sent audio chunk')
    except Exception as e:
        trace = traceback.format_exc()
        print(e)
        print(trace)


@service()
async def start_s2s(model, system_prompt, on_command, on_audio_chunk=None, voice='marin',
                    play_local=True, context=None, **kwargs):
    """
        Start a speech-to-speech OpenAI realtime websocket session.
        Session will be identified by context.log_id

        Arguments:

            model: model name, e.g. 'gpt-realtime'

            system_prompt: system prompt string

            on_command: async callback function to handle function call commands from the server.
                        Arg 1: function name, arg 2: function parameters dict.

            on_audio_chunk: async callback function to handle audio chunks from the server.
                            Arg: audio bytes in int16 PCM format at 24000 Hz sample rate.

            voice: OpenAI voice to use (alloy, echo, fable, onyx, nova, shimmer)

            play_local: whether to play audio locally (default True)

    """
    global openai_sockets
    global openai_loops
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

    url = "wss://api.openai.com/v1/realtime?model=gpt-realtime"
    headers = ["Authorization: Bearer " + OPENAI_API_KEY]

    if model is None:
        model = 'gpt-realtime'
        
    async def on_message(ws, message):
        try:
            logger.debug(f"S2S_DEBUG: Received message from OpenAI")
            server_event = json.loads(message)
            print(f"Received server event: {server_event['type']}")
            if server_event['type'] == "response.output_audio.delta":
                print()
                print("Audio chunk received !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                print()
                print()
                audio_bytes = base64.b64decode(server_event['delta'])
                print(f"Audio chunk: {len(audio_bytes)} bytes, first 10 bytes: {audio_bytes[:10].hex()}")
                
                # Play locally if requested
                if play_local:
                    logger.info("S2S_DEBUG: Playing audio chunk locally")
                    print("Playing audio chunk locally...")
                    import numpy as np
                    audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
                    sd.play(audio_array, 24000, blocking=True)
                
                # Call the audio chunk callback if provided
                if on_audio_chunk:
                    #logger.debug(f"S2S_DEBUG: Calling on_audio_chunk callback with {len(audio_bytes)} bytes")
                    await on_audio_chunk(audio_bytes, context=context)
 

            elif server_event['type'] == "conversation.item.done":
                print("Conversation item done received")
                item = server_event['item']
                if item['type'] == "function_call":
                    print("Function call received:")
                    try:
                        arguments = json.loads(item['arguments'])
                    except Exception as e:
                        print("Error:",e)
                        raise Exception("Error:" + str(e))
                    try:
                        if item['name'] != 'output':
                            cmd_name = item['name']
                            args = json.loads(item['arguments'])
                            cmd = {}
                            cmd[cmd_name] = args
                            print("Invoking on_command callback for command:", str(cmd))
                            await on_command(cmd, context=context)
                        else:
                            cmd = json.loads(arguments['text'])
                            # if this is a list, loop over it
                            if isinstance(cmd, list):
                                for single_cmd in cmd:
                                    print("Invoking on_command callback for command:", str(single_cmd))
                                    await on_command(single_cmd, context=context)
                            else:
                                print("Invoking on_command callback for command:", str(cmd))
                                await on_command(cmd, context=context)
                    except Exception as e:
                        print("Error in on_command callback:")
                        trace = traceback.format_exc()
                        print(e)
                        print(trace)
                        asyncio.sleep(0.5)
                        err_content = ([{
                            "type": "text",
                            "text": f"[SYSTEM: Error executing command: {str(e)}\n{str(trace)}]"
                        }])
                        error_msg = { "role": "user", "content": content }
                        await send_s2s_message(error_msg, context=context)
                        #raise e
            else:
                print("received message:")
                print(message)
        except Exception as e:
            trace = traceback.format_exc()
            print(e)
            print(trace)
            raise e

    def on_message_(ws, message):
        # WebSocket runs in a thread pool, so we need to use the stored loop reference
        try:
            # Get the loop we stored when starting the session
            loop = openai_loops.get(context.log_id)
            if not loop:
                raise Exception(f"No event loop found for session {context.log_id}")
            asyncio.run_coroutine_threadsafe(on_message(ws, message), loop)
        except Exception as e:
            logger.error(f"Error in on_message_: {e}")


    def on_open(ws):
        try: 
            print("OpenAI realtime websocket connected to server.")
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
                            "transcription": {
                                "language": "en",
                                "model": "gpt-4o-transcribe"
                            },
                            "turn_detection": {
                                "type": "semantic_vad",
                                "eagerness": "medium",
                                "create_response": True, 
                                "interrupt_response": True
                            }
                        },
                        "output" : { 
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
            #"event_id": "5fc543c4-f59c-420f-8fb9-68c45d1546a7a2"
            }
            ws.send(json.dumps(session_update))
            #send_wavs(ws)
            print("OpenAI realtime initialized session.")
        except Exception as e:
            trace = traceback.format_exc()
            print(e)
            print(trace)

    ws = websocket.WebSocketApp(
        url,
        header=headers,
        on_open=on_open,
        on_message=on_message_,
    )
    openai_sockets[context.log_id] = ws
    # Store the current event loop for this session
    openai_loops[context.log_id] = asyncio.get_event_loop()
    loop = asyncio.get_event_loop()
    loop.run_in_executor(None, ws.run_forever)



@service()
async def send_s2s_message(message, context=None):
    global openai_sockets
    ws = openai_sockets.get(context.log_id)
    if not ws:
        raise Exception(f"No active OpenAI socket for log_id {context.log_id}")
    parts = [] 
    for item in message['content']:
        print(f"item is {item}")
        part = {}
        if item['type'] == 'text':
            part['type'] = 'input_text'
            part['text'] = item['text']
            parts.append(part)
        else:
            raise Exception("OpenAI s2s Unimplemented content part in message")

    event = {
        "type": "conversation.item.create",
        "item": {
            "type": "message",
            "role": "user",
            "content": parts
        },
        "event_id": nanoid.generate()
    }
    ws.send(json.dumps(event))
    ws.send(json.dumps({"type": "response.create"}))
    print(f"Sent message to OpenAI s2s: {event}")
 

@service()
async def send_s2s_audio_chunk(audio_bytes, context=None):
    """
        Send an audio chunk to OpenAI for processing.
        context.log_id identifies the session.

        audio_bytes: bytes of audio data (ulaw format from PySIP)
    """
    try:
        if not hasattr(send_s2s_audio_chunk, '_chunk_count'):
            send_s2s_audio_chunk._chunk_count = 0
        global openai_sockets
        
        # Audio is already in ulaw format from PySIP, just base64 encode it
        base64_chunk = base64.b64encode(audio_bytes).decode('ascii')
        event = {
            "type": "input_audio_buffer.append",
            "audio": base64_chunk
        }
        ws = openai_sockets.get(context.log_id)
        #logger.debug(f"S2S_DEBUG: send_s2s_audio_chunk called, log_id={context.log_id if context else None}")
        #logger.debug(f"S2S_DEBUG: WebSocket exists: {ws is not None}")
        
        if ws:
            send_s2s_audio_chunk._chunk_count += 1
            ws.send(json.dumps(event))
            #if send_s2s_audio_chunk._chunk_count % 50 == 0:
            #    #logger.info(f"S2S_DEBUG: Sent {send_s2s_audio_chunk._chunk_count} audio chunks to OpenAI")
        else:
            logger.error(f"S2S_DEBUG: No active OpenAI socket for log_id {context.log_id}")
            raise Exception(f"No active OpenAI socket for log_id {context.log_id}")
    except Exception as e:
        trace = traceback.format_exc()
        print(e)
        print(trace)
        raise e
