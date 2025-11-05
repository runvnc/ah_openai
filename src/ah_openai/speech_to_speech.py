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


openai_sockets = {}

def float_to_16bit_pcm(float32_array):
    clipped = [max(-1.0, min(1.0, x)) for x in float32_array]
    pcm16 = b''.join(struct.pack('<h', int(x * 32767)) for x in clipped)
    return pcm16

def base64_encode_audio(float32_array):
    pcm_bytes = float_to_16bit_pcm(float32_array)
    encoded = base64.b64encode(pcm_bytes).decode('ascii')
    return encoded

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
        Session will be identifiedby context.log_id

        Arguments:

            model: model name, e.g. 'gpt-realtime'

            system_prompt: system prompt string

            on_command: async callback function to handle function call commands from the server.
                        Arg 1: function name, arg 2: function parameters dict.

            on_audio_chunk: async callback function to handle audio chunks from the server.
                            Arg: audio bytes in float32 PCM format at 24000 Hz sample rate.

    """
    global openai_sockets
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

    url = "wss://api.openai.com/v1/realtime?model=gpt-realtime"
    headers = ["Authorization: Bearer " + OPENAI_API_KEY]

    if model is None:
        model = 'gpt-realtime'
        
    async def on_message(ws, message):
        try:
            server_event = json.loads(message)
            if server_event['type'] == "response.output_audio.delta":
                print()
                print("Audio chunk received !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                print()
                print()
                audio_bytes = base64.b64decode(server_event['delta'])
                if play_local:
                    # Convert bytes to numpy array for sounddevice
                    import numpy as np
                    audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
                    # Play and wait for completion
                    sd.play(audio_array, 24000, blocking=True)
                #if on_audio_chunk:
                #    await on_audio_chunk(audio_bytes, context=context)
 

            elif server_event['type'] == "conversation.item.done":
                print("Conversation item done received")
                item = server_event['item']
                if item['type'] == "function_call":
                    print("Function call received:")
                    arguments = json.loads(item['arguments'])
                    try:
                        cmd = json.loads(arguments['text'])
                        await on_command(cmd, context=context)
                    except json.JSONDecodeError:
                        pass
                    except Exception as e:
                        print("Error in on_command callback:")
                        trace = traceback.format_exc()
                        print(e)
                        print(trace)
                        raise e
            else:
                print("received message:")
                print(message)
        except Exception as e:
            trace = traceback.format_exc()
            print(e)
            print(trace)

    def on_message_(ws, message):
        asyncio.run(on_message(ws, message))


    def on_open(ws):
        try: 
            print("OpenAI realtime websocket connected to server.")
            session_update = {
                "type": "session.update",
                "session": {
                    "type": "realtime",
                    "instructions": system_prompt,
                    "audio": {"output" : { "voice": voice} },
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
            send_wavs(ws)
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
    ws.send(json.dumps({"type": "response.create"})
    print(f"Sent message to OpenAI s2s: {event}")
 

@service()
async def send_s2s_audio_chunk(audio_bytes, context=None):
    """
        Send an audio chunk to the server for processing.
        context.log_id identifies the session.

        audio_bytes: bytes of audio data in float 32 PCM format
                     at 24000 Hz sample rate.
    """
    global openai_sockets
    float32_array = struct.unpack('<' + 'f' * (len(audio_bytes) // 4), audio_bytes)
    base64_chunk = base64_encode_audio(float32_array)
    event = {
        "type": "input_audio_buffer.append",
        "audio": base64_chunk
    }
    ws = openai_sockets.get(context.log_id)
    if ws:
        ws.send(json.dumps(event))
    else:
        raise Exception(f"No active OpenAI socket for log_id {context.log_id}")
