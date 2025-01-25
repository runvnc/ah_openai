from lib.providers.services import service
import os
import base64
from io import BytesIO
from openai import AsyncOpenAI
import json

client = AsyncOpenAI(
    api_key=os.environ.get("OPENAI_API_KEY")
)

def concat_text_lists(message):
    """Concatenate text lists into a single string"""
    # if the message['content'] is a list
    # then we need to concatenate the list into a single string
    out_str = ""
    if isinstance(message['content'], str):
        return message
    else:
        for item in message['content']:
            if isinstance(item, str):
                out_str += item + "\n"
            else:
                out_str += item['text'] + "\n"
    message.update({'content': out_str})
    return message

@service()
async def stream_chat(model, messages=[], context=None, num_ctx=200000, 
                     temperature=0.0, max_tokens=5000, num_gpu_layers=0):
    try:
        
        model_name = os.environ.get("AH_OVERRIDE_LLM_MODEL", "o1-mini")
        
        messages = [concat_text_lists(m) for m in messages]

        if model_name == "o1-mini":
            messages[0]['role'] = "user"
            max_tokens = 20000
            tempeature = 1
        elif model_name.starts_with("o1"):
            messages[0]['irole'] = "developer"
            max_tokens = 20000
            temperature = 1
        print("model_name", model_name)

        stream = await client.chat.completions.create(
            model=model_name,
            messages=messages,
            stream=True,
            temperature=temperature,
            max_completion_tokens=max_tokens
        )

        print("Opened stream with model:", model_name)

        async def content_stream(original_stream):
            done_reasoning = False
            yield '[{"reasoning": "'
            async for chunk in original_stream:
                #if os.environ.get('AH_DEBUG') == 'True':
                #    #print('\033[93m' + str(chunk) + '\033[0m', end='')
                #    #print('\033[92m' + str(chunk.choices[0].delta.content) + '\033[0m', end='')
                if chunk.choices[0].delta.reasoning_content:
                    # we actually need to escape the reasoning_content but not convert it to full json
                    # i.e., it's a string, we don't want to add quotes around it
                    # but we need to escape it like a json string
                    json_str = json.dumps(chunk.choices[0].delta.reasoning_content)
                    without_quotes = json_str[1:-1]
                    yield without_quotes
                    print('\033[92m' + str(chunk.choices[0].delta.reasoning_content) + '\033[0m', end='')
                elif chunk.choices[0].delta.content:
                    if not done_reasoning:
                        yield '"}] <<CUT_HERE>>'
                        done_reasoning = True
                    yield chunk.choices[0].delta.content or ""

        return content_stream(stream)

    except Exception as e:
        print('OpenAI error:', e)
        #raise

@service()
async def format_image_message(pil_image, context=None):
    """Format image for DeepSeek using OpenAI's image format"""
    buffer = BytesIO()
    print('converting to base64')
    pil_image.save(buffer, format='PNG')
    image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    print('done')
    
    return {
        "type": "image_url",
        "image_url": {
            "url": f"data:image/png;base64,{image_base64}"
        }
    }

@service()
async def get_image_dimensions(context=None):
    """Return max supported image dimensions for DeepSeek"""
    return 4096, 4096, 16777216  # Max width, height, pixels
