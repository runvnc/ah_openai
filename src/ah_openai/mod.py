from lib.providers.services import service
import os
import openai

openai.api_key = os.environ.get("OPENAI_API_KEY")
 
client = openai.AsyncOpenAI()

@service()
async def stream_chat(model, messages=[], context=None, num_ctx=2048, temperature=0.0, max_tokens=3724, num_gpu_layers=12):
    # print in blue background with white text
    print('\033[44m' + 'stream_chat called' + '\033[0m')
    print("model", model)
    try:
        if model == 'o1-preview' or model == 'o1-mini':
            content = await sync_chat_o1(model, messages)

            async def content_stream_():
                yield content

            return content_stream_()

        if not model or model == '':
            model = 'chatgpt-4o-latest'
        print("model = ", model)
        stream = await client.chat.completions.create(
            model=model, #"chatgpt-4o-latest",
            #model="o1-preview",
            stream=True,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        print("Opened stream with model: ", model)
        async def content_stream(original_stream):
            async for chunk in original_stream:
                if os.environ.get('AH_DEBUG') == 'True':
                    print('\033[92m' + str(chunk.choices[0].delta.content) + '\033[0m', end='')

                yield chunk.choices[0].delta.content or ""

        return content_stream(stream)

    except Exception as e:
        print('openai error:', e)


from openai import OpenAI
sync_client = OpenAI()

async def sync_chat_o1(model, messages):
    messages_copy = messages.copy()
    messages_copy[0]['role'] = 'user'
    # print in blue background with white text
    print('\033[44m' + 'calling ' + model + '\033[0m')
    print("calling ", model)
    print("messages_copy", messages_copy)
    response = sync_client.chat.completions.create(
        model = model,
        messages = messages_copy
    )
    print("response from o1-preview received:")
    response = response.choices[0].message.content
    print(response)
    return response

