from openai import OpenAI
import openai
import time

# API Keys List
API_KEYS = [
    "bab5a926-5245-4843-a03d-d98b57a0c644",
    "31a55a95-3f5c-483b-9a35-5fa473a6006a",
    "1fe618ed-bb32-408e-a481-57f9c477692f",
    "60c63e5e-0082-4fa6-a642-f8d678f61ddc",
    "f1d22e5a-d33e-4d1a-a3f9-c99e22749b0b"
]
api_key_index = 0

# 初始化 OpenAI 客户端
client = OpenAI(base_url="https://api.sambanova.ai/v1", api_key=API_KEYS[api_key_index])

def code_generation(messages, model="Meta-Llama-3.1-8B-Instruct", temperature=0, max_tokens=500,retries=100, delay=1):
    global api_key_index

    for attempt in range(retries):
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
                stream_options={"include_usage": True}
            )

            full_response = ""
            tokens = 0
            for chunk in completion:
                if chunk.choices:
                    delta = chunk.choices[0].delta
                    if hasattr(delta, 'content'):
                        full_response += delta.content
                if chunk.usage:
                    tokens += chunk.usage.completion_tokens

            return full_response, tokens

        except openai.RateLimitError:
            print(f"Request rate limit exceeded, switching API key...")
            api_key_index = (api_key_index + 1) % len(API_KEYS)  # change the API
            client.api_key = API_KEYS[api_key_index]  # update the API
            time.sleep(delay)  # wait
        except Exception as e:
            print(f"Error occurred while generating code, retrying after {delay} seconds...")
            time.sleep(delay)

    return None, 0

if __name__ == '__main__':
    prompt="hello, how are you "
    messages=[
        {"role": "system", "content": "Environment: ipython"},
        {"role": "user", "content": prompt}
    ],
    completion, tokens = code_generation(messages)
    print(completion)