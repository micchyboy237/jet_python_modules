from openai import OpenAI

def main():
    messages = [
        {
            "role": "user",
            "content": "Why is the sky blue?",
        },
    ]
    client = OpenAI(base_url="http://shawn-pc.local:8080/v1", api_key="sk-1234")  # Dummy API key
    response = client.chat.completions.create(
        model="qwen3-instruct-2507:4b",
        messages=messages,
    )
    print(response.choices[0].message.content)

if __name__ == "__main__":
    main()