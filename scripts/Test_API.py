from openai import OpenAI

client = OpenAI(
    base_url="https://css-exciting-earned-announcement.trycloudflare.com/v1",
    api_key="dummy"
)

resp = client.chat.completions.create(
    model="dummy",
    messages=[{"role": "user", "content": "血糖8"}]
)
print(resp.choices[0].message.content)
