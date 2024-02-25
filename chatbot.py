from openai import OpenAI
client = OpenAI()

completion = client.chat.completions.create(
  model="gpt-3.5-turbo",
  messages=[
    {"role": "system", "content": "한국어로 대답하는 개인비서"},
    {"role": "user", "content": "월요일날 회사 가기 싫은데 좋은 방법은?"}
  ]
)

print(completion.choices[0].message)