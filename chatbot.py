import os
from openai import OpenAI
import streamlit as st

#환경변수 읽기
from dotenv import load_dotenv
load_dotenv()

st.title("Streamlit Test")
st.write("hello world")
st.write("""
# MarkDown
> comment
- one
- two
- three
""")

# print(os.environ.get("OPENAI_API_KEY"))

client =OpenAI(api_key=os.environ.get("OPENAI_API_KEY"),)
# OpenAI.api_key = os.environ.get("OPENAI_API_KEY")

completion = client.chat.completions.create(
  model="gpt-3.5-turbo",
  messages=[
    {"role": "system", "content": "한국어로 대답하는 개인비서"},
    {"role": "user", "content": "월요일날 회사 가기 싫은데 좋은 방법은?"}
  ]
)

st.write(completion.choices[0].message)
