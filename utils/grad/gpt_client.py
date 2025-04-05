import os
from openai import OpenAI

# 보안상 환경변수로 API 키를 설정하는 것이 좋아요.
# 예: export OPENAI_API_KEY="your-key"
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def ask_gpt(prompt: str) -> str:
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "당신은 졸업 진단 전문가입니다."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            max_tokens=1000
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"[GPT 요청 오류] {str(e)}"
