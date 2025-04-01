from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv
import os
import json

load_dotenv()
client = OpenAI()
app = FastAPI()

class RecommendRequest(BaseModel):
    major: str
    student_id: str
    year: int
    interest: str

# 데이터 불러오기
with open("data/computer_engineering_courses_cleaned.json", "r", encoding="utf-8") as f:
    curriculum = json.load(f)

@app.post("/recommend")
async def recommend_course(req: RecommendRequest):
    # 관련 과목 추출
    related_courses = [
        course for course in curriculum
        if req.interest in course.get("keywords", [])
    ]

    # context 제한 (3.5는 길이에 민감하므로)
    sample_courses = related_courses[:5] if related_courses else curriculum[:5]

    context_text = "\n".join([
        f"- {c['name']} ({c.get('category', '분류없음')}): {c.get('description', '설명 없음')}"
        for c in sample_courses
    ])

    # 프롬프트 구성
    prompt = f"""
너는 경희대학교 소프트웨어융합대학의 AI 조교야.
학생의 전공, 학년, 관심 분야를 고려해서 수강 과목을 추천해줘.

전공: {req.major}
학번: {req.student_id}
학년: {req.year}
관심 분야: {req.interest}

추천 가능한 과목 리스트:
{context_text}

아래 형식을 꼭 지켜서 JSON으로 2~3개 추천해줘:

예시:
{{
  "recommendations": [
    {{
      "name": "웹서버프로그래밍",
      "reason": "HTTP 서버와 백엔드 로직을 실습 중심으로 학습할 수 있는 과목입니다."
    }},
    {{
      "name": "데이터베이스",
      "reason": "백엔드 개발자에게 필수인 SQL과 데이터 모델링을 배웁니다."
    }}
  ]
}}

조건:
- 반드시 위 과목 리스트 안에서만 골라
- 존재하지 않는 과목 추천하지 마
- JSON 형식을 꼭 지켜
"""

    # GPT 호출
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": "너는 경희대학교 AI 조교 KHUGPT야. 과목 추천을 JSON 형식으로 정리해서 친절하게 제공해야 해."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.4
        )

        result_raw = response.choices[0].message.content.strip()

        # JSON 파싱 시도
        result_json = json.loads(result_raw)
        return result_json

    except json.JSONDecodeError:
        return {
            "raw_text": result_raw,
            "warning": "❗ GPT 응답이 JSON 형식이 아닙니다. 수동 확인 필요."
        }
    except Exception as e:
        return {"error": str(e)}
    
class ChatRequest(BaseModel):
    question: str
@app.post("/chat")
async def chat_qna(req: ChatRequest):
    # 샘플 context 과목 5개만 사용 (3.5 최적화)
    sample_courses = curriculum[:5]
    context_text = "\n".join([
        f"{c['name']} - {c.get('description', '설명 없음')}"
        for c in sample_courses
    ])

    # 프롬프트 구성
    prompt = f"""
너는 경희대학교 소프트웨어융합대학 학생들을 도와주는 AI 조교야.
학생이 수강 계획, 졸업 요건, 과목 선택 등에 대해 자유롭게 질문할 수 있어.

아래는 커리큘럼 일부 정보야:
{context_text}

질문: "{req.question}"

위 정보를 바탕으로 경희대 기준으로 성실하고 정확하게 답변해줘.
"""

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": "너는 KHUGPT라는 이름의 AI 조교야. 학생의 질문에 정확하고 친절하게 답변해. 과목에 대한 설명은 curriculum context를 참고하고, 경희대학교 기준으로 말해줘."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.5
        )

        return {"answer": response.choices[0].message.content.strip()}

    except Exception as e:
        return {"error": str(e)}