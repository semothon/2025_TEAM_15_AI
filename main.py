from fastapi import FastAPI, UploadFile, File, Form
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from typing import List, Optional, Union
from openai import OpenAI
from dotenv import load_dotenv
import json
from utils.time.image_parser import get_schedule_mask
from utils.time.visualize import visualize_free_mask
from utils.grad.analyzer import analyze_graduation_pdf
import numpy as np

# 환경 변수 로딩
load_dotenv()
client = OpenAI()

# 과목 데이터 로딩
with open("data/cse_course.json", "r", encoding="utf-8") as f:
    curriculum = json.load(f)

# FastAPI 앱 초기화
app = FastAPI()

# 요청 모델
class RecommendRequest(BaseModel):
    keyword: str
    add_info: str

# 추천 항목 모델
class RecommendationItem(BaseModel):
    title: str
    description: str

# 응답 모델
class RecommendResponse(BaseModel):
    keyword: str
    add_info: str
    ai_response: str  # 문자열 응답으로 구성

# 에러 응답 모델
class ErrorResponse(BaseModel):
    error: str
    raw_text: Optional[str] = None
    warning: Optional[str] = None

class ChatRequest(BaseModel):
    question: str

class ChatResponse(BaseModel):
    question: str
    ai_add_response: str

class DeficiencyItem(BaseModel):
    항목: str
    부족_내용: str
    보완_방법: str

class GraduationResult(BaseModel):
    졸업_가능: bool
    부족_항목: Optional[List[DeficiencyItem]]
    종합_의견: str
    검증_부족항목: Optional[List[str]] = None
##### 과목 추천 #####
@app.post("/recommend", response_model=Union[RecommendResponse, ErrorResponse])
async def recommend(request: RecommendRequest):
    keyword = request.keyword.lower()

    # 관련 키워드 과목 필터링
    related = [c for c in curriculum if keyword in [k.lower() for k in c.get("keywords", [])]]
    samples = related[:5] if related else curriculum[:5]

    # 프롬프트 구성
    context_text = "\n".join([
        f"- {c['name']} ({c['category']}, {c['credit']}학점): {c.get('description', '설명 없음')}"
        for c in samples
    ])

    prompt = f"""
    너는 대학생에게 과목을 추천해주는 AI 조교야.
    아래 사용자 정보와 과목 목록을 참고해서 3개 과목을 추천해줘.
    반드시 다음 JSON 형식으로만 응답해:

    {{
    "recommendations": [
        {{
        "title": "과목명",
        "description": "설명"
        }},
        ...
    ]
    }}

🔍 관심 분야: {request.keyword}
ℹ️ 추가 정보: {request.add_info}
과목 목록:
{context_text}
    """

    try:
        # OpenAI API 호출
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "너는 AI 조교야. 반드시 JSON만으로 답변해."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.4
        )

        result_raw = response.choices[0].message.content.strip()
        print("🧠 AI 응답 원문:\n", result_raw)

        result_json = json.loads(result_raw)
        recommendations = result_json.get("recommendations", [])

        # 문자열로 응답 포맷팅
        formatted_response = "\n".join([
            f"{i+1}. {r['title']} - {r['description']}" for i, r in enumerate(recommendations)
        ])

        return RecommendResponse(
            keyword=request.keyword,
            add_info=request.add_info,
            ai_response=formatted_response
        )

    except json.JSONDecodeError:
        return ErrorResponse(
            error="JSON 파싱 실패",
            raw_text=result_raw,
            warning="AI 응답을 JSON으로 변환하지 못했습니다."
        )

    except Exception as e:
        return ErrorResponse(error=str(e))

##### 질의응답 #####
conversation_history = []
@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    question = req.question.strip()

    # 과목 데이터 요약 (최대 100개만)
    summarized = "\n".join([
        f"[{c['code'] or 'N/A'}] {c['name']} | {c['category']}, {c['credit']}학점, {c['year']}학년 | 키워드: {', '.join(c.get('keywords', []))}\n설명: {c.get('description', '설명 없음')}"
        for c in curriculum[:1000]
    ])

    # AI 시스템 프롬프트
    system_prompt = """
    너는 경희대 컴퓨터공학과 GPT 조교야.
    학생이 커리큘럼과 관련된 질문을 하면 친절하고 정확하게 답해줘.
    존재하지 않는 정보는 "정보가 없습니다"라고 말해.
    """

    # 대화 히스토리에 질문 추가
    conversation_history.append({"role": "user", "content": question})

    try:
        # OpenAI API 호출
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": summarized}
            ] + conversation_history,
            temperature=0.4
        )

        reply = response.choices[0].message.content.strip()
        conversation_history.append({"role": "assistant", "content": reply})

        # ChatResponse 형식으로 응답
        return ChatResponse(
            question=question,
            ai_add_response=reply
        )

    except Exception as e:
        return JSONResponse(
            content={"error": str(e)},
            status_code=500
        )


from fastapi import UploadFile, File, Query
from fastapi.responses import JSONResponse

@app.post("/analyze-pdf")
async def analyze_pdf(
    major: str = Query(..., description="학과"),
    student_id: str = Query(..., description="학번"),
    file: UploadFile = File(...)
):
    result = await analyze_graduation_pdf(file, major, student_id)
    return {"result": result}


##### 공강 시간표 #####
@app.post("/timetable")
async def analyze_schedule_image(files: list[UploadFile]):
    masks = []

    for file in files:
        save_path = f"AI/data/시간표/{file.filename}"
        with open(save_path, "wb") as f:
            f.write(await file.read())

        mask = get_schedule_mask(save_path)
        masks.append(mask)

    # 공강 마스크 계산 (모두 수업 없을 때만 공강)
    combined = np.stack(masks, axis=0)
    free_mask = np.all(combined == 0, axis=0)

    result_path = "AI/data/공통공강22.png"
    visualize_free_mask(free_mask, result_path)

    return {
        "message": "공강 시각화 완료",
        "image": result_path
    }