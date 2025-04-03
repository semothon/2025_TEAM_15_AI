from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv
from fastapi.responses import JSONResponse
import os
import json
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from pathlib import Path
import fitz
import re
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()
client = OpenAI()
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# 교과목 데이터 로딩
with open("AI/data/cse_courses.json", "r", encoding="utf-8") as f:
    curriculum = json.load(f)

def extract_gyoyang(text):
    def get_match(pattern):
        m = re.search(pattern, text)
        return int(m.group(1)) if m else 0

    교양 = {}

    # 필수교과
    필수_영역 = get_match(r"필수교과\s+(\d+)\s*/\s*3")
    필수_학점 = get_match(r"필수교과.*?(\d+)\s*/\s*17")
    교양["필수교과"] = {
        "영역": 필수_영역, "영역기준": 3,
        "학점": 필수_학점, "학점기준": 17,
        "판정": "통과" if 필수_영역 >= 3 and 필수_학점 >= 17 else "미통과"
    }

    # 배분이수
    배분_영역 = get_match(r"배분이수교과.*?(\d+)\s*/\s*3")
    배분_학점 = get_match(r"배분이수교과.*?(\d+)\s*/\s*9")
    교양["배분이수"] = {
        "영역": 배분_영역, "영역기준": 3,
        "학점": 배분_학점, "학점기준": 9,
        "판정": "통과" if 배분_영역 >= 3 and 배분_학점 >= 9 else "미통과"
    }

    # 자유이수
    자유_영역 = get_match(r"자유이수.*?(\d+)\s*/\s*2")
    자유_학점 = get_match(r"자유이수.*?(\d+)\s*/\s*3")
    교양["자유이수"] = {
        "영역": 자유_영역, "영역기준": 2,
        "학점": 자유_학점, "학점기준": 3,
        "판정": "통과" if 자유_영역 >= 2 and 자유_학점 >= 3 else "미통과"
    }

    return 교양

# PDF에서 전체 정보 추출
def extract_info_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = doc[0].get_text()

    result = {}

    result["학번"] = re.search(r"학\s*번\s*(\d{10})", text)
    result["이름"] = re.search(r"성\s*명\s*([가-힣]+)", text)
    result["학과"] = re.search(r"학\s*과\s*([가-힣]+)", text)

    total_match = re.search(r"취득\s+(\d+)\s+0\(\d+\)", text)
    if total_match:
        result["총학점_기준"] = 130
        result["총학점_취득"] = int(total_match.group(1))

    major_match = re.search(
        r"전공계:\s*\d+\s*\(\s*전필:\s*(\d+)\s*/\s*42\s*전선:\s*(\d+)\s*/\s*27\s*전기:\s*(\d+)\s*/\s*12\s*\)", text
    )
    if major_match:
        result["전공필수"] = int(major_match.group(1))
        result["전공선택"] = int(major_match.group(2))
        result["전공기초"] = int(major_match.group(3))
        result["산학필수"] = 0  # 명시 없음

    eng_match = re.search(r"영어강의\s*(\d)", text)
    result["영어강좌"] = int(eng_match.group(1)) if eng_match else 0

    result["졸업논문"] = "미통과" if "졸업능력인증" in text and "미취득" in text else "통과"

    sw_match = re.search(r"s소프트웨어적사유\s*3", text)
    result["SW교양"] = 3 if sw_match else 0

    result["최종판정"] = "졸업유예" if "최종판정 졸업유예" in text else "졸업"

    result["교양"] = extract_gyoyang(text)

    for k, v in result.items():
        if isinstance(v, re.Match):
            result[k] = v.group(1)

    return result

# 졸업요건 기준
def check_graduation_eligibility(student_info):
    기준 = {
        "총학점_기준": 130,
        "전공기초": 12,
        "전공필수": 42,
        "산학필수": 12,
        "전공선택": 15,
        "영어강좌": 3,
        "졸업논문": "통과",
        "SW교양": 6
    }

    부족항목 = []

    if student_info.get("총학점_취득", 0) < 기준["총학점_기준"]:
        부족항목.append({
            "항목": "총 이수학점",
            "기준": 기준["총학점_기준"],
            "취득": student_info.get("총학점_취득", 0)
        })

    for 항목 in ["전공기초", "전공필수", "산학필수", "전공선택", "SW교양", "영어강좌"]:
        취득 = student_info.get(항목, 0)
        if 취득 < 기준[항목]:
            부족항목.append({
                "항목": 항목,
                "기준": 기준[항목],
                "취득": 취득
            })

    if student_info.get("졸업논문", "미통과") != "통과":
        부족항목.append({
            "항목": "졸업논문",
            "기준": "통과",
            "취득": student_info.get("졸업논문")
        })

    # 항목 비교
    교양 = student_info.get("교양", {})
    for 항목, 값 in 교양.items():
        if 값["판정"] == "미통과":
            부족항목.append({
                "항목": f"교양 - {항목}",
                "기준": f"영역 {값['영역기준']}, 학점 {값['학점기준']}",
                "취득": f"영역 {값['영역']}, 학점 {값['학점']}"
            })

    졸업가능 = len(부족항목) == 0
    return {
        "졸업판정": "졸업 가능" if 졸업가능 else "졸업 불가",
        "부족항목": 부족항목
    }
# ---------- FastAPI 엔드포인트 ----------

@app.post("/analyze-pdf")
async def analyze_pdf(file: UploadFile = File(...)):
    temp_path = f"temp/{file.filename}"
    Path("temp").mkdir(exist_ok=True)
    with open(temp_path, "wb") as f:
        f.write(await file.read())

    info = extract_info_from_pdf(temp_path)
    result = check_graduation_eligibility(info)

    return JSONResponse(content={
        "학생정보": info,
        "졸업판정": result["졸업판정"],
        "부족항목": result["부족항목"]
    })


### 과목 추천 ###
import httpx
class RecommendRequest(BaseModel):
    keyword: str
    add_info: str

@app.post("/recommend")
async def recommend_courses(req: RecommendRequest):
    keyword = req.keyword.lower()

    related_courses = [
        c for c in curriculum
        if keyword in [k.lower() for k in c.get("keywords", [])]
    ]

    # 관련 과목이 없을 경우 일부 샘플 포함
    sample_courses = related_courses[:5] if related_courses else curriculum[:5]

    # context 텍스트 구성
    context_text = "\n".join([
        f"- {c['name']} ({c['category']}, {c['credit']}학점): {c.get('description', '설명 없음')}"
        for c in sample_courses
    ])

    prompt = f"""
    너는 경희대학교 소프트웨어융합대학의 AI 조교야.
    아래는 학생의 관심 분야와 배경 정보야:

    🔍 관심 분야: {req.keyword}
    ℹ️ 추가 정보: {req.add_info}

    아래 과목들 중에서 2~3개를 추천해줘. 
    각 추천 과목에는 간단한 이유를 붙여줘. 반드시 JSON 형식으로 응답해.

    과목 목록:
    {context_text}

    응답 형식 예시:
    {{
      "recommendations": [
        {{
          "name": "과목명",
          "reason": "간단한 추천 이유"
        }}
      ]
    }}
    """

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "너는 친절한 GPT 조교야. JSON 형식으로만 응답해."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.4
        )
        result_raw = response.choices[0].message.content.strip()
        return json.loads(result_raw)

    except json.JSONDecodeError:
        return {
            "raw_text": result_raw,
            "warning": "❗ JSON 파싱 실패! 수동 확인 필요"
        }
    except Exception as e:
        return {"error": str(e)}

### 💬 자유 질문 응답 ###
class ChatRequest(BaseModel):
    question: str

# 대화 히스토리 저장 (간단 테스트용 → 실제 운영 시 세션별 관리 필요)
conversation_history = []

class ChatRequest(BaseModel):
    question: str

@app.post("/chat")
async def answer_question(req: ChatRequest):
    question = req.question.strip()

    # 요약 텍스트: 토큰 절약을 위해 상위 50개만
    summarized = "\n".join([
        f"[{c['code'] or 'N/A'}] {c['name']} | {c['category']}, {c['credit']}학점, {c['year']}학년 | 키워드: {', '.join(c.get('keywords', []))}\n설명: {c.get('description', '설명 없음')}"
        for c in curriculum[:100]
    ])

    system_prompt = """
    너는 경희대학교 컴퓨터공학과의 GPT 기반 AI 조교야.
    아래는 컴퓨터공학과의 실제 커리큘럼 데이터 구조야.
    학생이 질문을 하면, 반드시 이 데이터 안의 정보만 기반으로 자연스럽고 친절하게 답변해.

    ⚠️ 지켜야 할 규칙:
    - 답변에 포함된 과목은 반드시 아래에 나온 과목 중에서만 골라.
    - 존재하지 않는 과목, 또는 데이터에 없는 내용은 답하지 마.
    - 질문이 모호하거나 불명확하면 되물어봐.
    - 과목 정보는 정확히 반영하되, 학생 눈높이에 맞게 설명해줘.
    """

    # 새로운 질문 추가
    conversation_history.append({"role": "user", "content": question})

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"📘 다음은 컴퓨터공학과의 과목 정보 요약이야:\n{summarized}"}
            ] + conversation_history,
            temperature=0.4
        )

        # GPT 응답 저장
        reply = response.choices[0].message.content.strip()
        conversation_history.append({"role": "assistant", "content": reply})

        return JSONResponse(content={"answer": reply})

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

