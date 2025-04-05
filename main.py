from fastapi import FastAPI, UploadFile, File, Query, Form
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from typing import List, Optional, Union
from openai import OpenAI
from dotenv import load_dotenv
import json
from utils.time.image_parser import get_schedule_mask
from utils.time.visualize import visualize_free_mask
from utils.grad.analyzer import analyze_graduation_pdf
from utils.grad.pdf_parser import parse_pdf
import numpy as np
from typing import Dict, Any
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

import os
import shutil

class AnalyzePdfResponse(BaseModel):
    responseData: Dict[str, Any]

@app.post("/analyze-pdf", response_model=AnalyzePdfResponse)
async def analyze_pdf(
    file: UploadFile = File(..., description="졸업 진단표 PDF 파일"),
    department: str = Form(..., description="학과명"),
    studentId: str = Form(..., description="학번")
):
    try:
        # 1. 파일 형식 확인
        if file.content_type != "application/pdf":
            return AnalyzePdfResponse(responseData={
                "error": "PDF 파일만 업로드 가능합니다.",
                "fileName": file.filename
            })

        # 2. 파일 내용 읽기
        content = await file.read()

        # 3. 졸업 진단 실행
        result = await analyze_graduation_pdf(content, department, studentId)

        # 4. 분석 결과 감싸기
        if isinstance(result, dict):
            analysis_result = result
        else:
            analysis_result = {"result": result}

        # 5. 공통 메타 정보 + 분석 결과 포함
        return AnalyzePdfResponse(responseData={
            "studentId": studentId,
            "department": department,
            "fileName": file.filename,
            "message": "FastAPI 업로드 및 졸업 진단 성공 ✅",
            "analysis": analysis_result
        })

    except Exception as e:
        return AnalyzePdfResponse(responseData={
            "error": f"서버 오류: {str(e)}"
        })
##### 공강 시간표 #####

import cv2
import numpy as np
import pytesseract
import re
import os
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
from tempfile import NamedTemporaryFile
pytesseract.pytesseract.tesseract_cmd = "/opt/homebrew/bin/tesseract"

# ✅ 전처리 함수: 크기 조절 + ROI 추출
def preprocess_and_resize(img):
    large = cv2.resize(img, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
    rgb = cv2.pyrDown(large)
    gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 3)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    grad = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel)
    _, bw = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    connected = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 1)))

    contours, _ = cv2.findContours(connected.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    six_y = None
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w < 5 or h < 10:
            continue

        pad = 2
        x_pad = max(x - pad, 0)
        y_pad = max(y - pad, 0)
        w_pad = min(w + pad * 2, rgb.shape[1] - x_pad)
        h_pad = min(h + pad * 2, rgb.shape[0] - y_pad)

        roi = gray[y_pad:y_pad + h_pad, x_pad:x_pad + w_pad]
        roi_resized = cv2.resize(roi, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
        text = pytesseract.image_to_string(roi_resized, config='--psm 10 -c tessedit_char_whitelist=0123456789').strip()

        if text == '6':
            six_y = y_pad

    cutoff_y = six_y + 80 if six_y is not None else rgb.shape[0]
    cropped = rgb[:cutoff_y, :]
    resized = cv2.resize(cropped, (787, 1420), interpolation=cv2.INTER_AREA)
    return resized

# ✅ 공통 공강 마스크 생성 및 하이라이팅
def find_common_free_slots(img1, img2, start_x=40, start_y=36, end_x=787, end_y=1420):
    region1 = img1[start_y:end_y, start_x:end_x]
    region2 = img2[start_y:end_y, start_x:end_x]

    # 비어있는 시간 = 밝은 색 (흰색 or 회색 계열)
    mask1 = ((region1 == [17, 17, 17]).all(axis=2)) | ((region1 == [255, 255, 255]).all(axis=2))
    mask2 = ((region2 == [17, 17, 17]).all(axis=2)) | ((region2 == [255, 255, 255]).all(axis=2))
    common_mask = mask1 & mask2

    region1[common_mask] = [0, 255, 255]  # 노란색으로 공강 표시
    region2[common_mask] = [0, 255, 255]

    img1[start_y:end_y, start_x:end_x] = region1
    img2[start_y:end_y, start_x:end_x] = region2
    return img1, img2

# ✅ FastAPI 엔드포인트
@app.post("/timetable")
async def highlight_timetables(
    file1: UploadFile = File(...),
    file2: UploadFile = File(...)
):
    try:
        # 1. 임시 파일 저장
        temp1 = NamedTemporaryFile(delete=False, suffix=".jpg")
        temp2 = NamedTemporaryFile(delete=False, suffix=".jpg")
        temp1.write(await file1.read())
        temp2.write(await file2.read())
        temp1.close()
        temp2.close()

        # 2. 이미지 로딩
        img1 = cv2.imread(temp1.name)
        img2 = cv2.imread(temp2.name)

        if img1 is None or img2 is None:
            return JSONResponse(status_code=400, content={
                "resultCode": "ERROR",
                "message": "이미지 로드 실패. JPEG 형식인지 확인해주세요.",
                "data": None
            })

        # 3. 전처리 및 공강 분석
        processed1 = preprocess_and_resize(img1)
        processed2 = preprocess_and_resize(img2)
        highlighted1, highlighted2 = find_common_free_slots(processed1.copy(), processed2.copy())

        # 4. 결과 저장 디렉토리
        result_dir = "output"
        os.makedirs(result_dir, exist_ok=True)

        # 5. 저장 경로 생성
        out_path1 = os.path.join(result_dir, f"highlighted_{file1.filename}")
        out_path2 = os.path.join(result_dir, f"highlighted_{file2.filename}")

        cv2.imwrite(out_path1, highlighted1)
        cv2.imwrite(out_path2, highlighted2)

        # 6. 스프링에 맞는 응답 구조
        return {
            "resultCode": "SUCCESS",
            "message": "공통 공강 시각화 완료",
            "data": {
                "highlightedTimetable1": out_path1,
                "highlightedTimetable2": out_path2
            }
        }

    except Exception as e:
        return JSONResponse(status_code=500, content={
            "resultCode": "ERROR",
            "message": f"FastAPI 처리 오류: {str(e)}",
            "data": None
        })
