from fastapi import FastAPI, UploadFile, File, Query, Form
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from typing import List, Optional, Union
from openai import OpenAI
from dotenv import load_dotenv
import json
from utils.grad.analyzer import analyze_graduation_pdf
import numpy as np
from typing import Dict, Any
# 환경 변수 로딩
load_dotenv()
client = OpenAI()
user_interest_memory = []

curriculum = []

for path in [
    "data/cse_course.json",             # 컴퓨터공학과
    "data/sw_courses.json",   # 소융
    "data/ai_courses.json"         # 인지
]:
    with open(path, "r", encoding="utf-8") as f:
        curriculum += json.load(f)

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

class TimetableResponseDto(BaseModel):
    imageUrl: str
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
    아래 사용자 정보와 과목 목록을 참고해서 반드시 그 목록 안에서만 3개의 과목을 추천해줘.
    목록에 없는 과목은 절대 추천하면 안 돼.
    응답은 반드시 다음 JSON 형식으로, 내용은 모두 한국어로 작성해:

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
    과목 목록:{context_text}
    """

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "너는 AI 조교야. 반드시 JSON만으로 답변해."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2
        )

        result_raw = response.choices[0].message.content.strip()
        result_json = json.loads(result_raw)
        recommendations = result_json.get("recommendations", [])

        # 📌 관심사 저장
        user_interest_memory.append({
            "keyword": request.keyword,
            "recommendations": recommendations
        })

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

    # 과목 요약
    summarized = "\n".join([
        f"[{c['code']}] {c['name']} | {c['category']}, {c['credit']}학점, {c['year']}학년 | 선수과목: {c['prerequisite'] or '없음'}"
        for c in curriculum[:100]
    ])

    # 📌 이전 추천 이력 요약
    interest_summary = "\n".join([
        f"- 관심 키워드: {m['keyword']} → 추천 과목: {', '.join([r['title'] for r in m['recommendations']])}"
        for m in user_interest_memory
    ]) or "없음"

    # 시스템 프롬프트 구성
    system_prompt = f"""
    너는 경희대 컴퓨터공학과 GPT 조교야.
    학생의 질문에 대해 커리큘럼 기반으로 친절하고 정확하게 답해줘.

    📌 이 사용자가 물어본 내용들을 기억해줘:
    {interest_summary}

    과목 추천이 필요할 땐 반드시 아래 과목 목록 중에서만 골라.
    존재하지 않는 정보는 "정보가 없습니다"라고 정확히 말해.
    절대로 영어를 사용하지 마. 모든 문장은 반드시 **한국어로만** 작성해.
    """

    conversation_history.append({"role": "user", "content": question})

    try:
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
        # # 1. 파일 형식 확인
        # if file.content_type != "application/pdf":
        #     return AnalyzePdfResponse(responseData={
        #         "error": "PDF 파일만 업로드 가능합니다.",
        #         "fileName": file.filename
        #     })

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
            # "studentId": studentId,
            # "department": department,
            # "fileName": file.filename,
            # "message": "FastAPI 업로드 및 졸업 진단 성공 ✅",
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

# 텍스트 인식 후 이미지 크기 조정
def preprocess_and_resize(img):
    large = cv2.resize(img, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
    rgb = cv2.pyrDown(large)
    gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 3)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    grad = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel)

    _, bw = cv2.threshold(grad, 0.0, 255.0, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 1))
    connected = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(connected.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    six_y = None
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w < 5 or h < 10:
            continue

        pad_w, pad_h = 2, 2
        x_pad = max(x - pad_w, 0)
        y_pad = max(y - pad_h, 0)
        w_pad = min(w + pad_w * 2, rgb.shape[1] - x_pad)
        h_pad = min(h + pad_h * 2, rgb.shape[0] - y_pad)

        roi = gray[y_pad:y_pad + h_pad, x_pad:x_pad + w_pad]
        roi_resized = cv2.resize(roi, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
        text = pytesseract.image_to_string(roi_resized, config='--psm 10 -c tessedit_char_whitelist=0123456789').strip()

        if re.fullmatch(r'\d', text) and text == '6':
            six_y = y_pad

    if six_y is not None:
        cutoff_y = six_y + 80
        cropped = rgb[:cutoff_y, :]
    else:
        cropped = rgb

    resized = cv2.resize(cropped, (787, 1420), interpolation=cv2.INTER_AREA)
    return resized

# 숫자 '7'을 찾고 그 아래의 영역을 삭제하는 함수
def find_and_remove_below_seven(img, start_x=40, start_y=36, end_x=787, end_y=1420):
    region = img[start_y:end_y, start_x:end_x]
    
    # 숫자 7을 인식하여 그 위치를 찾기
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    text = pytesseract.image_to_string(gray, config='--psm 6').strip()

    # 숫자 7을 찾고 위치 파악
    seven_y = None
    for cnt in region:
        if '7' in text:
            # '7' 위치 찾기
            seven_y = text.index('7')
            break
    
    if seven_y is not None:
        # 숫자 7 아래 부분을 삭제
        img = img[:seven_y, :] # 숫자 7보다 아래 부분 삭제

    return img

# 두 이미지에서 공통 공강 시간 부분을 초록색으로 하이라이팅
def find_common_free_slots(img1, img2, start_x=40, start_y=36, end_x=787, end_y=1420):
    region1 = img1[start_y:end_y, start_x:end_x]
    region2 = img2[start_y:end_y, start_x:end_x]

    # 두 이미지에서 동일한 영역을 찾기 위한 조건
    mask1 = ((region1 == [17, 17, 17]).all(axis=2)) | ((region1 == [255, 255, 255]).all(axis=2))
    mask2 = ((region2 == [17, 17, 17]).all(axis=2)) | ((region2 == [255, 255, 255]).all(axis=2))

    common_mask = mask1 & mask2

    # 공통 부분을 초록색으로 칠하기
    region1[common_mask] = [0, 255, 0]  # 초록색
    region2[common_mask] = [0, 255, 0]  # 초록색

    # 변경된 이미지를 원본 이미지에 반영
    img1[start_y:end_y, start_x:end_x] = region1
    img2[start_y:end_y, start_x:end_x] = region2

    return img1, img2
@app.post("/timetable", response_model=TimetableResponseDto)
async def highlight_timetables(images: List[UploadFile] = File(...)):
    try:
        # 임시 저장
        temp_paths = []
        for image in images:
            temp = NamedTemporaryFile(delete=False, suffix=".jpg")
            temp.write(await image.read())
            temp.close()
            temp_paths.append(temp.name)

        # 이미지 로딩
        img1 = cv2.imread(temp_paths[0])
        img2 = cv2.imread(temp_paths[1])

        if img1 is None or img2 is None:
            raise ValueError("이미지를 읽을 수 없습니다. JPEG 형식인지 확인해주세요.")

        # 전처리 및 분석
        processed1 = preprocess_and_resize(img1)
        processed2 = preprocess_and_resize(img2)
        highlighted, _ = find_common_free_slots(processed1.copy(), processed2.copy())

        # ✅ 부드러운 하이라이트 처리 시작
        def apply_soft_highlight(base_img, highlighted_img, mask_color=(0, 255, 0), alpha=0.35):
            mask = cv2.inRange(highlighted_img, (0, 250, 0), (0, 255, 0))  # 초록 마스크
            result = base_img.copy()
            result[mask > 0] = (
                alpha * np.array(mask_color) + (1 - alpha) * result[mask > 0]
            ).astype(np.uint8)
            return result

        result_image = apply_soft_highlight(processed1, highlighted)

        # 6. 결과 저장
        output_dir = "output"
        os.makedirs(output_dir, exist_ok=True)
        result_path = os.path.join(output_dir, f"highlighted_{images[0].filename}")
        cv2.imwrite(result_path, result_image)

        # 7. 결과 반환
        return TimetableResponseDto(imageUrl=result_path)

    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})
