import json
import pdfplumber
import io
from utils.grad.gpt_client import ask_gpt

# 졸업 요건 통합 로딩
with open("data/cse_grad_2019_2025.json", "r", encoding="utf-8") as f1, \
     open("data/소프트웨어융합학과_졸업요건_2019_2025.json", "r", encoding="utf-8") as f2, \
     open("data/인공지능학과_졸업요건_2022_2025.json", "r", encoding="utf-8") as f3:
    graduation_rules_all = {
        **json.load(f1),
        **json.load(f2),
        **json.load(f3)
    }

async def analyze_graduation_pdf(file_content: bytes, major: str, student_id: str) -> str:
    # PDF 텍스트 추출
    with pdfplumber.open(io.BytesIO(file_content)) as pdf:
        full_text = "\n".join(
            [page.extract_text() for page in pdf.pages if page.extract_text()]
        )

    key = f"{major}{student_id[:4]}"
    rules = graduation_rules_all.get(key)
    if not rules:
        return f"[오류] 졸업 요건 데이터에서 '{key}' 항목을 찾을 수 없습니다."

    requirements_text = f"""
📌 졸업 기준 정보 ({key} 기준):

- 총 취득 학점: {rules['total_credits']}학점
- 전공 기초: {rules['foundation_credits']}학점
- 전공 필수: {rules['major_required_credits']}학점
- 전공 선택: {rules['major_elective_credits']}학점
- 산학 필수: {rules['industry_required_credits']}학점
- 필수 과목: {', '.join(rules['required_courses'])}
- 영어 교양: 일반학생 {rules['english_required']['regular']}학점 / 편입생 {rules['english_required']['transfer']}학점

이 기준을 엄격하게 적용해 주세요.
"""

    prompt = f"""
[졸업 진단 요청]

다음은 한 학생의 졸업 진단표 텍스트 전체입니다.
주어진 PDF 텍스트를 분석하고, 아래 졸업 요건 기준에 맞춰 졸업 가능 여부를 판단해주세요.

{requirements_text}

---
1. 영역별 취득 학점 현황:
- 전공 기초: ?학점 / {rules['foundation_credits']}학점
- 전공 필수: ?학점 / {rules['major_required_credits']}학점
- 전공 선택: ?학점 / {rules['major_elective_credits']}학점
- 교양 영역: ?학점 / 기준 없음
- 자유 이수: ?학점 / 기준 없음
- 총 취득 학점: ?학점 / {rules['total_credits']}학점

2. 부족한 영역 및 학점 수:
- 예: 전공 기초: 15학점 부족

3. 학점 외 필수 조건 충족 여부:
- 영어강의: ✅ 충족 / ❌ 미충족
- 논문: ✅ / ❌
- SW인증: ✅ / ❌
- 졸업능력인증: ✅ / ❌
- 필수 과목(캡스톤디자인 등): ✅ / ❌

4. 최종 졸업 가능 여부:
- ✅ 졸업 가능 / ❌ 졸업 불가
- 간단한 사유를 한 문장으로 작성
---

📄 졸업 진단표 텍스트:
{full_text}
"""

    try:
        gpt_response = ask_gpt(prompt)
        return gpt_response
    except Exception as e:
        return f"[GPT 오류] {str(e)}"
