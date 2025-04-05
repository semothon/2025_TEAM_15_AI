import json
from utils.grad.pdf_parser import parse_pdf
from utils.grad.gpt_client import ask_gpt  # GPT 호출 함수
from io import BytesIO

# 졸업 요건 불러오기
with open("data/rules/graduation_cse_2019.json", "r", encoding="utf-8") as f:
    graduation_rules_all = json.load(f)

async def analyze_graduation_pdf(file_content: bytes, major: str, student_id: str) -> str:
    # PDF 파싱
    parsed_data = parse_pdf(file_content)
    student_info = parsed_data.get("student_info", {})
    courses = parsed_data.get("courses", [])

    # 학과 졸업 요건 불러오기
    graduation_rules = graduation_rules_all.get(major)
    if not graduation_rules:
        return f"[오류] '{major}' 학과의 졸업 요건이 존재하지 않습니다."

    # 프롬프트 구성
    prompt = f"""
[졸업 진단 요청]

학생 정보:
- 이름: {student_info.get('name')}
- 학번: {student_info.get('student_id')}
- 학과: {student_info.get('department')}

졸업 요건:
{json.dumps(graduation_rules, ensure_ascii=False, indent=2)}

이수 과목 목록:
{json.dumps(courses, ensure_ascii=False, indent=2)}

위 정보를 기반으로 학생이 졸업 요건을 충족했는지 판단해 주세요:
1. 충족된 항목
2. 부족한 항목
3. 부족한 학점 수
4. 최종 졸업 가능 여부
"""

    # GPT 호출
    try:
        gpt_response = ask_gpt(prompt)
        return gpt_response
    except Exception as e:
        return f"[GPT 오류] {str(e)}"
