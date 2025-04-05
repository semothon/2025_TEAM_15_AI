import json
from .pdf_parser import parse_pdf
from .gpt_client import ask_gpt

with open("data/rules/graduation_rules_2019.json", "r", encoding="utf-8") as f:
    graduation_rules_all = json.load(f)

async def analyze_graduation_pdf(file, major, student_id_from_query):
    content = await file.read()

    # 진단표 분석
    parsed_data = parse_pdf(content)
    student_info = parsed_data.get("student_info", {})
    courses = parsed_data.get("courses", [])

    # student_id mismatch 체크 (선택사항)
    if student_info.get("student_id") != student_id_from_query:
        return {
            "error": "입력된 학번과 PDF 내 학번이 일치하지 않습니다.",
            "pdf_student_id": student_info.get("student_id"),
            "query_student_id": student_id_from_query
        }

    # 학과 요건 불러오기
    graduation_rules = graduation_rules_all.get(major)
    if not graduation_rules:
        return f"[오류] '{major}' 학과의 졸업 요건이 존재하지 않습니다."

    # GPT 프롬프트 구성
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

이 학생이 졸업 요건을 충족하는지 판단해 주세요:
1. 어떤 항목이 충족되었고
2. 어떤 항목이 부족하며
3. 총 몇 학점이 부족한지
4. 졸업 가능 여부를 정리해서 알려주세요.
    """

    return ask_gpt(prompt)
