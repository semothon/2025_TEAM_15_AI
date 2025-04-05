import io
import pdfplumber
import re

# 예시: 간단한 키워드 기반 분류
# 이름 기반으로 category 추론
# 📍 위치: pdf_parser.py
def infer_category(course_name: str, mark: str) -> str:
    if course_name in [
        "객체지향프로그래밍", "웹/파이선프로그래밍", "논리회로",
        "컴퓨터구조", "자료구조", "오픈소스SW개발방법및도구",
        "운영체제", "컴퓨터네트워크", "알고리즘",
        "데이터베이스", "소프트웨어공학", "기계학습"
    ]:
        return "전공필수"
    elif course_name in ["캡스톤디자인", "졸업프로젝트", "졸업논문(컴퓨터공학)"]:
        return "전공필수"
    elif mark == "#" and course_name not in ["캡스톤디자인", "졸업프로젝트"]:
        return "산학필수"
    elif course_name in ["확률및랜덤변수", "미분방정식"]:
        return "전공기초"
    else:
        return "전공선택"




def parse_pdf(file_bytes: bytes) -> dict:
    result = {
        "student_info": {},
        "courses": []
    }

    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        full_text = "\n".join(
            [page.extract_text() for page in pdf.pages if page.extract_text()]
        )

        # print("📄 추출된 진단표 텍스트:", full_text)  # 앞부분만 잘라서 보기

    # 기본 정보 추출 (학번, 이름, 학과)
    student_id_match = re.search(r"학\s*번\s+(\d{9,10})", full_text)
    name_match = re.search(r"성\s*명\s+([가-힣]{2,4})", full_text)
    department_match = re.search(r"학\s*과\s+([가-힣]+)", full_text)


    result["student_info"] = {
        "student_id": student_id_match.group(1) if student_id_match else "Unknown",
        "name": name_match.group(1) if name_match else "Unknown",
        "department": department_match.group(1) if department_match else "Unknown"
    }

    # 과목 추출
    course_pattern = re.compile(
    r"([A-Z]{2,6}[0-9]{3})\s+([es#]*)?([가-힣A-Za-z0-9()/+#]+)\s+(\d)\s+(\d{4}) ?/ ?(\d)"
    )
    current_group_code = None  # 이수구분 코드 초기화
    for match in course_pattern.finditer(full_text):
        if re.match(r"^(0[1-9]|1[0-2])\s", match.group(0)):
            current_group_code = match.group(1)  # 이수구분 코드 저장
        code, mark, name, credits, year, semester = match.groups()
        category = infer_category(name, mark)
        result["courses"].append({
            "code": code,
            "name": name.strip(),
            "credits": int(credits),
            "semester": f"{year}/{semester}",
            "category": category,
            "mark": (mark or "").strip().lower(),
        })
    # 편입인정학점 추출 (이수구분코드와 함께)
    transfer_pattern = re.compile(r"(\d{2})\s+편입인정학점\s+(\d+)")
    for match in transfer_pattern.finditer(full_text):
        group_code, credits = match.groups()
        result["courses"].append({
            "code": "편입",
            "name": "편입인정학점",
            "credits": int(credits),
            "semester": "",
            "category": "편입",
            "mark": "",
            "group_code": group_code  # 중요
        })


    return result


