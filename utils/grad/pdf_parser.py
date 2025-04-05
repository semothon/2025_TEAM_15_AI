import io
import pdfplumber
import re

# 과목명 기반 카테고리 추론
CATEGORY_MAP = {
    "전공기초": ["확률및랜덤변수", "미분방정식"],
    "전공필수": [
        "객체지향프로그래밍", "웹/파이선프로그래밍", "논리회로",
        "컴퓨터구조", "자료구조", "오픈소스SW개발방법및도구",
        "운영체제", "컴퓨터네트워크", "알고리즘",
        "데이터베이스", "소프트웨어공학", "기계학습"
    ],
    "전공필수_특정": ["캡스톤디자인", "졸업프로젝트", "졸업논문(컴퓨터공학)"],
    "산학필수": ["SW스타트업비즈니스", "최신기술콜로키움2", "블록체인", "클라우드컴퓨팅", "연구연수활동1(컴퓨터공학)"]
}


def infer_category(course_name: str, mark: str) -> str:
    if course_name in CATEGORY_MAP["전공필수"]:
        return "전공필수"
    elif course_name in CATEGORY_MAP["전공필수_특정"]:
        return "전공필수"
    elif mark == "#" and course_name not in ["캡스톤디자인", "졸업프로젝트"]:
        return "산학필수"
    elif course_name in CATEGORY_MAP["산학필수"]:
        return "산학필수"
    elif course_name in CATEGORY_MAP["전공기초"]:
        return "전공기초"
    elif mark == "s":
        return "SW인증"
    elif mark == "e":
        return "영어강의"
    else:
        return "전공선택"


def parse_pdf(file_bytes: bytes) -> dict:
    result = {
        "student_info": {},
        "courses": [],
        "summary": {}  # 총 취득학점 등 추가 정보
    }

    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        full_text = "\n".join(
            [page.extract_text() for page in pdf.pages if page.extract_text()]
        )

        print("📄 PDF 전체 텍스트 추출 결과:\n")
        print(full_text[:2000])  # 일부만 출력하여 확인

        # 전공 요약 정보 추출
        match = re.search(
            r"전공계:\s*(\d+).*?전필:\s*(\d+)/\d+\s+전선:\s*(\d+)/\d+\s+전기:\s*(\d+)/\d+",
            full_text
        )
        if match:
            result["summary"]["총전공"] = int(match.group(1))
            result["summary"]["전공필수"] = int(match.group(2))
            result["summary"]["전공선택"] = int(match.group(3))
            result["summary"]["전공기초"] = int(match.group(4))

    # 학생 정보 추출
    student_id_match = re.search(r"학\s*번\s+(\d{9,10})", full_text)
    name_match = re.search(r"성\s*명\s+([가-힣]{2,4})", full_text)
    department_match = re.search(r"학\s*과\s+([가-힣]+)", full_text)

    result["student_info"] = {
        "student_id": student_id_match.group(1) if student_id_match else "Unknown",
        "name": name_match.group(1) if name_match else "Unknown",
        "department": department_match.group(1) if department_match else "Unknown"
    }

    # 총 취득 학점 파싱 시도 (예: "취득 123(142)")
    total_credits_match = re.search(r"취득\s+(\d+)(?:\(\d+\))?", full_text)
    if total_credits_match:
        result["summary"]["total_credits"] = int(total_credits_match.group(1))

    # 과목 정보 추출
    course_pattern = re.compile(
        r"([A-Z]{2,6}[0-9]{3})\s+([es#]*)?([가-힣A-Za-z0-9()/+#]+)\s+(\d)\s+(\d{4})\s*/\s*(\d)"
    )
    for match in course_pattern.finditer(full_text):
        code, mark, name, credits, year, semester = match.groups()
        category = infer_category(name.strip(), (mark or "").strip())

        result["courses"].append({
            "code": code,
            "name": name.strip(),
            "credits": int(credits),
            "semester": f"{year}/{semester}",
            "category": category,
            "mark": (mark or "").strip().lower(),
        })

    return result