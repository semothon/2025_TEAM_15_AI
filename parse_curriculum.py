import json
import re
import os

# JSON 파일 불러오기
with open("data/computer_engineering_courses.json", "r", encoding="utf-8") as f:
    courses = json.load(f)

# 과목명 클렌징 함수
def clean_name(raw_name):
    name = re.sub(r"^[\d\s|\\/]+", "", raw_name)  # 숫자/기호 제거
    name = re.sub(r"\bas\b", "", name, flags=re.IGNORECASE)
    name = re.sub(r"[^\w가-힣\s]", "", name)
    name = re.sub(r"(?<=[가-힣])\s+(?=[가-힣])", "", name)
    return name.strip()

# 전공 분류용 기준 목록
필수과목 = ["자료구조", "알고리즘", "졸업프로젝트", "객체지향프로그래밍", "운영체제", "컴퓨터구조"]
기초과목 = ["기초미분적분학", "프로그래밍기초", "컴퓨터개론", "파이썬프로그래밍", "논리회로"]

# 자동 분류 함수
def assign_category(name):
    if "산학" in name or "세미나" in name:
        return "전공선택"  # 산학필수는 전공선택으로 포함
    elif name in 필수과목:
        return "전공필수"
    elif name in 기초과목:
        return "전공기초"
    else:
        return "전공선택"

# 리팩토링 적용
cleaned_courses = []
for course in courses:
    cleaned_name = clean_name(course["name"])
    course["name"] = cleaned_name
    course["category"] = assign_category(cleaned_name)
    cleaned_courses.append(course)

# 저장
os.makedirs("data", exist_ok=True)
with open("data/computer_engineering_courses_cleaned.json", "w", encoding="utf-8") as f:
    json.dump(cleaned_courses, f, ensure_ascii=False, indent=2)

print(f"✅ {len(cleaned_courses)}개 과목 정제 및 분류 완료 → 'computer_engineering_courses_cleaned.json'")
