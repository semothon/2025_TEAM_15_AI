# utils/graduation.py
import re

def extract_gyoyang(text):
    def get_match(pattern):
        m = re.search(pattern, text)
        return int(m.group(1)) if m else 0

    교양 = {
        "필수교과": {
            "영역": get_match(r"필수교과\s+(\d+)\s*/\s*3"),
            "영역기준": 3,
            "학점": get_match(r"필수교과.*?(\d+)\s*/\s*17"),
            "학점기준": 17
        },
        "배분이수": {
            "영역": get_match(r"배분이수교과.*?(\d+)\s*/\s*3"),
            "영역기준": 3,
            "학점": get_match(r"배분이수교과.*?(\d+)\s*/\s*9"),
            "학점기준": 9
        },
        "자유이수": {
            "영역": get_match(r"자유이수.*?(\d+)\s*/\s*2"),
            "영역기준": 2,
            "학점": get_match(r"자유이수.*?(\d+)\s*/\s*3"),
            "학점기준": 3
        },
    }

    for key, value in 교양.items():
        value["판정"] = "통과" if value["영역"] >= value["영역기준"] and value["학점"] >= value["학점기준"] else "미통과"

    return 교양

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
