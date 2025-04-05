import json
def convert_to_json(parsed_data: dict) -> dict:
    print("📚 받은 과목 수:", len(parsed_data["courses"]))
    if not parsed_data["courses"]:
        print("❗과목이 하나도 없습니다. pdf_parser.py에서 과목 추출이 실패했을 수 있음.")
    info = parsed_data["student_info"]
    courses = parsed_data["courses"]
    
    summary = {
        "total_credits": 0,
        "major_credits": {
            "foundation": 0,
            "required": 0,
            "elective": 0,
            "industry": 0
        },
        "graduation_flags": {
            "capstone": False,
            "thesis": False,
            "project": False,
            "english_courses": 0
        }
    }

    for course in courses:
        name = course["name"]
        category = course["category"]
        mark = course["mark"]
        credits = course["credits"]
        code = course.get("code", "")[:2]  # group code (예: "04")

        # 편입인정학점인 경우 따로 판단
        if "편입인정학점" in name:
            if code == "04":
                summary["major_credits"]["required"] += credits
                summary["total_credits"] += credits
                print(f"✅ 편입인정학점 반영됨 (전공필수: {credits}학점)")
            elif code == "05":
                summary["major_credits"]["elective"] += credits
                summary["total_credits"] += credits
                print(f"✅ 편입인정학점 반영됨 (전공선택: {credits}학점)")
            elif code == "11":
                summary["major_credits"]["foundation"] += credits
                summary["total_credits"] += credits
                print(f"✅ 편입인정학점 반영됨 (전공기초: {credits}학점)")
            else:
                print(f"🚫 편입인정학점 제외됨 (편입) → {credits}학점")
            continue  # ❗ 일반 과목 처리로 넘어가지 않도록 반드시 필요

        # ✅ 일반 과목 처리
        summary["total_credits"] += credits
        if category == "전공기초":
            summary["major_credits"]["foundation"] += credits
        elif category == "전공필수":
            summary["major_credits"]["required"] += credits
        elif category == "전공선택":
            summary["major_credits"]["elective"] += credits
        elif (mark == "#" and not ("캡스톤디자인" in name or "졸업프로젝트" in name)) or \
             category == "산학필수":
            summary["major_credits"]["industry"] += credits


        # 졸업 필수 항목 추적
        if name == "캡스톤디자인":
            summary["graduation_flags"]["capstone"] = True
        elif "졸업논문" in name:
            summary["graduation_flags"]["thesis"] = True
        elif "졸업프로젝트" in name:
            summary["graduation_flags"]["project"] = True
        if mark.lower() == "e":
            summary["graduation_flags"]["english_courses"] += 1
    print("🧠 GPT에 전달할 요약 정보 ↓↓↓")
    print(json.dumps({
        "student_id": info["student_id"],
        "name": info["name"],
        "department": info["department"],
        "summary": summary,
    }, ensure_ascii=False, indent=2))
    return {
        "student_id": info["student_id"],
        "name": info["name"],
        "department": info["department"],
        "summary": summary,
        "courses": courses
    }
