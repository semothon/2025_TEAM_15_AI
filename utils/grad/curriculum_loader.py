# AI/utils/curriculum_loader.py
import json
import os

def load_graduation_rules(year: int, department: str) -> dict:
    filename = f"data/rules/graduation_rules_{year}.json"  # ✅ 반드시 이렇게
    if not os.path.exists(filename):
        print(f"[경고] 졸업요건 파일이 존재하지 않음: {filename}")
        return None
    with open(filename, "r", encoding="utf-8") as f:
        all_rules = json.load(f)
    return all_rules.get(department)
