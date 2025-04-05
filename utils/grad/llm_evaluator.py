# AI/evaluator/llm_evaluator.py

import os
from openai import OpenAI
import json
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def evaluate_with_llm(student_json: dict, graduation_rules: dict) -> str:
    system_prompt = (
        "너는 대학교 졸업 요건 평가 시스템이다. "
        "입력된 졸업 요건과 학생의 이수 현황을 비교하여 졸업 가능 여부를 판단하고, "
        "부족한 항목이 있다면 구체적으로 설명해줘. 마지막에 조치사항도 추천해줘.\n"
        "응답은 다음 형식을 따라야 한다:\n"
        "- 졸업 가능 여부\n"
        "- 부족 항목들\n"
        "- 👉 조치사항"
    )

    user_input = {
        "student": student_json,
        "rules": graduation_rules
    }

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": str(user_input)}
            ],
            temperature=0.3
        )
        print("🔍 GPT로 전달하는 student_json ↓↓↓")
        print(json.dumps(student_json, ensure_ascii=False, indent=2))
        print("🔍 GPT로 전달하는 graduation_rules ↓↓↓")
        print(json.dumps(graduation_rules, ensure_ascii=False, indent=2))

        result = response.choices[0].message.content
        print("🧠 GPT 응답 ↓↓↓\n", result)
        return result
        

    except Exception as e:
        print(f"🔥 GPT 호출 실패: {e}")
        return f"LLM 분석 중 오류 발생: {e}"
