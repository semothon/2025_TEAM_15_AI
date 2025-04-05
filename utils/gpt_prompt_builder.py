from utils.requirements import GRADUATION_REQUIREMENTS

def build_graduation_prompt(student_id: str, department: str, transcript_text: str) -> str:
    return f"""아래는 {department} {student_id} 학생의 졸업진단표 요약 정보야.

      ------------------------------
      {transcript_text.strip()}
      ------------------------------

      위 내용을 기반으로 이 학생의 졸업 가능 여부를 판단해줘.
      다음 항목을 포함해서 JSON으로 응답해줘:

      1. 졸업 가능 여부 (true / false)
      2. 부족한 항목 (항목명, 부족 내용, 보완 방법 포함)
      3. 종합 의견 (이 학생의 졸업을 위한 조언 등)

      다음과 같은 JSON 형식으로 응답해:

      {{
        "졸업_가능": false,
        "부족_항목": [
          {{
            "항목": "전공 필수",
            "부족_내용": "현재 3학점, 총 42학점 필요. 39학점 부족.",
            "보완_방법": "다음 학기부터 전공필수 과목을 매 학기 2~3과목씩 수강 필요"
          }}
        ],
        "종합_의견": "졸업까지 전공필수와 전공기초가 많이 부족하므로 4학기 이상의 추가 수강이 필요할 수 있습니다."
      }}
      """
