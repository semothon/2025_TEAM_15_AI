# utils/pdf_utils.py
import re
import fitz  # PyMuPDF
from .graduation import extract_gyoyang

def extract_info_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = doc[0].get_text()

    result = {}

    result["학번"] = re.search(r"학\s*번\s*(\d{10})", text)
    result["이름"] = re.search(r"성\s*명\s*([가-힣]+)", text)
    result["학과"] = re.search(r"학\s*과\s*([가-힣]+)", text)

    total_match = re.search(r"취득\s+(\d+)\s+0\(\d+\)", text)
    if total_match:
        result["총학점_기준"] = 130
        result["총학점_취득"] = int(total_match.group(1))

    major_match = re.search(
        r"전공계:\s*\d+\s*\(\s*전필:\s*(\d+)\s*/\s*42\s*전선:\s*(\d+)\s*/\s*27\s*전기:\s*(\d+)\s*/\s*12\s*\)", text
    )
    if major_match:
        result["전공필수"] = int(major_match.group(1))
        result["전공선택"] = int(major_match.group(2))
        result["전공기초"] = int(major_match.group(3))
        result["산학필수"] = 0

    eng_match = re.search(r"영어강의\s*(\d)", text)
    result["영어강좌"] = int(eng_match.group(1)) if eng_match else 0

    result["졸업논문"] = "미통과" if "졸업능력인증" in text and "미취득" in text else "통과"

    sw_match = re.search(r"s소프트웨어적사유\s*3", text)
    result["SW교양"] = 3 if sw_match else 0

    result["최종판정"] = "졸업유예" if "최종판정 졸업유예" in text else "졸업"

    result["교양"] = extract_gyoyang(text)

    for k, v in result.items():
        if isinstance(v, re.Match):
            result[k] = v.group(1)

    return result