from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import pytesseract
from pdf2image import convert_from_bytes
import uvicorn
from PIL import Image
import cv2
import numpy as np
import re

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

app = FastAPI()

def extract_tables_from_pdf(pdf_bytes: bytes):
    images = convert_from_bytes(
        pdf_bytes,
        dpi=300,
        poppler_path=r"C:\Users\ddd22\Downloads\Release-24.08.0-0\poppler-24.08.0\Library\bin"
    )

    all_data = []

    for i, image in enumerate(images):
       
        cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
        _, thresh = cv2.threshold(cv_image, 180, 255, cv2.THRESH_BINARY)
        image = Image.fromarray(thresh)
 
        text = pytesseract.image_to_string(image, lang='kor+eng', config='--psm 6')

        lines = text.strip().split('\n')
        lines = [line for line in lines if line.strip() != ""]

        if len(lines) >= 2:
            headers = re.split(r'\s{2,}', lines[0].strip())
            for row in lines[1:]:
                values = re.split(r'\s{2,}', row.strip())
                if len(values) == len(headers):
                    row_dict = dict(zip(headers, values))
                    all_data.append(row_dict)

    return all_data


'''
graduation_check1 = {}
type_graduation = ["수강학점", "취득학점", "전공", "교양", "졸업평점", "영어강의", "논문", "한국어능력인증", "졸업능력인증", "SW인증"]
for check in type_graduation:
    graduation_check1[type_graduation] = #인식한거

for a in graduation_check1:
    if graduation_check1[type_graduation] #취득한게 기준을 넘는지 확인인
    

graduation_check2 = {}
type_gen_ed = ["배분이수교과(2024~)(영역)", "배분이수교과(2024~)(학점)", "자유이수(영역)", "자유이수(학점)", "필수교과(영역)", "필수교과(학점점)"]
for check in type_gen_ed:
    graduation_check2[type_gen_ed] = 
'''
graduation_check3 = {}
type_major = ["전공필수", "전공선택", "전공기초"]


@app.post("/extract-table")
async def extract_table(file: UploadFile = File(...)):
    content = await file.read()
    table_data = extract_tables_from_pdf(content)
    return JSONResponse(content={"extracted_table": table_data})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)


