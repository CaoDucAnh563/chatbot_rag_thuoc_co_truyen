import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import time
import fitz
import easyocr



ocr_reader = easyocr.Reader(['vi'], gpu=True)

pdf_path = "data/Những bài thuốc tâm huyết của 800 danh y.pdf"
pdf = fitz.open(pdf_path)

start = time.time()
extracted_texts = {}  

test_pages = min(10, len(pdf))

for page_num in range(test_pages):
    page = pdf[page_num]

    pix = page.get_pixmap(dpi=150)
    img_bytes = pix.tobytes("png")

    text_list = ocr_reader.readtext(img_bytes, detail=0)
    extracted_texts[page_num + 1] = " ".join(text_list)

total_pages = len(pdf)

pdf.close()

elapsed = time.time() - start

print(f"Xử lý {test_pages} trang hết: {elapsed:.2f} giây")
print(f"Trung bình: {elapsed / test_pages:.2f} giây/trang")

estimate = (elapsed / test_pages) * total_pages / 60
print(f"Dự đoán toàn bộ {total_pages} trang sách: {estimate:.2f} phút")