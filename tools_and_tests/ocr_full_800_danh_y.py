import os
import json
import time
import fitz
import easyocr
from tqdm import tqdm

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

pdf_path = "data/Những bài thuốc tâm huyết của 800 danh y.pdf"
output_path = "data/ket_qua_ocr_800_danh_y.json"

ocr_reader = easyocr.Reader(['vi'], gpu=True)

extracted_data = {}
if os.path.exists(output_path):
    try:
        with open(output_path, "r", encoding="utf-8") as f:
            extracted_data = json.load(f)
        print(f"-> Đã tìm thấy file kết quả cũ. Đã OCR trước đó: {len(extracted_data)} trang.")
    except Exception:
        print("-> Không thể đọc file cũ, sẽ tạo file mới.")

pdf = fitz.open(pdf_path)
total_pages = len(pdf)

print(f"Bắt đầu OCR toàn bộ {total_pages} trang...")
start_time = time.time()

for page_num in tqdm(range(total_pages), desc="Đang OCR"):
    page_str = str(page_num + 1)
    
    if page_str in extracted_data:
        continue

    page = pdf[page_num]
    pix = page.get_pixmap(dpi=150)
    img_bytes = pix.tobytes("png")

    text_list = ocr_reader.readtext(img_bytes, detail=0)
    extracted_data[page_str] = " ".join(text_list)

    if (page_num + 1) % 10 == 0 or (page_num + 1) == total_pages:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(extracted_data, f, ensure_ascii=False, indent=2)

pdf.close()
elapsed = time.time() - start_time

print(f"\n Hoàn tất! Đã lưu kết quả vào: {output_path}")
print(f"Tổng thời gian: {elapsed / 60:.2f} phút.")