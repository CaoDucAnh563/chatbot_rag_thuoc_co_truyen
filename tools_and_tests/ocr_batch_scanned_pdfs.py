import os
import json
import time
import fitz
import easyocr
from tqdm import tqdm

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

data_dir = "data"

pdf_files = [
    "nam-duoc-than-hieu-pdf.pdf",
    "nhung-cay-thuoc-va-vi-thuoc-viet-nam-vnras.pdf",
    "toa-thuoc-dong-y-co-truyen-viet-nam.pdf"
]

ocr_reader = easyocr.Reader(['vi'], gpu=True)

for filename in pdf_files:
    pdf_path = os.path.join(data_dir, filename)
    
    if not os.path.exists(pdf_path):
        print(f"\nKhông tìm thấy file: {pdf_path}, bỏ qua.")
        continue

    clean_name = filename.replace(".pdf", "").replace("-", "_")
    output_path = os.path.join(data_dir, f"ket_qua_ocr_{clean_name}.json")

    print(f"\n==========================================")
    print(f"Bắt đầu OCR: {filename}")
    print(f"File lưu: {output_path}")

    extracted_data = {}
    if os.path.exists(output_path):
        try:
            with open(output_path, "r", encoding="utf-8") as f:
                extracted_data = json.load(f)
            print(f"-> Đã tìm thấy tiến trình cũ: {len(extracted_data)} trang đã hoàn thành.")
        except Exception:
            print("-> Khởi tạo file lưu mới.")

    doc = fitz.open(pdf_path)
    total_pages = len(doc)
    start_time = time.time()

    for page_num in tqdm(range(total_pages), desc=f"Đang OCR {filename[:15]}..."):
        page_str = str(page_num + 1)

        if page_str in extracted_data:
            continue

        page = doc[page_num]
        pix = page.get_pixmap(dpi=150)
        img_bytes = pix.tobytes("png")

        text_list = ocr_reader.readtext(img_bytes, detail=0)
        extracted_data[page_str] = " ".join(text_list)

        if (page_num + 1) % 10 == 0 or (page_num + 1) == total_pages:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(extracted_data, f, ensure_ascii=False, indent=2)

    doc.close()
    elapsed = time.time() - start_time
    print(f"Hoàn thành {filename} ({total_pages} trang) trong {elapsed / 60:.2f} phút!")

print("\nĐÃ HOÀN THÀNH OCR TOÀN BỘ CÁC FILE SCAN!")