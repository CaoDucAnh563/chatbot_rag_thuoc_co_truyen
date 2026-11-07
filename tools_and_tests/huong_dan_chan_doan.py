import json
import fitz

pdf_path = "data/Hướng dẫn chẩn đoán và điều trị bệnh.pdf"
output_path = "data/ket_qua_huong_dan_chan_doan.json"

doc = fitz.open(pdf_path)
extracted_data = {}

for page_num in range(len(doc)):
    text = doc[page_num].get_text()
    extracted_data[str(page_num + 1)] = text.strip()

doc.close()

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(extracted_data, f, ensure_ascii=False, indent=2)

print(f"Đã trích xuất xong {len(extracted_data)} trang văn bản chuẩn!")