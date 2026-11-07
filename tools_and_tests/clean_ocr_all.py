import os
import json
import time
from dotenv import load_dotenv
from google import genai
from tqdm import tqdm

load_dotenv()
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

data_dir = "data"

raw_files = [
    "ket_qua_ocr_800_danh_y.json",
    "ket_qua_ocr_nam_duoc_than_hieu_pdf.json",
    "ket_qua_ocr_nhung_cay_thuoc_va_vi_thuoc_viet_nam_vnras.json",
    "ket_qua_ocr_toa_thuoc_dong_y_co_truyen_viet_nam.json"
]

SYSTEM_PROMPT = """
Bạn là một chuyên gia hiệu đính văn bản Y học cổ truyền Việt Nam.
Nhiệm vụ của bạn là sửa lỗi chính tả từ kết quả OCR của sách thuốc cổ truyền.

Quy tắc bắt buộc:
1. Sửa lỗi gõ sai, thiếu/nhầm dấu tiếng Việt, dính chữ, khoảng trắng thừa.
2. Chuẩn hóa chính xác các thuật ngữ Y học cổ truyền (tên vị thuốc, bài thuốc, liều lượng, phép trị, triệu chứng).
3. KHÔNG tóm tắt, KHÔNG thêm bớt ý, KHÔNG bình luận hay giải thích gì thêm.
4. Giữ nguyên cấu trúc văn bản gốc, chỉ trả về đoạn văn bản đã được làm sạch.
"""

def clean_text_with_llm(raw_text, retries=3):
    if not raw_text.strip():
        return ""
        
    for attempt in range(retries):
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=f"Hãy sửa lỗi OCR cho trang sách sau:\n\n{raw_text}",
                config=genai.types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    temperature=0.1
                )
            )
            return response.text.strip()
        except Exception as e:
            if attempt == retries - 1:
                print(f"\nLỗi API tại trang này (Bỏ qua): {e}")
                return raw_text 
            time.sleep(2)

for filename in raw_files:
    raw_path = os.path.join(data_dir, filename)
    if not os.path.exists(raw_path):
        print(f"Không tìm thấy file: {raw_path}, bỏ qua.")
        continue

    clean_filename = filename.replace("ket_qua_ocr_", "clean_")
    output_path = os.path.join(data_dir, clean_filename)

    print(f"\n==========================================")
    print(f"Đang làm sạch: {filename}")
    print(f"File đầu ra: {output_path}")

    with open(raw_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    cleaned_data = {}
    if os.path.exists(output_path):
        try:
            with open(output_path, "r", encoding="utf-8") as f:
                cleaned_data = json.load(f)
            print(f"-> Khôi phục tiến trình: Đã làm sạch {len(cleaned_data)}/{len(raw_data)} trang.")
        except Exception:
            cleaned_data = {}

    pages_to_process = [p for p in raw_data.keys() if p not in cleaned_data]
    
    for page_str in tqdm(pages_to_process, desc=f"Sửa chính tả {clean_filename[:15]}"):
        raw_text = raw_data[page_str]
        cleaned_text = clean_text_with_llm(raw_text)
        cleaned_data[page_str] = cleaned_text

        if len(cleaned_data) % 10 == 0 or len(cleaned_data) == len(raw_data):
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(cleaned_data, f, ensure_ascii=False, indent=2)

    print(f"Hoàn thành làm sạch: {clean_filename}")

print("\nTẤT CẢ DỮ LIỆU ĐÃ ĐƯỢC LÀM SẠCH BỞI GEMINI FLASH!")