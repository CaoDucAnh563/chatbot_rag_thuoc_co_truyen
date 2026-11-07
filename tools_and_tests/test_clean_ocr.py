import os
import json
from dotenv import load_dotenv
from google import genai

load_dotenv()
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

raw_json_path = "data/ket_qua_ocr_800_danh_y.json"
with open(raw_json_path, "r", encoding="utf-8") as f:
    raw_data = json.load(f)

SYSTEM_PROMPT = """
Bạn là một chuyên gia hiệu đính văn bản Y học cổ truyền Việt Nam.
Nhiệm vụ của bạn là sửa lỗi chính tả từ kết quả OCR của sách thuốc cổ truyền.

Quy tắc bắt buộc:
1. Sửa lỗi gõ sai, thiếu/nhầm dấu tiếng Việt, dính chữ, khoảng trắng thừa.
2. Chuẩn hóa chính xác các thuật ngữ Y học cổ truyền (tên vị thuốc, bài thuốc, liều lượng, phép trị, triệu chứng).
3. KHÔNG tóm tắt, KHÔNG thêm bớt ý, KHÔNG bình luận hay giải thích gì thêm.
4. Giữ nguyên cấu trúc văn bản gốc, chỉ trả về đoạn văn bản đã được làm sạch.
"""

test_pages = ["1", "2", "3", "4", "5"]

print("=== BẮT ĐẦU TEST LÀM SẠCH VĂN BẢN ĐÔNG Y ===\n")

for page_num in test_pages:
    if page_num not in raw_data:
        continue
        
    raw_text = raw_data[page_num]
    
    prompt = f"Hãy sửa lỗi OCR cho trang sách sau:\n\n{raw_text}"
    
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=genai.types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0.1 
        )
    )
    
    cleaned_text = response.text
    
    print(f"--- TRANG {page_num} ---")
    print(f"OCR Thô  : {raw_text[:150]}...")
    print(f"Sạch AI  : {cleaned_text[:150]}...")
    print("-" * 50)