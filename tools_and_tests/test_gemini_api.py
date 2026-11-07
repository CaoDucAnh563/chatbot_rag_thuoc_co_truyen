import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    print("Lỗi: Chưa tìm thấy GOOGLE_API_KEY trong file .env!")
    exit()

client = genai.Client(api_key=api_key)

print("Đang kết nối tới Gemini API...")

try:
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents="Hãy trả lời duy nhất câu: Kết nối thành công!",
    )
    print("Kết quả:", response.text)
except Exception as e:
    print("Kết nối thất bại:", e)