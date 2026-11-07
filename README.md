# Trợ Lý Ảo Tra Cứu Y Học Cổ Truyền Việt Nam (Advanced RAG Chatbot)
Hệ thống Trợ lý ảo Y học Cổ truyền Việt Nam là một ứng dụng Hỏi - Đáp (Q&A) thông minh áp dụng kiến trúc Advanced RAG (Retrieval-Augmented Generation).
Hệ thống được thiết kế nhằm giải quyết bài toán tra cứu phương tễ Đông y và phác đồ y tế chính thống từ nguồn tài liệu chuẩn hóa, giúp triệt tiêu hoàn toàn hiện tượng "ảo giác" (hallucination) của các mô hình ngôn ngữ lớn (LLM).
## Kho tri thức đóng (Knowledge Base):
1. Hướng dẫn chẩn đoán và điều trị bệnh – Bộ Y tế (2020) (Văn bản phác đồ y tế chuẩn Tây Y / Đông Tây Y kết hợp).

2. Toa thuốc Đông y cổ truyền Việt Nam – Bác sĩ Nguyễn Văn Hưởng (Sách phương tễ và kinh nghiệm lâm sàng 621 trang).

3. Những bài thuốc tâm huyết của 800 danh y (Bách khoa bài thuốc cổ phương và nghiệm phương).

4. Nam Dược Thần Hiệu – Đại y thiền sư Tu Tĩnh (Tài liệu y học cổ truyền Việt Nam kinh điển).

5. Những cây thuốc và vị thuốc Việt Nam – Giáo sư Tiến sĩ Đỗ Tất Lợi (Bách khoa toàn thư dược liệu và vị thuốc Đông y).
## Tính Năng Nổi Bật
- Số hóa dữ liệu dạng ảnh quét (Hybrid OCR & Cleaning Pipeline): Sử dụng PyMuPDF kết hợp EasyOCR bóc tách chính xác tiếng Việt từ sách cổ phương bị mờ/scan, chuyển đổi thành dữ liệu cấu trúc JSON và làm sạch nhiễu tự động.
- Tự động làm sạch & Khử trùng lặp (Deduplication): Sử dụng bộ lọc ký tự nhiễu và thuật toán seen_chunks để triệt tiêu các đoạn dữ liệu trùng lặp trong Vector Database.
- Truy xuất hai tầng (Two-Stage Retrieval System):
+ Tầng 1 (MMR Retrieval): Lấy $k=20$ phân đoạn đa dạng về mặt ngữ nghĩa từ ChromaDB (với fetch_k=40).
+ Tầng 2 (Cross-Encoder Reranking): Dùng mô hình ms-marco-MiniLM-L-6-v2 đọc kỹ và chấm điểm lại mức độ liên quan, chọn ra Top 8 phân đoạn chuẩn nhất.
- Minh bạch nguồn gốc (Data Provenance): Hiển thị rõ ràng tên tài liệu gốc và vị trí phân đoạn (chunk_id) phục vụ kiểm chứng y khoa.
- Chế độ gỡ lỗi (Debug Mode): Cho phép xem trực tiếp các đoạn văn bản thô mà LLM tiếp nhận.
## Kiến Trúc Hệ Thống
[Người dùng nhập câu hỏi trên Streamlit UI (app.py)]
       │
       ▼
[MMR Retriever] ──(Truy vấn VectorDB Chroma tại data/chroma_db)──► [Lấy Top 20 Chunks]
       │
       ▼
[Cross-Encoder Reranker] ──(Chấm điểm ngữ nghĩa logic)──► [Chọn Top 8 Chunks xuất sắc nhất]
       │
       ▼
[Prompt Template Strict Rules] + [Gemini 2.5 Flash]
       │
       ▼
[Phản hồi có Cảnh báo Y tế & Trích dẫn Nguồn gốc trên Web Browser]
## Cấu Trúc Thư Mục Dự Án
chatbot_rag_thuoc_co_truyen/
│
├── app.py                               # Giao diện chính Streamlit Web App & RAG Pipeline
├── requirements.txt                     # Danh sách thư viện phụ thuộc Python
├── README.md                            # Tài liệu hướng dẫn sử dụng và báo cáo đồ án
│
├── data/                                # Thư mục lưu trữ Kho tài liệu & Cơ sở dữ liệu Vector
│   ├── chroma_db/                       # Cơ sở dữ liệu Vector ChromaDB (chroma.sqlite3)
│   ├── Hướng dẫn chẩn đoán và điều trị bệnh.pdf
│   ├── nam-duoc-than-hieu-pdf.pdf
│   ├── Những bài thuốc tâm huyết của 800 danh y.pdf
│   ├── nhung-cay-thuoc-va-vi-thuoc-viet-nam-vnras.pdf
│   ├── toa-thuoc-dong-y-co-truyen-viet-nam.pdf
│   ├── ket_qua_huong_dan_chan_doan.json # Dữ liệu trích xuất dạng JSON
│   ├── ket_qua_ocr_*.json               # Kết quả OCR thô dạng JSON
│   └── clean_*.json                     # Dữ liệu JSON đã được làm sạch nhiễu chính tả
│
└── tools_and_tests/                     # Phân hệ script số hóa, làm sạch & kiểm thử
    ├── .env                             # Cấu hình API Key bảo mật (GOOGLE_API_KEY)
    ├── build_vector_db.py               # Script đóng gói & xây dựng Vector Database ChromaDB
    ├── clean_ocr_all.py                 # Script làm sạch nhiễu và chuẩn hóa file OCR
    ├── ocr_batch_scanned_pdfs.py        # Pipeline OCR quét hàng loạt tài liệu PDF Scan
    ├── ocr_full_800_danh_y.py           # Script chuyên dụng OCR toàn bộ sách 800 Danh Y
    ├── huong_dan_chan_doan.py           # Script xử lý dữ liệu Phác đồ Bộ Y tế
    ├── benchmark_800_danh_y.py          # Script đo đạc hiệu năng & tốc độ OCR
    ├── check_all_pdf.py                 # Tool kiểm tra định dạng PDF (Native Text vs Scan Image)
    ├── interactive_rag_cli.py           # Chạy thử nghiệm RAG trên giao diện dòng lệnh (CLI)
    ├── test_clean_ocr.py                # Script kiểm thử chất lượng dữ liệu làm sạch
    ├── test_gemini_api.py               # Script kiểm tra kết nối API Gemini 2.5 Flash
    └── test_rag_pipeline.py             # Script kiểm thử luồng truy xuất tri thức RAG
## Hướng Dẫn Cài Đặt & Khởi Chạy
1. Yêu cầu môi trường
- Hệ điều hành: Windows / macOS / Linux
- Python: phiên bản >= 3.10 (Khuyên dùng môi trường Anaconda)
2. Cài đặt thư viện phụ thuộc
- Mở Terminal / Command Prompt tại thư mục chatbot_rag_thuoc_co_truyen và thực hiện:
```
pip install -r requirements.txt
```
- Nếu dùng Anaconda trên Windows, có thể chạy lệnh trực tiếp:
```
C:\anaconda3\python.exe -m pip install -r requirements.txt
```
3. Cấu hình API Key
- Kiểm tra file .env nằm trong thư mục tools_and_tests/.env (hoặc tạo thêm tại thư mục gốc) chứa thông số:
```
GOOGLE_API_KEY="..."
```
## Quy Trình Thực Thi Dự Án (Execution Workflow)
### Bước 1: Số hóa & Làm sạch dữ liệu (OCR & Cleaning)
(Tiến hành quét OCR các PDF scan và sinh các file clean_*.json trong thư mục data/)
```
C:\anaconda3\python.exe tools_and_tests/ocr_batch_scanned_pdfs.py
C:\anaconda3\python.exe tools_and_tests/clean_ocr_all.py
```
### Bước 2: Nạp dữ liệu vào Cơ sở dữ liệu Vector (Build Vector DB)
- Sau khi đã có đầy đủ các file clean_*.json trong thư mục data/, nạp dữ liệu vào ChromaDB:
```
C:\anaconda3\python.exe tools_and_tests/build_vector_db.py
```
Script sẽ băm nhỏ văn bản (chunk size 1200, overlap 200), khử trùng lặp và lưu trữ vĩnh viễn vào data/chroma_db.
### Bước 3: Khởi chạy ứng dụng Web Chatbot (Streamlit App)
```
C:\anaconda3\python.exe -m streamlit run app.py
```
Trình duyệt sẽ tự động kích hoạt ứng dụng Web tại địa chỉ local: http://localhost:8501
## Kịch Bản Kiểm Thử Đề Xuất (Sample Test Cases)
1. Tra cứu ca bệnh đa triệu chứng:
```
"Tôi bị đau đầu, cảm cúm, sổ mũi và tê bì chân tay. Trong tài liệu có bài thuốc nào hỗ trợ không?"
```
2. Tra cứu bài thuốc cụ thể trong sách Bác sĩ Nguyễn Văn Hưởng:
```
"Bài thuốc chữa hen suyễn trong sách Toa thuốc Đông y của bác sĩ Nguyễn Văn Hưởng gồm những vị gì và định lượng ra sao?"
```
3. Tra cứu công dụng vị thuốc trong sách Đỗ Tất Lợi / Tu Tĩnh:
```
"Vị thuốc Ba Kích hoặc Cà Gai Leo có công dụng và liều dùng như thế nào theo tài liệu y học cổ truyền?"
```
4. Tra cứu phác đồ Bộ Y tế:
```
"Phác đồ chẩn đoán và điều trị bệnh Tăng huyết áp của Bộ Y tế quy định như thế nào?"
```
## Cảnh Báo Y Tế (Medical Disclaimer)
### LƯU Ý QUAN TRỌNG:
Thông tin do Chatbot cung cấp được trích xuất hoàn toàn từ các tài liệu Y học Cổ truyền và Phác đồ y tế phục vụ cho mục đích nghiên cứu học thuật và tham khảo. Hệ thống KHÔNG thay thế cho việc khám, chẩn đoán, tư vấn hoặc điều trị trực tiếp từ bác sĩ chuyên khoa.