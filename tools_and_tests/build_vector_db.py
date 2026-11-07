import os
import json
import chromadb
from chromadb.utils import embedding_functions
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import tqdm

print("Đang tải mô hình Embedding tiếng Việt...")
vietnamese_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="bkai-foundation-models/vietnamese-bi-encoder"
)

db_path = "data/chroma_db"
chroma_client = chromadb.PersistentClient(path=db_path)
collection = chroma_client.get_or_create_collection(
    name="y_hoc_co_truyen",
    embedding_function=vietnamese_ef
)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150,
    separators=["\n\n", "\n", ". ", ";", " ", ""]
)

clean_files = [
    "ket_qua_huong_dan_chan_doan.json",
    "clean_800_danh_y.json",
    "clean_nam_duoc_than_hieu_pdf.json",
    "clean_nhung_cay_thuoc_va_vi_thuoc_viet_nam_vnras.json",
    "clean_toa_thuoc_dong_y_co_truyen_viet_nam.json"
]

data_dir = "data"

for file_name in clean_files:
    file_path = os.path.join(data_dir, file_name)
    if not os.path.exists(file_path):
        print(f"Bỏ qua file không tồn tại: {file_name}")
        continue

    print(f"\n==========================================")
    print(f"Đang xử lý file: {file_name}")

    with open(file_path, "r", encoding="utf-8") as f:
        doc_data = json.load(f)

    documents = []
    metadatas = []
    ids = []

    doc_source = file_name.replace("clean_", "").replace("ket_qua_", "").replace(".json", "")

    for page_num, text in tqdm(doc_data.items(), desc="Cắt đoạn & Chuẩn bị metadata"):
        if not text.strip():
            continue

        chunks = text_splitter.split_text(text)

        for i, chunk in enumerate(chunks):
            chunk_id = f"{doc_source}_p{page_num}_c{i}"
            documents.append(chunk)
            metadatas.append({
                "source": doc_source,
                "page": int(page_num) if page_num.isdigit() else page_num
            })
            ids.append(chunk_id)

    batch_size = 200
    total_chunks = len(documents)
    print(f"Tổng cộng {total_chunks} chunks. Đang tạo Vector Embedding...")

    for idx in tqdm(range(0, total_chunks, batch_size), desc="Đang nạp vào ChromaDB"):
        batch_docs = documents[idx : idx + batch_size]
        batch_meta = metadatas[idx : idx + batch_size]
        batch_ids = ids[idx : idx + batch_size]

        collection.upsert(
            documents=batch_docs,
            metadatas=batch_meta,
            ids=batch_ids
        )

print(f"\nHOÀN THÀNH! Tổng số vectors hiện có trong DB: {collection.count()}")