import os
import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
from google import genai

load_dotenv()

client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

db_path = "data/chroma_db"
vietnamese_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="bkai-foundation-models/vietnamese-bi-encoder"
)
chroma_client = chromadb.PersistentClient(path=db_path)
collection = chroma_client.get_collection(
    name="y_hoc_co_truyen",
    embedding_function=vietnamese_ef
)

def ask_rag(question, top_k=4):
    print(f"\nĐang truy vấn Vector DB cho câu hỏi: '{question}'...")
    
    results = collection.query(
        query_texts=[question],
        n_results=top_k
    )
    
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    
    context_blocks = []
    for doc, meta in zip(docs, metas):
        context_blocks.append(f"[Nguồn: {meta['source']} | Trang: {meta['page']}]\n{doc}")
    
    context_str = "\n\n---\n\n".join(context_blocks)
    
    prompt = f"""
Bạn là một chuyên gia Y học cổ truyền Việt Nam.
Hãy trả lời câu hỏi dựa TRỰC TIẾP và CHỈ DỰA VÀO bối cảnh tài liệu được cung cấp dưới đây.
Nếu bối cảnh không có thông tin để trả lời, hãy nói rõ "Tài liệu hiện tại không đề cập đến thông tin này".

=== BỐI CẢNH TÀI LIỆU ===
{context_str}

=== CÂU HỎI ===
{question}
"""

    print("Đang tổng hợp câu trả lời từ Gemini...")
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=genai.types.GenerateContentConfig(temperature=0.2)
    )
    
    print("\n================ CÂU TRẢ LỜI ================\n")
    print(response.text)
    print("\n================ TRÍCH DẪN NGUỒN TÀI LIỆU ================")
    for meta in metas:
        print(f"Sách: {meta['source']} | Trang: {meta['page']}")

if __name__ == "__main__":
    query = "Bài thuốc chữa đau dạ dày gồm những vị thuốc nào và cách dùng ra sao?"
    ask_rag(query)