import os
import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
from google import genai
from google.genai import types
from sentence_transformers import CrossEncoder

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

print("Đang tải mô hình Reranker (BAAI/bge-reranker-v2-m3)...")
reranker = CrossEncoder("BAAI/bge-reranker-v2-m3")

SYSTEM_PROMPT = """
Bạn là một chuyên gia Y học cổ truyền Việt Nam.
Nhiệm vụ của bạn là trả lời thắc mắc của người dùng dựa TRỰC TIẾP vào BỐI CẢNH TÀI LIỆU được cung cấp và LỊCH SỬ TRÒ CHUYỆN.

Quy tắc bắt buộc:
1. Chỉ sử dụng thông tin có trong BỐI CẢNH TÀI LIỆU. Nếu tài liệu không có thông tin, hãy trả lời rõ: "Tài liệu hiện tại không đề cập đến thông tin này".
2. Trả lời rõ ràng, chính xác về tên vị thuốc, bài thuốc, liều lượng, công dụng và cách dùng (nếu có).
3. Luôn duy trì văn phong lịch sự, chu đáo và chuyên nghiệp.
"""

def generate_standalone_query(chat_history, user_question):
    """Chuyển câu hỏi nối tiếp thành câu truy vấn độc lập chứa đầy đủ ngữ cảnh."""
    if not chat_history:
        return user_question
    
    history_str = ""
    for msg in chat_history[-6:]: 
        role = "Người dùng" if msg["role"] == "user" else "Chatbot"
        history_str += f"{role}: {msg['content']}\n"

    prompt = f"""Dựa vào Lịch sử trò chuyện và Câu hỏi mới nhất, hãy viết lại câu hỏi mới thành MỘT CÂU TRUY VẤN ĐỘC LẬP đầy đủ ngữ cảnh để tìm kiếm trong CSDL sách Y học cổ truyền.
Chỉ trả về duy nhất câu truy vấn được viết lại, không giải thích hay chào hỏi.

=== LỊCH SỬ TRÒ CHUYỆN ===
{history_str}

=== CÂU HỎI MỚI ===
{user_question}

Câu truy vấn tìm kiếm:"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(temperature=0.1)
    )
    return response.text.strip()

def search_and_rerank(search_query, initial_top_k=15, final_top_k=6):
    """Tìm kiếm thô bằng Vector DB và sắp xếp lại bằng Reranker Model."""
    results = collection.query(
        query_texts=[search_query],
        n_results=initial_top_k
    )
    
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    
    if not docs:
        return [], []

    pairs = [[search_query, doc] for doc in docs]
    scores = reranker.predict(pairs)

    scored_results = list(zip(scores, docs, metas))
    scored_results.sort(key=lambda x: x[0], reverse=True)

    top_results = scored_results[:final_top_k]
    
    reranked_docs = [item[1] for item in top_results]
    reranked_metas = [item[2] for item in top_results]
    
    return reranked_docs, reranked_metas

def chat_cli():
    chat_history = [] 
    
    print("\n" + "="*65)
    print("CHATBOT Y HỌC CỔ TRUYỀN VIỆT NAM (RAG + RERANKER + MEMORY) 🌿")
    print("Gõ 'exit' hoặc 'quit' để kết thúc trò chuyện.")
    print("="*65 + "\n")

    while True:
        try:
            user_input = input("Bạn: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ["exit", "quit"]:
                print("\nCảm ơn bạn đã sử dụng hệ thống. Tạm biệt!")
                break

            standalone_query = generate_standalone_query(chat_history, user_input)
            if standalone_query != user_input and len(chat_history) > 0:
                print(f"[Rewritten Query]: '{standalone_query}'")

            print("Đang tìm kiếm & xếp hạng tài liệu liên quan...")
            docs, metas = search_and_rerank(standalone_query, initial_top_k=15, final_top_k=6)

            context_blocks = []
            for doc, meta in zip(docs, metas):
                context_blocks.append(f"[Nguồn: {meta['source']} | Trang: {meta['page']}]\n{doc}")
            context_str = "\n\n---\n\n".join(context_blocks)

            history_str = ""
            for msg in chat_history[-6:]:
                role = "Người dùng" if msg["role"] == "user" else "Chatbot"
                history_str += f"{role}: {msg['content']}\n"

            prompt = f"""=== BỐI CẢNH TÀI LIỆU KHÔI PHỤC ===
{context_str}

=== LỊCH SỬ TRÒ CHUYỆN ===
{history_str}

=== CÂU HỎI MỚI CỦA NGƯỜI DÙNG ===
{user_input}
"""

            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    temperature=0.2
                )
            )

            answer = response.text.strip()

            print("\nChatbot:")
            print(answer)
            print("\nNguồn trích dẫn:")
            seen_sources = set()
            for meta in metas:
                source_key = f"{meta['source']} (Trang {meta['page']})"
                if source_key not in seen_sources:
                    seen_sources.add(source_key)
                    print(f"{source_key}")
            print("-" * 65 + "\n")

            chat_history.append({"role": "user", "content": user_input})
            chat_history.append({"role": "assistant", "content": answer})

        except Exception as e:
            print(f"\nĐã xảy ra lỗi: {e}\n")

if __name__ == "__main__":
    chat_cli()