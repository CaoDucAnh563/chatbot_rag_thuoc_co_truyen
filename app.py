import os
import streamlit as st
import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
from google import genai
from google.genai import types
from sentence_transformers import CrossEncoder

st.set_page_config(
    page_title="Chatbot Y Học Cổ Truyền",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .stChatMessage {
        border-radius: 10px;
        padding: 12px;
        margin-bottom: 10px;
    }
    .source-card {
        background-color: #f8f9fa;
        border-left: 4px solid #2e7d32;
        padding: 10px 15px;
        margin-bottom: 8px;
        border-radius: 4px;
        font-size: 0.9em;
    }
</style>
""", unsafe_allow_html=True)

env_path = os.path.join(os.path.dirname(__file__), "tools_and_tests", ".env")
load_dotenv(dotenv_path=env_path)

@st.cache_resource
def load_gemini_client():
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        st.error("Chưa tìm thấy GOOGLE_API_KEY trong file .env!")
        st.stop()
    return genai.Client(api_key=api_key)

@st.cache_resource
def load_chroma_collection():
    db_path = os.path.join(os.path.dirname(__file__), "data", "chroma_db")
    vietnamese_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="bkai-foundation-models/vietnamese-bi-encoder"
    )
    chroma_client = chromadb.PersistentClient(path=db_path)
    return chroma_client.get_collection(
        name="y_hoc_co_truyen",
        embedding_function=vietnamese_ef
    )

@st.cache_resource
def load_reranker_model():
    return CrossEncoder("BAAI/bge-reranker-v2-m3")

with st.spinner("Đang khởi tạo mô hình và cơ sở dữ liệu Vector..."):
    client = load_gemini_client()
    collection = load_chroma_collection()
    reranker = load_reranker_model()

st.sidebar.title("Cấu hình RAG")
st.sidebar.markdown("Tùy chỉnh thông số truy vấn CSDL Y học cổ truyền.")

SOURCE_MAP = {
    "Hướng dẫn chẩn đoán và điều trị bệnh": "huong_dan_chan_doan",
    "800 Bài thuốc tâm huyết của Danh Y": "800_danh_y",
    "Nam dược thần hiệu": "nam_duoc_than_hieu_pdf",
    "Những cây thuốc và vị thuốc VN": "nhung_cay_thuoc_va_vi_thuoc_viet_nam_vnras",
    "Toa thuốc Đông y cổ truyền VN": "toa_thuoc_dong_y_co_truyen_viet_nam"
}

selected_book_names = st.sidebar.multiselect(
    "Lọc nguồn sách tra cứu:",
    options=list(SOURCE_MAP.keys()),
    default=list(SOURCE_MAP.keys()),
    help="Chọn một hoặc nhiều cuốn sách bạn muốn tìm kiếm thông tin."
)

use_reranker = st.sidebar.toggle("Bật Reranker (BGE-M3)", value=True, help="Sử dụng Cross-Encoder để xếp hạng lại tài liệu chính xác hơn.")

top_k = st.sidebar.slider("Số đoạn tài liệu lấy (Top K):", min_value=2, max_value=10, value=5)

st.sidebar.divider()

if st.sidebar.button("Xóa lịch sử trò chuyện", use_container_width=True):
    st.session_state.messages = []
    st.rerun()

st.sidebar.caption("Sản phẩm hỗ trợ tra cứu Y học cổ truyền bằng AI & RAG.")

SYSTEM_PROMPT = """
Bạn là một chuyên gia Y học cổ truyền Việt Nam.
Nhiệm vụ của bạn là trả lời thắc mắc của người dùng dựa TRỰC TIẾP vào BỐI CẢNH TÀI LIỆU được cung cấp và LỊCH SỬ TRÒ CHUYỆN.

Quy tắc bắt buộc:
1. Chỉ sử dụng thông tin có trong BỐI CẢNH TÀI LIỆU. Nếu tài liệu không có thông tin, hãy trả lời rõ: "Tài liệu hiện tại không đề cập đến thông tin này".
2. Trả lời rõ ràng, chính xác về tên vị thuốc, bài thuốc, liều lượng, công dụng và cách dùng (nếu có).
3. Luôn duy trì văn phong lịch sự, chu đáo và chuyên nghiệp.
"""

def generate_standalone_query(chat_history, user_question):
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

def search_and_rerank(query_text, selected_sources, is_rerank_enabled, final_k):
    where_clause = None
    if len(selected_sources) == 1:
        where_clause = {"source": selected_sources[0]}
    elif len(selected_sources) > 1 and len(selected_sources) < len(SOURCE_MAP):
        where_clause = {"source": {"$in": selected_sources}}

    initial_k = final_k * 3 if is_rerank_enabled else final_k

    results = collection.query(
        query_texts=[query_text],
        n_results=initial_k,
        where=where_clause
    )

    docs = results["documents"][0]
    metas = results["metadatas"][0]

    if not docs:
        return [], []

    if not is_rerank_enabled:
        return docs[:final_k], metas[:final_k]

    pairs = [[query_text, doc] for doc in docs]
    scores = reranker.predict(pairs)

    scored_results = list(zip(scores, docs, metas))
    scored_results.sort(key=lambda x: x[0], reverse=True)

    top_results = scored_results[:final_k]
    return [item[1] for item in top_results], [item[2] for item in top_results]

st.title("Chatbot Tra Cứu Y Học Cổ Truyền")
st.caption("Hệ thống RAG thông minh tra cứu từ gần 4,000 trang sách Đông y đã được số hóa.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message and message["sources"]:
            with st.expander("Trích dẫn nguồn tài liệu"):
                for src in message["sources"]:
                    st.markdown(f"<div class='source-card'><b> Sách:</b> {src['source']}<br><b> Trang:</b> {src['page']}<br><i>{src['content']}</i></div>", unsafe_allow_html=True)

if user_input := st.chat_input("Nhập câu hỏi về bài thuốc, vị thuốc hoặc chứng bệnh..."):
    st.chat_message("user").markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    selected_sources_keys = [SOURCE_MAP[name] for name in selected_book_names]

    if not selected_sources_keys:
        st.warning("Bạn hãy chọn ít nhất 1 cuốn sách trong sidebar để tra cứu!")
        st.stop()

    with st.chat_message("assistant"):
        with st.status("Đang truy vấn CSDL & Xử lý câu trả lời...", expanded=False) as status:
            standalone_query = generate_standalone_query(st.session_state.messages[:-1], user_input)
            if standalone_query != user_input and len(st.session_state.messages) > 1:
                st.write(f"Viết lại câu hỏi:{standalone_query}")

            docs, metas = search_and_rerank(
                query_text=standalone_query,
                selected_sources=selected_sources_keys,
                is_rerank_enabled=use_reranker,
                final_k=top_k
            )

            context_blocks = [f"[Nguồn: {m['source']} | Trang: {m['page']}]\n{d}" for d, m in zip(docs, metas)]
            context_str = "\n\n---\n\n".join(context_blocks)

            history_str = ""
            for msg in st.session_state.messages[-7:-1]:
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
            answer_text = response.text.strip()
            status.update(label="Đã tìm thấy câu trả lời!", state="complete")

        st.markdown(answer_text)

        sources_data = []
        if docs:
            with st.expander("Trích dẫn nguồn tài liệu tham khảo"):
                for doc, meta in zip(docs, metas):
                    src_info = {
                        "source": meta["source"],
                        "page": meta["page"],
                        "content": doc[:200] + "..."
                    }
                    sources_data.append(src_info)
                    st.markdown(
                        f"<div class='source-card'>"
                        f"<b> Sách:</b> <code>{meta['source']}</code> | <b> Trang:</b> {meta['page']}<br>"
                        f"<b>Nội dung:</b> <i>{doc}</i>"
                        f"</div>",
                        unsafe_allow_html=True
                    )

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer_text,
            "sources": sources_data
        })