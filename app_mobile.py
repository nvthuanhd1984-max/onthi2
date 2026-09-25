# --- ĐOẠN CODE GHI ĐÈ SQLITE TỐI ƯU CHO STREAMLIT CLOUD (ĐẶT Ở ĐẦU FILE) ---
import sys
try:
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass
# ----------------------------------------------------------------------

import os
import tempfile
import base64
import pandas as pd
import streamlit as st

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

# -----------------------------------------------------------------------------
# 1. CẤU HÌNH GIAO DIỆN WEB DI ĐỘNG
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Trợ Lý AI Di Động",
    page_icon="📱",
    layout="centered",
    initial_sidebar_state="collapsed"
)

st.markdown("""
    <style>
        .stButton>button {
            width: 100%;
            height: 3em;
            font-size: 18px !important;
            font-weight: bold;
            border-radius: 10px;
        }
    </style>
""", unsafe_allow_html=True)

st.title("📱 Trợ Lý AI Đa Phương Thức")
st.caption("Nạp tài liệu -> Chụp/Chọn ảnh -> AI phân tích và trả lời.")

# -----------------------------------------------------------------------------
# 2. XỬ LÝ API KEY TỰ ĐỘNG KHÔNG LÀM DỪNG ỨNG DỤNG
# -----------------------------------------------------------------------------
google_api_key = st.secrets.get("GOOGLE_API_KEY", "")

with st.sidebar:
    st.header("⚙️ Cấu hình Hệ thống")
    if not google_api_key:
        google_api_key = st.text_input("Nhập Google Gemini API Key:", type="password")
        if google_api_key:
            st.success("✅ Đã nhận API Key!")
    else:
        st.success("✅ Đã kết nối API Key từ Secrets!")

PERSIST_DIR = "./chroma_db_mobile"

@st.cache_resource
def get_vectorstore(api_key):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=api_key
    )
    return Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)

# Khởi tạo vectorstore an toàn nếu đã có API Key
vectorstore = None
if google_api_key:
    os.environ["GOOGLE_API_KEY"] = google_api_key
    try:
        vectorstore = get_vectorstore(google_api_key)
    except Exception as e:
        st.error(f"Lỗi khởi tạo VectorStore: {str(e)}")

# -----------------------------------------------------------------------------
# 3. BƯỚC 1: HỌC TÀI LIỆU (PDF / TXT / EXCEL / CSV)
# -----------------------------------------------------------------------------
with st.expander("📄 **Bước 1: Quản lý & Học Tài Liệu (PDF / TXT / EXCEL / CSV)**", expanded=False):
    uploaded_files = st.file_uploader(
        "Tải tài liệu lên để AI nạp kiến thức:", 
        type=["pdf", "txt", "xlsx", "xls", "csv"],
        accept_multiple_files=True
    )

    if uploaded_files and st.button("📥 Nạp Dữ Liệu Vào AI"):
        if not google_api_key:
            st.error("⚠️ Vui lòng nhập Google Gemini API Key ở menu Sidebar trước khi nạp dữ liệu!")
        else:
            with st.spinner("Đang đọc và xử lý dữ liệu..."):
                all_docs = []
                for uploaded_file in uploaded_files:
                    file_ext = os.path.splitext(uploaded_file.name)[1].lower()
                    
                    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=file_ext)
                    tmp_file.write(uploaded_file.read())
                    tmp_path = tmp_file.name
                    tmp_file.close()

                    try:
                        docs = []
                        if file_ext == ".pdf":
                            loader = PyPDFLoader(tmp_path)
                            docs = loader.load()
                        elif file_ext == ".txt":
                            try:
                                loader = TextLoader(tmp_path, encoding="utf-8")
                                docs = loader.load()
                            except UnicodeDecodeError:
                                loader = TextLoader(tmp_path, encoding="utf-8-sig")
                                docs = loader.load()
                        elif file_ext == ".csv":
                            try:
                                df = pd.read_csv(tmp_path, encoding="utf-8")
                            except Exception:
                                df = pd.read_csv(tmp_path, encoding="utf-8-sig")
                            
                            for idx, row in df.iterrows():
                                row_str = ", ".join([f"{col}: {val}" for col, val in row.items()])
                                docs.append(Document(
                                    page_content=row_str,
                                    metadata={"source": uploaded_file.name, "row": idx}
                                ))
                        elif file_ext in [".xlsx", ".xls"]:
                            excel_data = pd.read_excel(tmp_path, sheet_name=None)
                            for sheet_name, df in excel_data.items():
                                for idx, row in df.iterrows():
                                    row_str = f"Sheet: {sheet_name} | " + ", ".join([f"{col}: {val}" for col, val in row.items()])
                                    docs.append(Document(
                                        page_content=row_str,
                                        metadata={"source": uploaded_file.name, "sheet": sheet_name, "row": idx}
                                    ))
                        all_docs.extend(docs)
                    except Exception as e:
                        st.error(f"Lỗi đọc file {uploaded_file.name}: {str(e)}")
                    finally:
                        if os.path.exists(tmp_path):
                            os.remove(tmp_path)

                if all_docs and vectorstore is not None:
                    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                    splits = text_splitter.split_documents(all_docs)
                    vectorstore.add_documents(splits)
                    st.success(f"✅ Đã nạp thành công {len(splits)} đoạn tri thức vào AI!")

st.divider()

# -----------------------------------------------------------------------------
# 4. BƯỚC 2: DỮ LIỆU HÌNH ẢNH (CAMERA / GALLERY)
# -----------------------------------------------------------------------------
st.subheader("📷 Bước 2: Dữ liệu Hình ảnh")

tab_cam, tab_file = st.tabs(["📸 Chụp từ Camera", "🖼️ Chọn từ Thư viện ảnh"])
image_to_process = None

with tab_cam:
    cam_img = st.camera_input("Chụp ảnh trực tiếp:")
    if cam_img:
        image_to_process = cam_img

with tab_file:
    file_img = st.file_uploader("Tải ảnh có sẵn trong máy:", type=["jpg", "jpeg", "png"])
    if file_img:
        image_to_process = file_img

st.divider()

# -----------------------------------------------------------------------------
# 5. BƯỚC 3: PHÂN TÍCH & TRẢ LỜI CÂU HỎI
# -----------------------------------------------------------------------------
st.subheader("💬 Bước 3: Phân tích & Hỏi đáp")

user_query = st.text_input(
    "Nhập yêu cầu / câu hỏi bổ sung:", 
    value="Hãy giải thích hoặc trả lời nội dung trong ảnh này dựa trên tài liệu đã học."
)

if image_to_process:
    st.image(image_to_process, caption="Ảnh được chọn", use_container_width=True)
    
    if st.button("🚀 BẮT ĐẦU PHÂN TÍCH", type="primary"):
        if not google_api_key:
            st.error("⚠️ Bạn cần nhập Google Gemini API Key ở Sidebar trước khi phân tích!")
        else:
            with st.spinner("AI đang truy vấn tài liệu & phân tích ảnh..."):
                try:
                    image_bytes = image_to_process.getvalue()
                    base64_image = base64.b64encode(image_bytes).decode('utf-8')

                    context_text = ""
                    relevant_docs = []
                    if vectorstore is not None:
                        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
                        relevant_docs = retriever.invoke(user_query)
                        context_text = "\n\n".join([doc.page_content for doc in relevant_docs])

                    llm = ChatGoogleGenerativeAI(
                        model="gemini-1.5-flash",
                        google_api_key=google_api_key,
                        temperature=0.2
                    )

                    system_prompt = (
                        "Bạn là trợ lý AI thông minh trên thiết bị di động.\n"
                        "Sử dụng tri thức từ tài liệu nội bộ bên dưới ĐỂ GIẢI THÍCH, TRẢ LỜI hoặc GIẢI BÀI TẬP trong hình ảnh.\n\n"
                        f"=== TRI THỨC TỪ TÀI LIỆU NỘI BỘ ===\n{context_text}\n================================="
                    )

                    message = HumanMessage(
                        content=[
                            {"type": "text", "text": f"{system_prompt}\n\nCâu hỏi người dùng: {user_query}"},
                            {
                                "type": "image_url",
                                "image_url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        ]
                    )

                    response = llm.invoke([message])

                    st.markdown("### 💡 Kết quả phân tích:")
                    st.write(response.content)

                    if relevant_docs:
                        with st.expander("🔍 Nguồn trích dẫn từ tài liệu đã học"):
                            for i, doc in enumerate(relevant_docs, 1):
                                st.write(f"**Đoạn {i}:** {doc.page_content[:200]}...")

                except Exception as e:
                    st.error(f"Đã xảy ra lỗi: {str(e)}")
else:
    st.info("💡 Hãy chụp hoặc chọn 1 bức ảnh ở **Bước 2** để bắt đầu phân tích.")
