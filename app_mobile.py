import os
import tempfile
import pandas as pd
import streamlit as st

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# -----------------------------------------------------------------------------
# 1. BẮT BỘC KHỞI TẠO API KEY VÀ VECTORSTORE TRƯỚC
# -----------------------------------------------------------------------------
google_api_key = st.secrets.get("GOOGLE_API_KEY", "")

with st.sidebar:
    st.header("⚙️ Cấu hình Hệ thống")
    if not google_api_key:
        google_api_key = st.text_input("Nhập Google Gemini API Key:", type="password")
        if not google_api_key:
            st.warning("⚠️ Vui lòng nhập API Key để ứng dụng hoạt động!")
            st.stop()

os.environ["GOOGLE_API_KEY"] = google_api_key
PERSIST_DIR = "./chroma_db_mobile"

# Định nghĩa hàm lấy vectorstore
@st.cache_resource
def get_vectorstore(api_key):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=api_key
    )
    return Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)

# 👉 KHỞI TẠO BIẾN VECTORSTORE TOÀN CỤC Ở ĐÂY (ĐẢM BẢO KHÔNG BỊ LỖI NameError)
vectorstore = get_vectorstore(google_api_key)

# -----------------------------------------------------------------------------
# 2. XỬ LÝ NẠP TÀI LIỆU (PDF / TXT / EXCEL / CSV)
# -----------------------------------------------------------------------------
with st.expander("📄 **Bước 1: Quản lý & Học Tài Liệu (PDF / TXT / EXCEL / CSV)**", expanded=False):
    uploaded_files = st.file_uploader(
        "Tải tài liệu lên để AI nạp kiến thức:", 
        type=["pdf", "txt", "xlsx", "xls", "csv"],
        accept_multiple_files=True
    )

    if uploaded_files and st.button("📥 Nạp Dữ Liệu Vào AI"):
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
                    st.error(f"Lỗi khi xử lý file {uploaded_file.name}: {str(e)}")
                
                finally:
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)

            if all_docs:
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                splits = text_splitter.split_documents(all_docs)

                # Gọi biến vectorstore đã khởi tạo sẵn ở trên
                vectorstore.add_documents(splits)
                st.success(f"✅ Đã nạp thành công {len(splits)} đoạn tri thức vào AI!")
