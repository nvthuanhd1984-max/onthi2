import os
import tempfile
import pandas as pd
import streamlit as st

from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# -----------------------------------------------------------------------------
# ĐOẠN CODE CẬP NHẬT: HỌC TÀI LIỆU (ĐÃ FIX TOÀN BỘ LỖI EXCEL, CSV, TEMPFILE)
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
                
                # 1. TẠO FILE TẠM AN TOÀN TRÊN CẢ WINDOWS & LINUX
                tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=file_ext)
                tmp_file.write(uploaded_file.read())
                tmp_path = tmp_file.name
                tmp_file.close()  # Đóng ngay để tránh lỗi PermissionError khi đọc/xóa

                try:
                    docs = []
                    # 2. XỬ LÝ THEO TỪNG ĐỊNH DẠNG FILE
                    if file_ext == ".pdf":
                        loader = PyPDFLoader(tmp_path)
                        docs = loader.load()

                    elif file_ext == ".txt":
                        # Thử utf-8, nếu lỗi đổi sang utf-8-sig
                        try:
                            loader = TextLoader(tmp_path, encoding="utf-8")
                            docs = loader.load()
                        except UnicodeDecodeError:
                            loader = TextLoader(tmp_path, encoding="utf-8-sig")
                            docs = loader.load()

                    elif file_ext == ".csv":
                        # Dùng pandas đọc CSV để tránh lỗi mã hóa ký tự
                        try:
                            df = pd.read_csv(tmp_path, encoding="utf-8")
                        except Exception:
                            df = pd.read_csv(tmp_path, encoding="utf-8-sig")
                        
                        # Chuyển từng hàng thành 1 Document có kèm tiêu đề cột
                        for idx, row in df.iterrows():
                            row_str = ", ".join([f"{col}: {val}" for col, val in row.items()])
                            docs.append(Document(
                                page_content=row_str,
                                metadata={"source": uploaded_file.name, "row": idx}
                            ))

                    elif file_ext in [".xlsx", ".xls"]:
                        # Đọc file Excel bằng pandas + openpyxl
                        excel_data = pd.read_excel(tmp_path, sheet_name=None)
                        
                        for sheet_name, df in excel_data.items():
                            # Chuyển từng dòng của sheet thành văn bản cấu trúc rõ ràng
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
                    # Dọn dẹp file tạm an toàn
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)

            # 3. TẠO VECTOR & NẠP VÀO DATABASE
            if all_docs:
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                splits = text_splitter.split_documents(all_docs)

                vectorstore.add_documents(splits)
                st.success(f"✅ Đã nạp thành công dữ liệu từ {len(uploaded_files)} tệp ({len(splits)} đoạn tri thức)!")
            else:
                st.warning("⚠️ Không tìm thấy nội dung hợp lệ trong các tệp đã tải lên.")
