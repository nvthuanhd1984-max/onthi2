# Import thêm bộ đọc CSV
from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader
from langchain_core.documents import Document
import pandas as pd

# ... [CÁC PHẦN CODE ĐẦU FILE GIỮ NGUYÊN] ...

# -----------------------------------------------------------------------------
# ĐOẠN CODE CẬP NHẬT: HỌC TÀI LIỆU (HỖ TRỢ THÊM EXCEL & CSV)
# -----------------------------------------------------------------------------
with st.expander("📄 **Bước 1: Quản lý & Học Tài Liệu (PDF / TXT / EXCEL / CSV)**", expanded=False):
    uploaded_files = st.file_uploader(
        "Tải tài liệu lên để AI nạp kiến thức:", 
        type=["pdf", "txt", "xlsx", "xls", "csv"],  # Đã thêm dạng file Excel & CSV
        accept_multiple_files=True
    )

    if uploaded_files and st.button("📥 Nạp Dữ Liệu Vào AI"):
        with st.spinner("Đang đọc và xử lý dữ liệu..."):
            all_docs = []
            for uploaded_file in uploaded_files:
                file_ext = os.path.splitext(uploaded_file.name)[1].lower()
                
                with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    tmp_path = tmp_file.name

                # Xử lý theo từng loại file
                if file_ext == ".pdf":
                    loader = PyPDFLoader(tmp_path)
                    docs = loader.load()
                elif file_ext == ".txt":
                    loader = TextLoader(tmp_path, encoding="utf-8")
                    docs = loader.load()
                elif file_ext == ".csv":
                    loader = CSVLoader(tmp_path, encoding="utf-8")
                    docs = loader.load()
                elif file_ext in [".xlsx", ".xls"]:
                    # Đọc file Excel bằng pandas và chuyển mỗi hàng/bảng thành văn bản cho AI học
                    excel_data = pd.read_excel(tmp_path, sheet_name=None)
                    docs = []
                    for sheet_name, df in excel_data.items():
                        # Chuyển dữ liệu bảng thành dạng text mô tả dễ hiểu
                        text_content = f"Sheet: {sheet_name}\n" + df.to_string(index=False)
                        docs.append(Document(page_content=text_content, metadata={"source": uploaded_file.name, "sheet": sheet_name}))
                
                all_docs.extend(docs)
                os.remove(tmp_path)

            # Cắt nhỏ văn bản tạo vector
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            splits = text_splitter.split_documents(all_docs)

            vectorstore.add_documents(splits)
            st.success(f"✅ Đã nạp thành công dữ liệu từ {len(uploaded_files)} tệp!")
