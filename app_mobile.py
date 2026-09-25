import os
import tempfile
import base64
import streamlit as st

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

# -----------------------------------------------------------------------------
# 1. CẤU HÌNH TRANG WEB CHUẨN CHO DI ĐỘNG (MOBILE-FIRST)
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Trợ Lý AI Di Động",
    page_icon="📱",
    layout="centered",  # Tối ưu giao diện thu gọn cho màn hình dọc
    initial_sidebar_state="collapsed"  # Ẩn Sidebar ban đầu để tiết kiệm diện tích màn hình
)

# Custom CSS giúp tối ưu các nút bấm to hơn, dễ chạm bằng ngón tay
st.markdown("""
    <style>
        .stButton>button {
            width: 100%;
            height: 3em;
            font-size: 18px !important;
            font-weight: bold;
            border-radius: 10px;
        }
        .stCameraInput {
            width: 100%;
        }
    </style>
""", unsafe_allow_html=True)

st.title("📱 Trợ Lý AI Đa Phương Thức")
st.caption("Tải tài liệu tự học -> Chụp ảnh/Tải ảnh từ điện thoại -> Nhận câu trả lời.")

# -----------------------------------------------------------------------------
# 2. CẤU HÌNH API KEY (Lấy từ Secrets hoặc Nhập tay)
# -----------------------------------------------------------------------------
# Ưu tiên lấy API Key từ Streamlit Secrets nếu đã cấu hình khi deploy Cloud
google_api_key = st.secrets.get("GOOGLE_API_KEY", "")

with st.sidebar:
    st.header("⚙️ Cấu hình Hệ thống")
    if not google_api_key:
        google_api_key = st.text_input("Nhập Google Gemini API Key:", type="password")
        if not google_api_key:
            st.warning("⚠️ Vui lòng nhập API Key để ứng dụng hoạt động!")
            st.stop()
    else:
        st.success("✅ Đã kết nối API Key từ Cấu hình Server!")

os.environ["GOOGLE_API_KEY"] = google_api_key
PERSIST_DIR = "./chroma_db_mobile"

# -----------------------------------------------------------------------------
# 3. HỌC TÀI LIỆU (DATABASE & EMBEDDINGS)
# -----------------------------------------------------------------------------
@st.cache_resource
def get_vectorstore(api_key):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=api_key
    )
    return Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)

vectorstore = get_vectorstore(google_api_key)

with st.expander("📄 **Bước 1: Quản lý & Học Tài Liệu (PDF / TXT)**", expanded=False):
    uploaded_files = st.file_uploader(
        "Tải tài liệu lên để AI nạp kiến thức:", 
        type=["pdf", "txt"], 
        accept_multiple_files=True
    )

    if uploaded_files and st.button("📥 Nạp Dữ Liệu Vào AI"):
        with st.spinner("Đang xử lý tài liệu..."):
            all_docs = []
            for uploaded_file in uploaded_files:
                with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    tmp_path = tmp_file.name

                if uploaded_file.name.endswith(".pdf"):
                    loader = PyPDFLoader(tmp_path)
                else:
                    loader = TextLoader(tmp_path, encoding="utf-8")
                
                docs = loader.load()
                all_docs.extend(docs)
                os.remove(tmp_path)

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            splits = text_splitter.split_documents(all_docs)

            vectorstore.add_documents(splits)
            st.success(f"✅ Đã nạp thành công {len(splits)} đoạn tri thức!")

# -----------------------------------------------------------------------------
# 4. THU NHẬN HÌNH ẢNH TỪ ĐIỆN THOẠI (CAMERA / GALLERY)
# -----------------------------------------------------------------------------
st.subheader("📷 Bước 2: Dữ liệu Hình ảnh")

# Tab lựa chọn hình thức nhập ảnh phù hợp trên điện thoại
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

# -----------------------------------------------------------------------------
# 5. XỬ LÝ & TRẢ LỜI CÂU HỎI BẰNG GEMINI 1.5 FLASH
# -----------------------------------------------------------------------------
st.subheader("💬 Bước 3: Phân tích & Hỏi đáp")

user_query = st.text_input(
    "Nhập yêu cầu / câu hỏi bổ sung:", 
    value="Hãy giải thích hoặc trả lời nội dung trong ảnh này dựa trên tài liệu đã học."
)

if image_to_process:
    st.image(image_to_process, caption="Ảnh được chọn", use_container_width=True)
    
    if st.button("🚀 BẮT ĐẦU PHÂN TÍCH", type="primary"):
        with st.spinner("AI đang truy vấn tài liệu & phân tích ảnh..."):
            try:
                # Mã hóa ảnh sang Base64
                image_bytes = image_to_process.getvalue()
                base64_image = base64.b64encode(image_bytes).decode('utf-8')

                # Tìm kiếm thông tin liên quan trong Vector Store
                retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
                relevant_docs = retriever.invoke(user_query)
                context_text = "\n\n".join([doc.page_content for doc in relevant_docs])

                # Khởi tạo Mô hình Gemini 1.5 Flash
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

                with st.expander("🔍 Nguồn trích dẫn từ tài liệu đã học"):
                    for i, doc in enumerate(relevant_docs, 1):
                        st.write(f"**Đoạn {i}:** {doc.page_content[:200]}...")

            except Exception as e:
                st.error(f"Đã xảy ra lỗi: {str(e)}")
else:
    st.info("💡 Hãy chụp hoặc chọn 1 bức ảnh ở **Bước 2** để bắt đầu.")
