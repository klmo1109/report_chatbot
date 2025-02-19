import streamlit as st
from draft.financial_rag import FinancialReportRAG  # 將 RAG 程式碼儲存為 financial_rag.py 並導入

# 初始化 Session 狀態
if "messages" not in st.session_state:
    st.session_state.messages = []

# Streamlit UI
st.title("📊 財報問答助手")

# PDF 來源網址
pdf_url = "https://s201.q4cdn.com/141608511/files/doc_financials/2025/q3/ed2a395c-5e9b-4411-8b4a-a718d192155a.pdf"


# 初始化 RAG 模型（下載 PDF 並建立索引）
@st.cache_resource(show_spinner="正在處理 PDF，請稍候...")
def load_rag_model():
    return FinancialReportRAG(pdf_url)


rag_model = load_rag_model()

# 聊天介面
st.subheader("💬 詢問財報內容")
user_input = st.text_input("請輸入您的問題：")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})

    # 產生 RAG 回應
    with st.spinner("正在搜尋財報內容..."):
        response = rag_model.generate_answer(user_input)

    st.session_state.messages.append({"role": "assistant", "content": response})

# 顯示聊天訊息
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# 清除聊天
if st.button("🛑 清除對話"):
    st.session_state.messages = []
    st.write("對話已清除。")
