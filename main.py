import streamlit as st
import requests
import os
import time
from rag_memory import FinancialReportRAG  # 確保 financial_rag.py 存在並可導入
# PDF 來源網址（默認為 NVIDIA 財報）
nvidia_url = "https://s201.q4cdn.com/141608511/files/doc_financials/2025/q3/ed2a395c-5e9b-4411-8b4a-a718d192155a.pdf"
st.set_page_config(layout="wide", initial_sidebar_state="collapsed")  # 預設隱藏 sidebar


# 初始化 Session 狀態
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "嗨 我是你的文件問答小助手！ 任何問題我都可以回答唷"}]
if "current_pdf_source" not in st.session_state:
    st.session_state.current_pdf_source = None  # 紀錄目前使用的 PDF 來源（網址或本地）
if "pdf_url" not in st.session_state:
    st.session_state.pdf_url = nvidia_url  # 預設網址
if "file_uploader_key" not in st.session_state:
    st.session_state.file_uploader_key = str(time.time())  # 初始化 file uploader 的 key

# Streamlit UI
st.title("文件問答小助手")
st.caption("💬 貼上網址或上傳pdf!讓我來回答你的問題吧！(預設為NVIDIA Q3 2024 Report)")

# **重置對話**
def reset_conversation():
    with st.spinner("正在重置內容..."):
        time.sleep(1)  # 模擬重置過程
        st.session_state.messages = [
            {"role": "assistant", "content": "嗨 我是你的文件問答小助手！ 任何問題我都可以回答唷"}
        ]
        # 清空 PDF 來源
        st.session_state.current_pdf_source = None
        st.session_state.current_pdf_url = None
        st.session_state.pdf_url = nvidia_url  # 這裡修改 pdf_url，而不是 pdf_url_input
        st.session_state.file_uploader_key = str(time.time())  # 強制刷新 file uploader
        st.rerun()  # 重新載入畫面



# 檢查並創建一個臨時目錄來保存上傳的文件
if not os.path.exists("temp_uploads"):
    os.makedirs("temp_uploads")


# **側邊欄**
with st.sidebar:
    st.subheader("設定 PDF 來源")
    text_input_container = st.empty()

    # `text_input` 讀取 `st.session_state.pdf_url`，確保同步
    pdf_url = text_input_container.text_input(
        "請輸入 PDF 網址：", value=st.session_state.pdf_url, key="pdf_url_input", label_visibility="collapsed"
    )

    # 如果使用者輸入新網址，更新 session_state
    if pdf_url != st.session_state.pdf_url:
        st.session_state.pdf_url = pdf_url

    # **上傳本地 PDF**
    pdf_files = st.file_uploader(
        "或上傳你的 PDF 文件", accept_multiple_files=False, type="pdf", key=st.session_state.file_uploader_key
    )

    if pdf_files:
        pdf_path = os.path.join("temp_uploads", pdf_files.name)
        with open(pdf_path, "wb") as f:
            f.write(pdf_files.getbuffer())
        st.success(f"已成功上傳文件：{pdf_files.name}")
        st.session_state.current_pdf_source = pdf_path  # 記錄當前 PDF 來源

    elif pdf_url:  # 若未上傳檔案，則使用 PDF 網址
        st.session_state.current_pdf_source = pdf_url

    # **重新開始對話按鈕**
    if st.button("Restart", key="reset_button"):
        reset_conversation()


# **初始化 RAG 模型**
@st.cache_resource(show_spinner="正在處理 PDF，請稍候...")
def load_rag_model(pdf_source):
    if not pdf_source or pdf_source.strip() == "":
        st.error("⚠️ 無效的 PDF 來源，請確認輸入網址或上傳文件。")
        return None
    try:
        return FinancialReportRAG(pdf_source)
    except FileNotFoundError:
        st.error("❌ 找不到指定的 PDF 文件，請檢查來源是否正確。")
        return None


# **載入 RAG 模型**
if st.session_state.current_pdf_source:
    rag_model = load_rag_model(st.session_state.current_pdf_source)
else:
    st.error("⚠️ 請輸入有效的 PDF 連結或上傳 PDF 檔案。")


# **檢查是否需要重新載入 PDF**
if "current_pdf_url" not in st.session_state or st.session_state.current_pdf_url != st.session_state.current_pdf_source:
    st.session_state.current_pdf_url = st.session_state.current_pdf_source
    st.session_state.messages = [
        {"role": "assistant", "content": "嗨 我是你的文件問答小助手！ 任何問題我都可以回答唷"}
    ]
    rag_model = load_rag_model(st.session_state.current_pdf_source)

# **聊天介面**
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# **輸入問題**
if prompt := st.chat_input("請輸入您的問題："):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 產生 RAG 回應
    with st.spinner("正在搜尋文件內容..."):
        if rag_model:
            response = rag_model.generate_answer(prompt)
        else:
            response = "❌ 無法載入 PDF，請確認 PDF 來源是否有效。"

    with st.chat_message("assistant"):
        st.write(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
#streamlit run main.py