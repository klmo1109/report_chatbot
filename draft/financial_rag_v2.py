import os
import numpy as np
import faiss
import requests
from dotenv import load_dotenv
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory

# 載入環境變數
load_dotenv()
os.environ[
    "USER_AGENT"] = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
openai_api_key = os.getenv("OPENAI_API_KEY")

class FinancialReportRAG:
    def __init__(self, input_source):
        """
        初始化 RAG 模型
        :param input_source: 可以是 PDF 網址或本地 PDF 文件路徑
        """
        if input_source.startswith("http"):
            # 如果是網址，下載 PDF
            self.pdf_path = self._download_pdf(input_source)
        else:
            # 如果是本地文件，直接使用
            self.pdf_path = input_source

        self.texts_with_pages = self._load_and_process_pdf()
        self.embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
        self.index, self.texts_with_pages = self._build_faiss_index()
        self.llm = ChatOpenAI(model="gpt-4", temperature=0)
        self.language = self._detect_language()


    def _download_pdf(self, url):
        """從網址下載 PDF 並存儲到本地"""
        pdf_filename = "financial_report.pdf"
        response = requests.get(url, stream=True)

        if response.status_code == 200:
            with open(pdf_filename, "wb") as f:
                for chunk in response.iter_content(chunk_size=1024):
                    f.write(chunk)
            return pdf_filename
        else:
            raise Exception(f"下載 PDF 失敗，狀態碼: {response.status_code}")

    def _load_and_process_pdf(self):
        """讀取 PDF 並使用 Overlapping Window 切割文本"""
        reader = PdfReader(self.pdf_path)
        texts_with_pages = []

        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if text:
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=800,  # ✅ 調整 chunk 大小
                    chunk_overlap=400,  # ✅ 增加 Overlapping Window，確保內容連貫
                    separators=["\n\n", "\n", " ", ""]
                )
                chunks = text_splitter.split_text(text)
                for chunk in chunks:
                    texts_with_pages.append((i + 1, chunk))

        return texts_with_pages

    def _build_faiss_index(self):
        """生成嵌入向量並建立 FAISS 搜索索引"""
        texts = [t[1] for t in self.texts_with_pages]  # 只取文本
        print(f"處理文本數量: {len(texts)}")  # 確保有文本輸入

        if not texts:
            raise ValueError("無法從 PDF 提取文本，請確認 PDF 內容是否可讀取。")

        embedded_texts = self.embeddings.embed_documents(texts)
        #print(f"嵌入數據: {embedded_texts}")  # 確保嵌入數據不為空

        if embedded_texts is None or len(embedded_texts) == 0:
            raise ValueError("嵌入生成失敗，請確認 OpenAI API 金鑰是否有效，或文本是否可用。")

        embedded_texts = np.array(embedded_texts, dtype=np.float32)

        if embedded_texts.shape[0] == 0:
            raise ValueError("嵌入矩陣為空，無法建立 FAISS 索引。請檢查前面步驟是否成功。")

        # 設置 FAISS 索引
        dimension = embedded_texts.shape[1]
        print(f"FAISS 索引維度: {dimension}")

        index = faiss.IndexFlatL2(dimension)
        index.add(embedded_texts)

        return index, self.texts_with_pages

    def _detect_language(self):
        """判斷文本是中文還是英文"""
        sample_text = " ".join([t[1] for t in self.texts_with_pages[:10]])  # 取前 10 段文本作為樣本
        chinese_char_count = sum(1 for char in sample_text if '\u4e00' <= char <= '\u9fff')  # 統計中文字數

        # 如果中文字數超過 10%，則判斷為中文
        if chinese_char_count / len(sample_text) > 0.1:
            return "zh"  # 中文
        else:
            return "en"  # 英文

    def _get_question_embedding(self, question):
        """取得問題的嵌入向量"""
        return np.array(self.embeddings.embed_query(question), dtype=np.float32)

    def _translate_question(self, question):
        """將問題翻譯成英文（如果文本是英文）"""
        if self.language == "en":
            prompt = PromptTemplate(
                input_variables=["text"],
                template="將這句話翻譯成英文：{text}",
            )
            translated_question = (prompt | self.llm).invoke({"text": question}).content
            return translated_question
        else:
            return question  # 如果是中文，直接返回原始問題

    def retrieve_relevant_texts(self, question, k=3):
        """根據問題檢索最相關的文本片段"""
        question_embedding = self._get_question_embedding(question)
        distances, indices = self.index.search(np.array([question_embedding]), k)

        results = []
        for i in indices[0]:
            page_num, text = self.texts_with_pages[i]
            results.append((page_num, text))

        return results

    def generate_answer(self, question):
        """使用 LLM 生成回答，並確保提供正確的來源頁碼"""
        # 根據語言翻譯問題
        translated_question = self._translate_question(question)

        # 檢索相關文本
        relevant_texts = self.retrieve_relevant_texts(translated_question)

        # 如果沒有找到足夠的相關內容，直接返回固定訊息
        if not relevant_texts:
            return "⚠️ 文件未提供相關資訊，請嘗試不同的問題或參考官方資料來源。"
        ranked_texts = sorted(relevant_texts, key=lambda x: len(x[1]), reverse=True)[:4]  # 取最相關的3個

        # 整理 Context 內容
        context = "\n\n".join([f"(頁碼: {p}) {t}" for p, t in relevant_texts])
        source_pages = {p for p, _ in relevant_texts}  # 記錄來源頁碼
        source_pages_str = ", ".join(map(str, sorted(source_pages)))

        # 根據語言選擇 Prompt
        if self.language == "en":
            prompt = f"""
            You are a financial analyst. Answer the question below based ONLY on the provided document excerpts.

            Context:
            {context}

            Question: {translated_question}

            Rules:
            - If the context does not contain relevant information, respond with: "The provided document does not contain relevant information."
            - Keep the answer Concise and professional.
            - Provide an answer based on the document content.
            - Do not repeat the question.
            - Only include the answer itself without additional explanations.
            - **Translate the content into Traditional Chinese used in Taiwan.**
            
            """
        else:
            prompt = f"""
            你是一位財務分析師。請根據提供的文件內容回答以下問題。
            
            規則：
            **翻譯為臺灣使用的繁體中文**
            如果內文不包含相關訊息，請回答：“提供的文檔不包含相關訊息。”
            請提供與文件內容相關的回答，
            避免重複使用問題句，
            只顯示具體回答內容，不要包含解釋性描述。

            內容：
            {context}

            問題：{question}
             
        """


        response = self.llm.invoke(prompt)

        # 回傳 LLM 回應並附上來源頁碼
        return f"{response.content}\n\n📌 來源頁碼: {source_pages_str}"



# 測試程式碼
if __name__ == "__main__":
    pdf_url = "https://s201.q4cdn.com/141608511/files/doc_financials/2025/q3/ed2a395c-5e9b-4411-8b4a-a718d192155a.pdf"
    #pdf_url = "https://www.cathayholdings.com/holdings/-/media/2703c7e79a714f51ae42d8ebd434ff09.pdf?sc_lang=zh-tw" #中文

    rag = FinancialReportRAG(pdf_url)

    question = "2025財年Q3主要營收國家有哪些及各多少美金？"

    # 測試檢索的文本
    relevant_texts = rag.retrieve_relevant_texts(question)
    print("🔍 檢索到的相關文本：")
    for page, text in relevant_texts:
        print(f"📄 頁碼: {page}\n{text}\n")

    # 測試 LLM 產生摘要回答
    summary = rag.generate_answer(question)
    print("\n🔎 LLM 摘要回應：\n", summary)
