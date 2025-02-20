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
from langchain.memory import ConversationBufferWindowMemory

from sklearn.metrics.pairwise import cosine_similarity

# 載入環境變數
load_dotenv()
os.environ["USER_AGENT"] = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
openai_api_key = os.getenv("OPENAI_API_KEY")

class FinancialReportRAG:
    def __init__(self, input_source):
        """
        初始化 RAG 模型
        :param input_source: 可以是 PDF 網址或本地 PDF 文件路徑
        """
        if input_source.startswith("http"):
            self.pdf_path = self._download_pdf(input_source)
        else:
            self.pdf_path = input_source

        self.texts_with_pages = self._load_and_process_pdf()
        self.embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
        self.index, self.texts_with_pages = self._build_faiss_index()
        self.llm = ChatOpenAI(model="gpt-4", temperature=0)

        self.memory = ConversationBufferWindowMemory(
            memory_key="history",
            k=4,  # 只記住最近 5 次對話
            return_messages=True
        )

        self.language = self._detect_language()
        self.retrieval_memory = {}

    def _download_pdf(self, url):
        """從網址下載 PDF 並存儲到本地"""
        pdf_filename = "draft/financial_report.pdf"
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
                    chunk_size=800,
                    chunk_overlap=400,
                    separators=["\n\n", "\n", " ", ""]
                )
                chunks = text_splitter.split_text(text)
                for chunk in chunks:
                    texts_with_pages.append((i + 1, chunk))
        return texts_with_pages

    def _build_faiss_index(self):
        """建立 FAISS 索引"""
        texts = [t[1] for t in self.texts_with_pages]
        if not texts:
            raise ValueError("無法從 PDF 提取文本，請確認 PDF 內容是否可讀取。")

        embedded_texts = self.embeddings.embed_documents(texts)
        if not embedded_texts:
            raise ValueError("嵌入生成失敗，請確認 OpenAI API 金鑰是否有效，或文本是否可用。")

        embedded_texts = np.array(embedded_texts, dtype=np.float32)
        index = faiss.IndexFlatL2(embedded_texts.shape[1])
        index.add(embedded_texts)

        return index, self.texts_with_pages

    def _detect_language(self):
        """判斷語言"""
        sample_text = " ".join([t[1] for t in self.texts_with_pages[:10]])
        chinese_char_count = sum(1 for char in sample_text if '\u4e00' <= char <= '\u9fff')
        return "zh" if chinese_char_count / len(sample_text) > 0.1 else "en"

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

    def retrieve_relevant_texts(self, question, k=5):
        """檢索最相關的文本片段"""
        question_embedding = self._get_question_embedding(question)
        distances, indices = self.index.search(np.array([question_embedding]), k)
        results = [(self.texts_with_pages[i][0], self.texts_with_pages[i][1]) for i in indices[0]]
        return results

    def is_related_to_previous_question(self, question):
        """
        判斷當前問題是否與上一個問題有關，並確保關聯性足夠高
        """
        history = self.memory.load_memory_variables({})["history"]

        if len(history) < 2:
            return False  # 沒有足夠的對話記錄，視為不相關

        last_question = history[-2].content if hasattr(history[-2], "content") else ""
        last_answer = history[-1].content if hasattr(history[-1], "content") else ""

        # **步驟 1：關鍵字檢測（增強版）**
        related_keywords = [
            "增長", "變化", "相比", "趨勢", "影響", "變動", "成長", "下降", "增加", "減少",
            "提升", "降低", "變更", "市場份額", "營收變動", "波動"
        ]
        if any(keyword in question.lower() for keyword in related_keywords):
            return True  # 若包含關鍵詞，視為相關問題

        # **步驟 2：Jaccard 相似度檢測**
        q_words = set(question.lower().split())
        last_q_words = set(last_question.lower().split())
        jaccard_similarity = len(q_words & last_q_words) / max(len(q_words | last_q_words), 1)

        if jaccard_similarity > 0.5:  # 降低閾值，允許更寬鬆的匹配
            return True

        # **步驟 3：語義相似度檢測（餘弦相似度）**
        question_embedding = np.array(self.embeddings.embed_query(question), dtype=np.float32).reshape(1, -1)
        last_question_embedding = np.array(self.embeddings.embed_query(last_question), dtype=np.float32).reshape(1, -1)

        similarity_score = cosine_similarity(question_embedding, last_question_embedding)[0][0]

        return similarity_score > 0.7  # 若語義相似度超過 0.7，則視為相關

    def generate_answer(self, question):
        """
        使用 LLM 生成回答，並根據問題是否與過去對話相關決定是否讀取記憶
        """
        is_related = self.is_related_to_previous_question(question)  # 檢查問題相關性
        history = self.memory.load_memory_variables({})["history"]

        if is_related and len(history) > 1:
            # **擷取上一個問題及回答**
            last_question = history[-2].content if hasattr(history[-2], "content") else ""
            last_answer = history[-1].content if hasattr(history[-1], "content") else ""

            # **將上一個回答的關鍵資訊代入新問題**
            revised_question = f"{question} (上一個問題為：「{last_question}」，其回答為：「{last_answer}」)"
            context = f"這是你的上一個問題及回答，請根據此資訊回答新的問題，並結合新的檢索內容來提供完整的回答：\n\n【上一個問題】{last_question}\n【上一個回答】{last_answer}\n\n"
            print("🔄 問題與前一個問題相關，將讀取上一個問題及回答，並重新檢索文本。")
        else:
            revised_question = question
            context = ""
            print("🆕 問題無關，進行全新檢索。")

        translated_question = revised_question if self.language == "zh" else self._translate_question(revised_question)

        # **重新檢索文本**
        relevant_texts = self.retrieve_relevant_texts(translated_question, k=7)  # 🔺擴大檢索範圍
        if not relevant_texts:
            return "⚠️ 文件未提供相關資訊，請嘗試不同的問題或參考官方資料來源。"

        document_context = "\n\n".join([f"(頁碼: {p}) {t}" for p, t in relevant_texts])
        source_pages = {p for p, _ in relevant_texts}
        source_pages_str = ", ".join(map(str, sorted(source_pages)))

        prompt_template = f"""
        你是一位分析師，請根據提供的文件內容回答問題。
    
        {context}  # 插入過去的問題與回答（如果適用）
    
        【文件內容】
        {document_context}
    
        【問題】
        {revised_question}
    
        規則：
        - **如果新問題與過去問題相關，請參考過去的問題與回答，並根據新的檢索內容補充完整資訊。**
        - 若內文不包含詳細數據，請根據已知資訊進行推理並補充。
        - 若新問題與過去問題無關，請完全依據當前文件內容回答。
        - 直接回答問題，不要重複問題內容。
        - 只顯示具體答案，不要加入解釋性描述。
        - 使用台灣的繁體中文通順語意，
        - 若回答涉及金錢單位，請確認內文為新台幣或美元。
        """

        response = self.llm.invoke(prompt_template)

        # **更新對話記憶**
        self.memory.save_context({"question": question}, {"answer": response.content})

        return f"{response.content}\n\n  (來源頁碼: {source_pages_str})"



# 測試程式碼
if __name__ == "__main__":
    pdf_url = "https://s201.q4cdn.com/141608511/files/doc_financials/2025/q3/ed2a395c-5e9b-4411-8b4a-a718d192155a.pdf"
    rag = FinancialReportRAG(pdf_url)

    question1 = "請問主要營運的單位是哪兩個單位"
    print("\n🔎 LLM 回應：\n", rag.generate_answer(question1))

    question2 = "請問這兩個單位的業務內容？"
    print("\n🔎 LLM 回應：\n", rag.generate_answer(question2))

    question3 = "請問該公司的未來投資計劃？"
    print("\n🔎 LLM 回應：\n", rag.generate_answer(question3))

