# Financial Report RAG

## Overview
This project implements a **Retrieval-Augmented Generation (RAG)** system designed to analyze **financial reports** using **FAISS** for semantic retrieval and **GPT-4o** for generating responses. The system extracts text from **PDF reports**, indexes them for fast retrieval, and enhances question-answering by leveraging the retrieved context.

## Features
- **Download and Process PDF Reports**: Supports both **online URLs** and **local files**.
- **Text Chunking & FAISS Indexing**: Uses **overlapping windows** for text segmentation and FAISS for fast semantic search.
- **Multi-round Dialogue Memory**: Retains recent conversation history to improve question-answer coherence.
- **Language Detection**: Determines whether the document is in **Chinese** or **English**.
- **Similarity-Based Context Awareness**: Uses **keyword matching, Jaccard similarity, and cosine similarity** to check if a question is related to previous ones.
- **GPT-4o Integration**: Generates responses based on retrieved information.

## Installation
### Prerequisites
Ensure you have the following installed:
- Python 3.8+
- Pip

### Dependencies
Run the following command to install required libraries:
```bash
pip install numpy faiss-cpu requests pypdf langchain langchain-openai langchain-community python-dotenv scikit-learn
```

## Usage
### 1. Load Environment Variables
Create a `.env` file in the project directory with your OpenAI API key:
```
OPENAI_API_KEY=your_openai_api_key
```

### 2. Run the System
Example usage in Python:
```python
from financial_report_rag import FinancialReportRAG

# Initialize system with a PDF file or URL
rag = FinancialReportRAG("path/to/financial_report.pdf")

# Ask a question
response = rag.generate_answer("What is the revenue growth trend?")
print(response)
```

## How It Works
### Step-by-Step Process
1. **Load PDF**: Extracts text from the report.
2. **Chunk Text**: Splits text into overlapping segments.
3. **Create FAISS Index**: Embeds text and stores in FAISS.
4. **Retrieve Relevant Texts**: Finds the most relevant chunks based on the user’s query.
5. **Determine Context Relevance**: Checks if the question relates to previous ones.
6. **Generate Response**: Uses GPT-4o to formulate an answer based on retrieved content.

## Configuration
### Modify Memory Window
To adjust conversation memory length:
```python
self.memory = ConversationBufferWindowMemory(memory_key="history", k=2, return_messages=True)
```
Increase `k` for longer memory retention.

## License
This project is licensed under the **MIT License**.

## Contact
For questions or contributions, reach out via [GitHub Issues](https://github.com/yourrepo).

