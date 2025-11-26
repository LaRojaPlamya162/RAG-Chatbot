from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from file_loader import PDFLoader
from vector_db import RAGVectorDB
class RAGChatbot:
    def __init__(self, vector_store):
        """
        vector_store: InMemoryVectorStore đã build xong
        """
        self.vector_store = vector_store
        self.retriever = vector_store.as_retriever(search_kwargs={"k": 3})

        # LLM offline
        self.llm = ChatOllama(
            model="qwen2.5:1.5b-instruct", 
            base_url="http://localhost:11434",
            temperature=0.1
        )

        # Prompt template
        self.prompt = ChatPromptTemplate.from_template("""
Bạn là một trợ lý AI. Dưới đây là ngữ cảnh lấy từ tài liệu:

{context}

Câu hỏi: {question}

Hãy trả lời thật chính xác dựa trên ngữ cảnh phía trên. 
Nếu không tìm thấy câu trả lời trong tài liệu, hãy trả lời: "Không có thông tin trong tài liệu."
""")

    def build_prompt(self, question, docs):
        """
        Ghép docs thành prompt input
        """
        context = "\n\n".join([f"[Source]\n{d.page_content}" for d in docs])
        return self.prompt.format(context=context, question=question)

    def ask(self, question: str):
        """
        Nhận câu hỏi → retrieve docs → gọi LLM → trả lời
        """
        # Retrieve top-k docs
        docs = self.retriever.invoke(question)

        # Build prompt
        prompt_to_llm = self.build_prompt(question, docs)

        # Query LLM offline
        response = self.llm.invoke(prompt_to_llm)

        return {
            "answer": response.content,
            "sources": docs
        }
if __name__ == "__main__":
    # 1. Load PDF
    url = "https://arxiv.org/pdf/1706.03762.pdf"
    loader = PDFLoader(url)
    docs = loader.load()

    # 2. Tạo vectorDB
    rag_vector_db = RAGVectorDB(docs)
    rag_vector_db.build_vector_store()

    # 3. Tạo chatbot
    bot = RAGChatbot(rag_vector_db.vector_store)

    # 4. Hỏi thử
    question = "What is the main idea of the Transformer model?"
    print(question)
    result = bot.ask(question)
    print("AI trả lời:")
    print(result["answer"])

    print("\nNguồn tham chiếu:")
    for i, d in enumerate(result["sources"]):
        print(f"[{i+1}] {d.metadata}")

