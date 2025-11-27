from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from src.base.file_loader import PDFLoader
from src.base.vector_db import RAGVectorDB
class LLMRetriver:
    def __init__(self, vector_store_dir: str = "./src/sources/vector_store"):
        """
        vector_store: InMemoryVectorStore đã build xong
        """
        self.vector_store = RAGVectorDB.load(vector_store_dir).vector_store
        self.retriever = self.vector_store.as_retriever(search_kwargs={"k": 3})

        # LLM offline
        self.llm = ChatOllama(
            model="qwen2.5:1.5b-instruct", 
            base_url="http://localhost:11434",
            temperature=0.1
        )

        # Prompt template
        self.prompt = ChatPromptTemplate.from_template("""
You are an AI assistant that helps users by providing information from the given context.

{context}

Question:  {question}

Please provide a detailed and accurate answer based on the above context. If the context does not contain relevant information, respond with "I don't know."
""")

    def build_prompt(self, question, docs):
        """
        Convert retrieved docs + question to prompt for LLM
        """
        context = "\n\n".join([f"[Source]\n{d.page_content}" for d in docs])
        return self.prompt.format(context=context, question=question)

    def ask(self, question: str):
        """
        Receive question from user, return answer and sources
        """
        print("Retrieving relevant documents...")
        # Retrieve top-k docs
        docs = self.retriever.invoke(question)

        # Build prompt
        prompt_to_llm = self.build_prompt(question, docs)

        response = self.llm.invoke(prompt_to_llm)

        return {
            "answer": response.content,
            "sources": docs
        }
if __name__ == "__main__":
    #url = "https://arxiv.org/pdf/1706.03762.pdf"
    #loader = PDFLoader([url])
    #docs = loader.load()

    #rag_vector_db = RAGVectorDB(docs).load("./src/sources/vector_store")
    #rag_vector_db = RAGVectorDB(docs)
    #rag_vector_db.build_vector_store()
    
    #bot = LLMRetriver(rag_vector_db.vector_store)
    bot = LLMRetriver()
    question = "What is the main idea of the Transformer model?"
    print(question)
    result = bot.ask(question)
    print("Answer:")
    print(result["answer"])

    print("\nResources:")
    for i, d in enumerate(result["sources"]):
        print(f"[{i+1}] {d.metadata}")

