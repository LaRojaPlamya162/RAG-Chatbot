from src.base.file_loader import PDFLoader
from src.base.vector_db import RAGVectorDB
from src.base.offline_rag import LLMRetriver
from src.sources.content import ContentSources
class RAGChatBot:
  def __init__(self):
      self.bot = LLMRetriver()
  def ask(self, question: str):
      result = self.bot.ask(question)
      return result["answer"], result["sources"]
  
if __name__ == "__main__":
    #loader = PDFLoader(ContentSources().get_pdf_urls())
    #docs = loader.load()
    #rag_vector_db = RAGVectorDB(docs)
    #rag_chat_bot = RAGChatBot(rag_vector_db.vector_store)
    
    rag_chat_bot = RAGChatBot()
    question = "What is the main idea of the Transformer model?"
    print("Question:", question)
    answer, sources = rag_chat_bot.ask(question)
    print("Answer:", answer)
    print("Sources:", sources)