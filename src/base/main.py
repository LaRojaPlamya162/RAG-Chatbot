from src.base.file_loader import PDFLoader
from src.base.vector_db import RAGVectorDB
from src.base.offline_rag import LLMRetriver
from src.sources.content import ContentSources
class RAGChatBot:
  def __init__(self):
      self.content = [ContentSources().file_links[i]["url"] for i in range(len(ContentSources().file_links))]
      print(self.content)
      self.loader = PDFLoader(self.content)
      self.docs = self.loader.load()
      self.rag_vector_db = RAGVectorDB(self.docs)
      self.rag_vector_db.build_vector_store()
      self.bot = LLMRetriver(self.rag_vector_db.vector_store)
  def ask(self, question: str):
      result = self.bot.ask(question)
      return result["answer"], result["sources"]
  
if __name__ == "__main__":
    rag_chat_bot = RAGChatBot()
    question = "What is the main contribution of the paper?"
    print("Question:", question)
    answer, sources = rag_chat_bot.ask(question)
    print("Answer:", answer)
    print("Sources:", sources)