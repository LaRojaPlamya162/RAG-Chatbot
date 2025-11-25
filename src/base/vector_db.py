from typing import List
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from file_loader import PDFLoader

class RAGVectorDB:
    def __init__(self, documents: List[Document]):
        self.documents = documents
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
        self.vector_store = InMemoryVectorStore(embedding=self.embeddings)

    def chunking(self, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Document]:
        """Chia nhỏ tài liệu thành các đoạn nhỏ hơn để dễ dàng xử lý."""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            add_start_index=True
        )

        chunked_docs = []
        for doc in self.documents:
            chunks = text_splitter.split_text(doc.page_content)
            for chunk in chunks:
                chunked_docs.append(Document(page_content=chunk, metadata=doc.metadata))

        return chunked_docs

    def build_vector_store(self):
        """Xây dựng vector store từ các đoạn tài liệu đã chia nhỏ."""
        chunked_docs = self.chunking()
        self.vector_store.add_documents(chunked_docs)
        print(f"Đã thêm {len(chunked_docs)} đoạn tài liệu vào vector store.")
        print(chunked_docs[0].page_content[:500])  # In ra 500 ký tự đầu tiên của đoạn đầu tiên


if __name__ == "__main__":
    url = "https://arxiv.org/pdf/1706.03762.pdf"
    loader = PDFLoader(url)
    docs = loader.load()
    rag_vector_db = RAGVectorDB(docs)
    rag_vector_db.build_vector_store()
  