from typing import List
import os
import json
import numpy as np
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.base.file_loader import PDFLoader
from src.sources.content import ContentSources

class RAGVectorDB:
    def __init__(self, documents: List[Document]):
        self.documents = documents
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
        self.vector_store = InMemoryVectorStore(embedding=self.embeddings)

    def chunking(self, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Document]:
        """Chunk documents into smaller pieces from better retrieval."""
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
        """Create vector store from chunked documents."""
        print("Building vector store...")
        chunked_docs = self.chunking()
        self.vector_store.add_documents(chunked_docs)
    def save(self, path: str):
        """Save vectors, texts, metadata safely."""
        os.makedirs(path, exist_ok=True)

        # 1. retrieve all docs
        stored_docs = self.vector_store.similarity_search("", k=1000000)

        # 2. embed text again to get vectors (API-safe)
        texts = [d.page_content for d in stored_docs]
        metadata = [d.metadata for d in stored_docs]
        vectors = self.embeddings.embed_documents(texts)

        # Save vectors
        np.save(os.path.join(path, "vectors.npy"), np.array(vectors))

        # Save texts + metadata
        with open(os.path.join(path, "texts.json"), "w", encoding="utf-8") as f:
            json.dump(texts, f, ensure_ascii=False)

        with open(os.path.join(path, "metadata.json"), "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False)

        print(f"[OK] Vector store saved → {path}")

    @classmethod
    def load(cls, path: str):
        """Load previously saved vector store."""
        texts = json.load(open(os.path.join(path, "texts.json"), "r", encoding="utf-8"))
        metadata = json.load(open(os.path.join(path, "metadata.json"), "r", encoding="utf-8"))

        docs = [Document(page_content=t, metadata=m) for t, m in zip(texts, metadata)]

        obj = cls(docs)
        obj.vector_store = InMemoryVectorStore(embedding=obj.embeddings)

        # Add back to new vector store
        obj.vector_store.add_documents(docs)

        print(f"[OK] Vector store loaded ← {path}")
        return obj

if __name__ == "__main__":
    #url = "https://arxiv.org/pdf/1706.03762.pdf"
    loader = PDFLoader(ContentSources().get_pdf_urls())
    docs = loader.load()
    rag_vector_db = RAGVectorDB(docs)
    rag_vector_db.build_vector_store()
    rag_vector_db.save("./src/sources/vector_store")

  