from langchain_community.document_loaders import PyPDFium2Loader
from langchain_community.document_loaders import OnlinePDFLoader
from typing import List
from langchain_core.documents import Document

def remove_ununicode_character(text: str) -> str:
    # Only keep ASCII characters
    return ''.join([char if ord(char) < 128 else ' ' for char in text])

class PDFLoader:
    def __init__(self, source: List[str]):
        self.source = source
        self.loader = None
    def load(self) -> List[Document]:
        print("Loading PDF documents...")
        raw_docs = []
        cleaned_docs = []
        for url in self.source:
            #self.loader = OnlinePDFLoader(url)
            self.loader = PyPDFium2Loader(url)
            raw_docs.extend(self.loader.load())

        for doc in raw_docs:
            cleaned_text = remove_ununicode_character(doc.page_content)
            cleaned_docs.append(
                Document(page_content=cleaned_text, metadata=doc.metadata)
            )
        
        return cleaned_docs

if __name__ == "__main__":
    url = "https://arxiv.org/pdf/1706.03762.pdf"
    loader = PDFLoader([url])
    docs = loader.load()
    print(docs[0].page_content[:500])  # Print first 500 characters of the first document

