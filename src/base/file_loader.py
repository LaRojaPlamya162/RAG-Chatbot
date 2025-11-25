from langchain_community.document_loaders import PyPDFium2Loader
from typing import List
from langchain_core.documents import Document

def remove_ununicode_character(text: str) -> str:
    # Chỉ giữ lại ký tự ASCII (0–127)
    return ''.join([char if ord(char) < 128 else ' ' for char in text])

class PDFLoader:
    def __init__(self, source: str):
        self.source = source.strip()
        self.loader = PyPDFium2Loader(self.source)

    def load(self) -> List[Document]:
        raw_docs = self.loader.load()
        cleaned_docs = []

        for doc in raw_docs:
            cleaned_text = remove_ununicode_character(doc.page_content)
            cleaned_docs.append(
                Document(page_content=cleaned_text, metadata=doc.metadata)
            )
        
        return cleaned_docs


if __name__ == "__main__":
    url = "https://arxiv.org/pdf/1706.03762.pdf"
    loader = PDFLoader(url)
    docs = loader.load()
    print(docs[0].page_content[:500])
