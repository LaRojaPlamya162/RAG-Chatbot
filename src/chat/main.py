from langchain_ollama import ChatOllama
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
import json
import os
from src.base.offline_rag import RAGChatbot
from src.base.vector_db import RAGVectorDB
from src.base.file_loader import PDFLoader

class ChatPipeline:
    """
    Offline chatbot using Ollama + saving history to JSON file
    """

    def __init__(self, model_name: str = "qwen2.5:1.5b-instruct", history_file: str = "chat_history.json", vector_store=None):
        self.model_name = model_name
        self.history_file = history_file
        # LLM offline + RAG chatbot
        self.rag_bot = RAGChatbot(vector_store=vector_store)  

        # Load chat history from file
        self.chat_history = self._load_history()

        # Create conversational chain with history support
        self.conversational_chain = self._create_chain()

    def _load_history(self) -> ChatMessageHistory:
        """Read chat history from JSON file"""
        history = ChatMessageHistory()
        if os.path.exists(self.history_file):
            try:
                with open(self.history_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for msg in data:
                        if msg["type"] == "human":
                            history.add_message(HumanMessage(content=msg["content"]))
                        elif msg["type"] == "ai":
                            history.add_message(AIMessage(content=msg["content"]))
            except:
                print(f"Không thể đọc file lịch sử {self.history_file}, bắt đầu mới.")
        return history

    def _save_history(self):
        """Save chat history to JSON file"""
        messages = []
        for msg in self.chat_history.messages:
            if isinstance(msg, HumanMessage):
                messages.append({"type": "human", "content": msg.content})
            elif isinstance(msg, AIMessage):
                messages.append({"type": "ai", "content": msg.content})

        with open(self.history_file, "w", encoding="utf-8") as f:
            json.dump(messages, f, ensure_ascii=False, indent=2)

    def _create_chain(self):
        """Create conversational chain with history support"""
        prompt = ChatPromptTemplate.from_messages([
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}"),
        ])

        chain = prompt | self.rag_bot.llm

        # Use RunnableWithMessageHistory to manage chat history
        return RunnableWithMessageHistory(
            chain,
            lambda session_id: self.chat_history,  # return the same history for all sessions
            input_messages_key="input",
            history_messages_key="history",
        )

    def ask(self, user_input: str) -> str:
        """Send user input to chatbot, get response"""
        if not user_input.strip():
            return "Vui lòng nhập nội dung!"

        response = self.conversational_chain.invoke(
            {"input": user_input},
            config={"configurable": {"session_id": "default"}}  # constant session_id
        )

        # save history after each interaction
        self._save_history()

        return response.content

    def clear_history(self):
        """Clear chat history both in memory and file"""
        if os.path.exists(self.history_file):
            os.remove(self.history_file)
        self.chat_history.clear()
        print("Chat history deleted.")


if __name__ == "__main__":
    url = "https://arxiv.org/pdf/1706.03762.pdf"
    loader = PDFLoader(url)
    docs = loader.load()
    rag_vector_db = RAGVectorDB(docs)
    rag_vector_db.build_vector_store()
    chat = ChatPipeline(model_name="qwen2.5:1.5b-instruct", vector_store=rag_vector_db.vector_store) 

    print("=== Offline Chatbot with Ollama ===")
    print("Type exit for escape, type 'clear' for history deleting\n")

    while True:
        user_msg = input("You: ").strip()

        if user_msg.lower() == "exit":
            print("Bye!")
            break
        if user_msg.lower() == "clear":
            chat.clear_history()
            continue
        if not user_msg:
            continue

        print("Bot:", end=" ")
        response = chat.ask(user_msg)
        print(response)
        print() 