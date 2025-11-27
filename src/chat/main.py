from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage
import json
import os
from src.base.main import RAGChatBot


class ChatPipeline:
    """
    Offline RAG chatbot using RAGChatBot() + saving history to JSON file
    """

    def __init__(self, history_file="chat_history.json"):
        self.history_file = history_file

        # Load the RAG-based chatbot
        self.bot = RAGChatBot()

        # Load saved history
        self.chat_history = self._load_history()

    # ---------------------- HISTORY I/O ----------------------

    def _load_history(self) -> ChatMessageHistory:
        """Load chat history from JSON file into ChatMessageHistory object."""
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
                print(f"[WARN] Cannot read {self.history_file}. Starting fresh.")

        return history

    def _save_history(self):
        """Store history from ChatMessageHistory back to JSON file."""
        data = []

        for msg in self.chat_history.messages:
            if isinstance(msg, HumanMessage):
                data.append({"type": "human", "content": msg.content})
            elif isinstance(msg, AIMessage):
                data.append({"type": "ai", "content": msg.content})

        with open(self.history_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    # ---------------------- MAIN CHAT METHOD ----------------------

    def ask(self, question: str):
        """
        User → history → RAGChatBot → answer → history
        """

        if not question.strip():
            return "Please enter a valid question!"

        # Save user's message
        self.chat_history.add_message(HumanMessage(content=question))

        # Call your RAGChatBot (RAG retrieval + LLM answer)
        answer, sources = self.bot.ask(question)

        # Save model answer to history
        self.chat_history.add_message(AIMessage(content=answer))

        # Persist file
        self._save_history()

        # Return full result
        return {
            "answer": answer,
            "sources": sources,
        }

    # ---------------------- CLEAN HISTORY ----------------------

    def clear_history(self):
        """Clear both file & in-memory chat history."""
        if os.path.exists(self.history_file):
            os.remove(self.history_file)
        self.chat_history.clear()
        print("Chat history cleared.")
if __name__ == "__main__":
    chat = ChatPipeline()
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
        print(response["answer"])
        print('/n')
        print(response["sources"])