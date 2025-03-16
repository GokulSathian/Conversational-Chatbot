# 🧠 Conversational-Chatbot MemoryBot – Your AI-Powered Knowledge Companion  

## 📌 Project Overview  

MemoryBot is an AI-driven chatbot designed to **store, retrieve, and analyze conversations and documents**. It integrates **OpenAI’s GPT models**, **MongoDB**, **Streamlit**and **FAISS vector storage** to provide a seamless memory-enhanced chat experience.  

🚀 **Key Capabilities:**  
✔️ **Conversational Memory** – Remembers previous chats for continuity  
✔️ **PDF Processing** – Extracts and retrieves knowledge from uploaded PDFs  
✔️ **Secure Authentication** – User login and session management  
✔️ **Admin Dashboard** – Manage users and monitor prompt usage  
✔️ **MongoDB & FAISS** – Efficient data retrieval and storage  

---

## ✨ Features  

✅ **Natural Language Chat** – Uses OpenAI’s ChatGPT models for interactive conversations  
✅ **Memory-Based Retrieval** – Enhances response quality with persistent chat memory  
✅ **PDF Knowledge Extraction** – Upload PDFs and ask questions based on the content  
✅ **Admin Controls** – Manage users, reset prompt limits, and monitor usage statistics  
✅ **Secure Login System** – User authentication with encrypted passwords  
✅ **Scalable Architecture** – MongoDB for structured data storage and FAISS for vectorized retrieval  

---

## 📂 Project Structure  

| File Name          | Description |
|--------------------|-------------|
| `app_v10.py`      | Main Streamlit app integrating chat, memory storage, and admin panel. |
| `.env`            | Environment variables for API keys and database credentials. |
| `vector_stores/`  | Stores FAISS vector embeddings for document retrieval. |
| `database/`       | MongoDB collections for user authentication and chat logs. |

---

## ⚙️ Installation  

### **1️⃣ Clone the Repository**  
```bash
git clone https://github.com/your-username/MemoryBot.git
cd MemoryBot

### **2️⃣ Install Dependencies**
```bash
pip install -r requirements.txt

### **3️⃣ Set Up Environment Variables**
OPENAI_API_KEY=your_openai_api_key
MONGODB_URI=mongodb://localhost:27017/
ADMIN_USERNAME=
ADMIN_PASSWORD=


### **4️⃣ Start MongoDB**
mongod --dbpath /path-to-your-data-directory

## 🛠 Code Workflow
1️⃣ User Authentication – Verifies login credentials and session state
2️⃣ Chat Memory Initialization – Loads conversation history from MongoDB
3️⃣ PDF Processing – Extracts text and converts it into FAISS embeddings
4️⃣ Conversational Logic – Uses LangChain and OpenAI’s GPT models for responses
5️⃣ Admin Controls – Monitors user activities and updates prompt limits



