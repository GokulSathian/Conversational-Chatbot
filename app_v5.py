import streamlit as st
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
import os
from dotenv import load_dotenv
import tempfile
import pymongo
from pymongo import MongoClient
import bcrypt
from datetime import datetime

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")

# MongoDB setup
client = MongoClient(MONGODB_URI)
db = client['chatbot_db']
users_collection = db['users']
chats_collection = db['chats']  # New collection for storing chats

# Set Streamlit page configuration
st.set_page_config(page_title='🧠MemoryBot🤖', layout='wide')

# Initialize session states
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "username" not in st.session_state:
    st.session_state.username = None
if "chats" not in st.session_state:
    st.session_state.chats = {}
if "current_chat" not in st.session_state:
    st.session_state.current_chat = None
if "input" not in st.session_state:
    st.session_state.input = ""
if "stop_signal" not in st.session_state:
    st.session_state.stop_signal = False
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

# Function to process uploaded PDF
def process_pdf(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name

    loader = PyPDFLoader(tmp_file_path)
    documents = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)
    
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    vector_store = FAISS.from_documents(texts, embeddings)
    
    os.unlink(tmp_file_path)  # Clean up the temporary file
    return vector_store

# Function to load user's chats from MongoDB
def load_user_chats(username):
    chats = {}
    user_chats = chats_collection.find({"username": username})
    
    for chat in user_chats:
        chat_id = chat['chat_id']
        # Create a new memory instance with correct memory_key
        memory = ConversationBufferMemory(
            memory_key="history",  # Changed from "chat_history" to "history"
            return_messages=True
        )
        
        # Reconstruct the conversation history in memory
        messages = chat['messages']
        for i in range(0, len(messages), 2):  # Process pairs of messages
            if i + 1 < len(messages):  # Make sure we have both user and assistant messages
                user_message = messages[i]['content']
                assistant_message = messages[i + 1]['content']
                # Add the message pair to memory
                memory.save_context(
                    {"input": user_message},
                    {"output": assistant_message}  # Changed from "answer" to "output"
                )
        
        chats[chat_id] = {
            "title": chat['title'],
            "messages": chat['messages'],
            "memory": memory,
            "vector_store": None,
            "chat_history": chat.get('chat_history', [])
        }
    return chats

# Function to save chat to MongoDB
def save_chat(username, chat_id, chat_data):
    chat_doc = {
        "username": username,
        "chat_id": chat_id,
        "title": chat_data["title"],
        "messages": chat_data["messages"],
        "chat_history": chat_data["chat_history"],
        "updated_at": datetime.now()
    }
    
    # Update or insert the chat document
    chats_collection.update_one(
        {"username": username, "chat_id": chat_id},
        {"$set": chat_doc},
        upsert=True
    )

# Function to start a new chat
def new_chat():
    chat_id = datetime.now().strftime("%Y%m%d%H%M%S")
    st.session_state.chats[chat_id] = {
        "title": f"Chat {len(st.session_state.chats) + 1}",
        "messages": [],
        "memory": ConversationBufferMemory(
            memory_key="history",  # Changed from "chat_history" to "history"
            return_messages=True
        ),
        "vector_store": None,
        "chat_history": []
    }
    st.session_state.current_chat = chat_id
    # Save the new chat
    save_chat(st.session_state.username, chat_id, st.session_state.chats[chat_id])

# Function to rename the current chat
def rename_chat(new_title):
    if st.session_state.current_chat is not None:
        st.session_state.chats[st.session_state.current_chat]["title"] = new_title

def create_test_user():
    # Test user credentials
    test_username = "testuser"
    test_password = "password123"
    
    # Check if user already exists
    if not users_collection.find_one({"username": test_username}):
        hashed_password = bcrypt.hashpw(test_password.encode('utf-8'), bcrypt.gensalt())
        users_collection.insert_one({
            "username": test_username,
            "password": hashed_password,
            "created_at": datetime.now()
        })
        return test_username, test_password
    return None

# Modify verify_user to load chats after successful login
def verify_user(username, password):
    user = users_collection.find_one({"username": username})
    if user and bcrypt.checkpw(password.encode('utf-8'), user['password']):
        # Load user's chats into session state
        st.session_state.chats = load_user_chats(username)
        return True
    return False

# Login form
def show_login_form():
    st.title("🔐 Login to MemoryBot")
    
    # Create columns for centered login form
    col1, col2, col3 = st.columns([1,2,1])
    
    with col2:
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Login"):
                if verify_user(username, password):
                    st.session_state.authenticated = True
                    st.session_state.username = username
                    st.rerun()
                else:
                    st.error("Invalid username or password")
        with col2:
            if st.button("Create Test User"):
                result = create_test_user()
                if result:
                    username, password = result
                    st.success(f"""
                    Test user created successfully!
                    Username: {username}
                    Password: {password}
                    """)
                else:
                    st.info("Test user already exists")

# Main app logic
def main():
    if not st.session_state.authenticated:
        show_login_form()
        return

    # Sidebar for chat selection and new chat creation
    with st.sidebar:
        st.title("Chat History")
        # Add sidebar expander
        with st.expander("Chat List", expanded=True):
            for chat_id, chat in st.session_state.chats.items():
                col1, col2 = st.columns([4, 1])
                with col1:
                    if st.button(chat["title"], key=f"chat_{chat_id}"):
                        st.session_state.current_chat = chat_id
                with col2:
                    if st.button("✏️", key=f"edit_{chat_id}", help="Edit chat name"):
                        st.session_state[f"edit_mode_{chat_id}"] = True

                # Show text input when edit mode is active
                if st.session_state.get(f"edit_mode_{chat_id}", False):
                    new_title = st.text_input(
                        "New title:", 
                        value=chat["title"],
                        key=f"title_{chat_id}",
                        label_visibility="visible"
                    )
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("Save", key=f"save_{chat_id}", use_container_width=True):
                            st.session_state.chats[chat_id]["title"] = new_title
                            # Save the updated chat
                            save_chat(st.session_state.username, chat_id, st.session_state.chats[chat_id])
                            st.session_state[f"edit_mode_{chat_id}"] = False
                            st.rerun()
                    with col2:
                        if st.button("Cancel", key=f"cancel_{chat_id}", use_container_width=True):
                            st.session_state[f"edit_mode_{chat_id}"] = False
                            st.rerun()
            
            st.button("New Chat", on_click=new_chat)

    # Main chat interface
    if st.session_state.current_chat is not None:
        chat = st.session_state.chats[st.session_state.current_chat]
        st.title(chat["title"])
        
        # Add file upload button in the chat interface
        uploaded_file = st.file_uploader("Upload a PDF document", type=['pdf'], key=f"uploader_{st.session_state.current_chat}")
        if uploaded_file is not None:
            if st.button("Process PDF"):
                with st.spinner("Processing PDF..."):
                    chat["vector_store"] = process_pdf(uploaded_file)
                    st.success("PDF processed successfully! You can now ask questions about it.")
                    st.rerun()

        # Display conversation history with aligned messages
        for message in chat["messages"]:
            if message["role"] == "user":
                with st.chat_message(message["role"], avatar="🧑"):
                    col1, col2, col3 = st.columns([3, 1, 1])
                    with col3:
                        st.markdown(message["content"])
            else:
                with st.chat_message(message["role"], avatar="🤖"):
                    st.markdown(message["content"])

        # Input box for user message
        if prompt := st.chat_input("Type your message here..."):
            with st.chat_message("user", avatar="🧑"):
                col1, col2, col3 = st.columns([3, 1, 1])
                with col3:
                    st.markdown(prompt)
            chat["messages"].append({"role": "user", "content": prompt})

            # Initialize LLM
            llm = ChatOpenAI(model="gpt-4o-mini", openai_api_key=OPENAI_API_KEY)
            
            # Choose between regular conversation and PDF-based QA
            if chat["vector_store"] is not None:
                qa_chain = ConversationalRetrievalChain.from_llm(
                    llm=llm,
                    retriever=chat["vector_store"].as_retriever(),
                    memory=chat["memory"],
                    return_source_documents=True,
                    chain_type="stuff"
                )
                
                # Get response
                if not st.session_state.stop_signal:
                    result = qa_chain({"question": prompt})
                    response = result["answer"]
                    chat["chat_history"].append((prompt, response))
            else:
                conversation = ConversationChain(
                    llm=llm,
                    memory=chat["memory"],
                    verbose=True
                )

            # Get response
            if not st.session_state.stop_signal:
                response = conversation.run(input=prompt)

            # Display assistant response with stop button
            with st.chat_message("assistant", avatar="🤖"):
                response_placeholder = st.empty()
                stop_placeholder = st.empty()
                
                stop_button = stop_placeholder.button("Stop")
                if stop_button:
                    st.session_state.stop_signal = True
                
                if not st.session_state.stop_signal:
                    response_placeholder.markdown(response)
                    chat["messages"].append({"role": "assistant", "content": response})
                    stop_placeholder.empty()
                else:
                    response_placeholder.markdown("Response generation stopped.")
                    st.session_state.stop_signal = False
                    stop_placeholder.empty()

            # Save the updated chat after new messages
            save_chat(st.session_state.username, st.session_state.current_chat, chat)
    else:
        st.title("Welcome to 🧠MemoryBot🤖")
        st.write("Select a chat from the sidebar or start a new one.")

if __name__ == "__main__":
    main()

