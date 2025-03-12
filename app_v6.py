import streamlit as st
import uuid
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
ADMIN_USERNAME = os.getenv("ADMIN_USERNAME", "admin@memorybot.com")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "Admin@123#")
DEFAULT_MAX_PROMPTS = int(os.getenv("DEFAULT_MAX_PROMPTS", "50"))  # Default prompt limit for new users

# MongoDB setup
client = MongoClient(MONGODB_URI)
db = client['chatbot_db']
users_collection = db['users']
chats_collection = db['chats']  # New collection for storing chats
user_settings_collection = db['user_settings']  # New collection for user settings

# Set Streamlit page configuration
st.set_page_config(page_title='MemoryBot🤖', layout='wide')

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
if "is_admin" not in st.session_state:
    st.session_state.is_admin = False

# Function to process uploaded PDF
def process_pdf(uploaded_files):
    all_texts = []
    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name

        loader = PyPDFLoader(tmp_file_path)
        documents = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        texts = text_splitter.split_documents(documents)
        all_texts.extend(texts)
        
        os.unlink(tmp_file_path)  # Clean up the temporary file
    
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    vector_store = FAISS.from_documents(all_texts, embeddings)
    return vector_store

# Function to load user's chats from MongoDB
def load_user_chats(username):
    chats = {}
    user_chats = chats_collection.find({"username": username})
    
    for chat in user_chats:
        chat_id = chat['chat_id']
        # Create a new memory instance with correct memory_key
        memory = ConversationBufferMemory(
            memory_key="history",  # For regular conversations
            return_messages=True
        )
        
        # Create separate memory for PDF QA with explicit output_key
        qa_memory = ConversationBufferMemory(
            memory_key="chat_history",  # For PDF QA
            return_messages=True,
            output_key="answer"  # Explicitly set output_key
        )
        
        # Reconstruct the conversation history in memory
        messages = chat['messages']
        for i in range(0, len(messages), 2):
            if i + 1 < len(messages):
                user_message = messages[i]['content']
                assistant_message = messages[i + 1]['content']
                # Add to both memories
                memory.save_context(
                    {"input": user_message},
                    {"output": assistant_message}
                )
                qa_memory.save_context(
                    {"input": user_message},
                    {"answer": assistant_message} 
                )
        
        # Load vector store if it exists with safe deserialization
        vector_store = None
        if chat.get('has_vector_store', False):
            vector_store_path = f"vector_stores/{username}/{chat_id}"
            if os.path.exists(vector_store_path):
                embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
                vector_store = FAISS.load_local(
                    vector_store_path, 
                    embeddings,
                    allow_dangerous_deserialization=True  # Add this parameter
                )
        
        chats[chat_id] = {
            "title": chat['title'],
            "messages": chat['messages'],
            "memory": memory,
            "qa_memory": qa_memory,
            "vector_store": vector_store,
            "chat_history": chat.get('chat_history', [])
        }
    return chats

# Function to save chat to MongoDB
def save_chat(username, chat_id, chat_data):
    # Serialize vector store if it exists
    vector_store_exists = chat_data["vector_store"] is not None
    
    chat_doc = {
        "username": username,
        "chat_id": chat_id,
        "title": chat_data["title"],
        "messages": chat_data["messages"],
        "chat_history": chat_data["chat_history"],
        "has_vector_store": vector_store_exists,  # Flag to indicate vector store existence
        "updated_at": datetime.now()
    }
    
    # Save vector store separately if it exists
    if vector_store_exists:
        vector_store_path = f"vector_stores/{username}/{chat_id}"
        os.makedirs(os.path.dirname(vector_store_path), exist_ok=True)
        chat_data["vector_store"].save_local(vector_store_path)
    
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
            memory_key="history",
            return_messages=True
        ),
        "qa_memory": ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"  # Explicitly set output_key
        ),
        "vector_store": None,
        "chat_history": []
    }
    st.session_state.current_chat = chat_id
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

# Update register_user function to include default prompt limit
def register_user(username, password):
    if users_collection.find_one({"username": username}):
        return False, "Username already exists"
    
    hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
    users_collection.insert_one({
        "username": username,
        "password": hashed_password,
        "created_at": datetime.now()
    })
    
    # Initialize user settings with default prompt limit
    user_settings_collection.insert_one({
        "username": username,
        "max_prompts": DEFAULT_MAX_PROMPTS,
        "prompts_used": 0,
        "is_restricted": False
    })
    
    return True, "Registration successful"

# Update the prompt count tracking function
def update_prompt_count(username):
    try:
        result = user_settings_collection.find_one_and_update(
            {"username": username},
            {"$inc": {"prompts_used": 1}},
            return_document=True
        )
        return True, result
    except Exception as e:
        return False, str(e)

# Add function to check if user can send prompts
def can_send_prompt(username):
    settings = user_settings_collection.find_one({"username": username})
    if settings:
        if settings["prompts_used"] >= settings["max_prompts"]:
            return False, f"You have reached your maximum prompt limit ({settings['max_prompts']} prompts). Please contact admin."
    return True, ""

# Update verify_admin function to use environment variables
def verify_admin(username, password):
    return username == ADMIN_USERNAME and password == ADMIN_PASSWORD

# Add function to get user settings
def get_user_settings(username):
    settings = user_settings_collection.find_one({"username": username})
    if not settings:
        # Create default settings if none exist
        settings = {
            "username": username,
            "max_prompts": DEFAULT_MAX_PROMPTS,
            "prompts_used": 0
        }
        user_settings_collection.insert_one(settings)
    return settings

# Update show_admin_panel function to fix reset count and display
def show_admin_panel():
    st.title("👑 Admin Dashboard")
    
    # Add logout button for admin
    col1, col2, col3 = st.columns([6,2,2])
    with col3:
        if st.button("Logout", type="primary"):
            st.session_state.authenticated = False
            st.session_state.username = None
            st.session_state.is_admin = False
            st.rerun()
    
    # Display all registered users with additional controls
    st.header("Registered Users")
    
    try:
        # Get all users with their settings
        users = list(users_collection.find({}, {"_id": 0, "password": 0}))
        settings = list(user_settings_collection.find({}, {"_id": 0}))
        
        # Combine user data with settings
        user_data = []
        for user in users:
            user_setting = next((s for s in settings if s["username"] == user["username"]), None)
            # Count total prompts used from chat history
            total_prompts = 0
            user_chats = chats_collection.find({"username": user["username"]})
            for chat in user_chats:
                total_prompts += len([m for m in chat.get("messages", []) if m["role"] == "user"])
            
            if user_setting:
                user_data.append({
                    "username": user["username"],
                    "created_at": user["created_at"],
                    "prompts_used": total_prompts,  # Use actual count from chat history
                    "max_prompts": user_setting.get("max_prompts", DEFAULT_MAX_PROMPTS)
                })
                # Update the prompts_used in user_settings to match actual usage
                user_settings_collection.update_one(
                    {"username": user["username"]},
                    {"$set": {"prompts_used": total_prompts}}
                )
            else:
                default_settings = {
                    "username": user["username"],
                    "max_prompts": DEFAULT_MAX_PROMPTS,
                    "prompts_used": total_prompts
                }
                user_settings_collection.insert_one(default_settings)
                user_data.append({
                    "username": user["username"],
                    "created_at": user["created_at"],
                    "prompts_used": total_prompts,
                    "max_prompts": DEFAULT_MAX_PROMPTS
                })
        
        if user_data:
            # Convert to DataFrame
            import pandas as pd
            df = pd.DataFrame(user_data)
            df['created_at'] = pd.to_datetime(df['created_at'])
            
            # Display statistics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Users", len(df))
            with col2:
                today_users = len(df[df['created_at'].dt.date == pd.Timestamp.now().date()])
                st.metric("New Users Today", today_users)
            with col3:
                total_prompts = df['prompts_used'].sum()
                st.metric("Total Prompts Used", total_prompts)
            
            # Display basic user table
            st.subheader("User Overview")
            st.dataframe(df)
            
            # User management section
            st.subheader("User Management")
            for _, row in df.iterrows():
                with st.expander(f"User: {row['username']}", expanded=False):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write(f"Prompts Used: {row['prompts_used']}/{row['max_prompts']}")
                        new_limit = st.number_input(
                            "Set Prompt Limit",
                            min_value=1,
                            value=int(row['max_prompts']),
                            key=f"limit_{row['username']}"
                        )
                        if st.button("Update Limit", key=f"update_{row['username']}"):
                            result = user_settings_collection.update_one(
                                {"username": row['username']},
                                {"$set": {"max_prompts": new_limit}},
                                upsert=True
                            )
                            if result.modified_count > 0:
                                st.success("Limit updated successfully!")
                                st.rerun()
                            else:
                                st.error("Failed to update limit")
                    
                    with col2:
                        if st.button("Reset Prompt Count", key=f"reset_{row['username']}"):
                            result = user_settings_collection.update_one(
                                {"username": row['username']},
                                {"$set": {"prompts_used": 0}},
                                upsert=True
                            )
                            if result.modified_count > 0:
                                st.success("Prompt count reset successfully!")
                                st.rerun()
                            else:
                                st.error("Failed to reset prompt count")
        else:
            st.info("No registered users found.")
            
    except Exception as e:
        st.error(f"Error loading admin panel: {str(e)}")
        st.write("Please check your MongoDB connection and collections.")

# Modify the login form to include admin login
def show_login_form():
    st.title("🔐 Login to MemoryBot")
    
    if st.session_state.authenticated:
        if st.session_state.is_admin:
            show_admin_panel()
        else:
            # Show regular user logout button
            col1, col2, col3 = st.columns([1,2,1])
            with col2:
                st.write(f"Logged in as: {st.session_state.username}")
                if st.button("Logout"):
                    st.session_state.authenticated = False
                    st.session_state.username = None
                    st.session_state.is_admin = False
                    st.session_state.chats = {}
                    st.session_state.current_chat = None
                    st.rerun()
        return

    # Create tabs for login, registration, and admin
    tab1, tab2, tab3, tab4 = st.tabs(["Login", "Register", "Create Test User", "Admin Login"])
    
    # Login Tab
    with tab1:
        col1, col2, col3 = st.columns([1,2,1])
        with col2:
            username = st.text_input("Username", key="login_username")
            password = st.text_input("Password", type="password", key="login_password")
            
            if st.button("Login", use_container_width=True):
                if verify_user(username, password):
                    st.session_state.authenticated = True
                    st.session_state.username = username
                    st.rerun()
                else:
                    st.error("Invalid username or password")
    
    # Registration Tab
    with tab2:
        col1, col2, col3 = st.columns([1,2,1])
        with col2:
            new_username = st.text_input("Choose Username", key="reg_username")
            new_password = st.text_input("Choose Password", type="password", key="reg_password")
            confirm_password = st.text_input("Confirm Password", type="password", key="confirm_password")
            
            if st.button("Register", use_container_width=True):
                if not new_username or not new_password:
                    st.error("Please fill in all fields")
                elif new_password != confirm_password:
                    st.error("Passwords do not match")
                else:
                    success, message = register_user(new_username, new_password)
                    if success:
                        st.success(message)
                        st.info("Please go to the Login tab to sign in")
                    else:
                        st.error(message)
    
    # Create Test User Tab
    with tab3:
        col1, col2, col3 = st.columns([1,2,1])
        with col2:
            if st.button("Create Test User", use_container_width=True):
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

    # Admin Login Tab
    with tab4:
        col1, col2, col3 = st.columns([1,2,1])
        with col2:
            admin_username = st.text_input("Admin Username", key="admin_username")
            admin_password = st.text_input("Admin Password", type="password", key="admin_password")
            
            if st.button("Admin Login", use_container_width=False):
                if verify_admin(admin_username, admin_password):
                    st.session_state.authenticated = True
                    st.session_state.username = admin_username
                    st.session_state.is_admin = True
                    st.rerun()
                else:
                    st.error("Invalid admin credentials")

# Function to delete chat from MongoDB and file system
def delete_chat(username, chat_id):
    # Delete from MongoDB
    chats_collection.delete_one({"username": username, "chat_id": chat_id})
    
    # Delete vector store if it exists
    vector_store_path = f"vector_stores/{username}/{chat_id}"
    if os.path.exists(vector_store_path):
        import shutil
        shutil.rmtree(vector_store_path)

# Main app logic
def main():
    if not st.session_state.authenticated:
        show_login_form()
        return

    if st.session_state.is_admin:
        show_admin_panel()
        return

    # Show user's prompt usage and chat management in sidebar
    with st.sidebar:
        # Title and user info section
        st.title("🧠MemoryBot🤖")
        
        # Create a container for the logout button and user info
        col1, col3 = st.columns([6,4])
        with col1:
            st.write(f"User: {st.session_state.username}")
        with col3:
            if st.button("Logout", type="primary", key="user_logout", use_container_width=True):
                st.session_state.authenticated = False
                st.session_state.username = None
                st.session_state.chats = {}
                st.session_state.current_chat = None
                st.rerun()
        
        # User info and prompt count
        user_settings = get_user_settings(st.session_state.username)
        prompts_used = user_settings.get('prompts_used', 0)
        max_prompts = user_settings.get('max_prompts', DEFAULT_MAX_PROMPTS)
        st.write(f"Prompts: {prompts_used}/{max_prompts}")
        
        # Add warning if close to limit
        if prompts_used >= max_prompts:
            st.error("⚠️ Prompt limit reached!")
        elif prompts_used >= (max_prompts * 0.9):  # Warning at 90% usage
            st.warning(f"⚠️ Only {max_prompts - prompts_used} prompts remaining!")
        
        st.divider()

        # Add chat selection/management
        st.subheader("💬 Your Chats")
        
        # New chat button
        if st.button("➕ New Chat", use_container_width=True):
            new_chat_id = str(uuid.uuid4())
            st.session_state.chats[new_chat_id] = {
                "title": "New Chat",
                "messages": [],
                "created_at": datetime.now(),
                "vector_store": None,
                "memory": ConversationBufferMemory(),
                "qa_memory": ConversationBufferMemory(memory_key="chat_history", return_messages=True),
                "chat_history": []
            }
            st.session_state.current_chat = new_chat_id
            save_chat(st.session_state.username, new_chat_id, st.session_state.chats[new_chat_id])
            st.rerun()

        # Display existing chats
        for chat_id, chat in st.session_state.chats.items():
            chat_title = chat.get("title", "Untitled Chat")
            col1, col2 = st.columns([4, 1])
            
            with col1:
                if st.button(chat_title, key=f"chat_{chat_id}", use_container_width=True):
                    st.session_state.current_chat = chat_id
                    st.rerun()
            
            with col2:
                if st.button("🗑️", key=f"delete_{chat_id}"):
                    if chat_id == st.session_state.current_chat:
                        st.session_state.current_chat = None
                    del st.session_state.chats[chat_id]
                    delete_chat(st.session_state.username, chat_id)
                    st.rerun()

    # Main chat interface
    if st.session_state.current_chat is not None:
        chat = st.session_state.chats[st.session_state.current_chat]
        st.title(chat["title"])
        
        # Add chat title editing
        new_title = st.text_input("Chat Title", chat["title"])
        if new_title != chat["title"]:
            chat["title"] = new_title
            save_chat(st.session_state.username, st.session_state.current_chat, chat)
            st.rerun()

        # File upload section
        uploaded_files = st.file_uploader(
            "Upload PDF documents", 
            type=['pdf'], 
            accept_multiple_files=True,
            key=f"uploader_{st.session_state.current_chat}"
        )
        
        if uploaded_files:
            if st.button("Process PDFs"):
                with st.spinner("Processing PDFs..."):
                    chat["vector_store"] = process_pdf(uploaded_files)
                    save_chat(st.session_state.username, st.session_state.current_chat, chat)
                    st.success("PDFs processed successfully! You can now ask questions about them.")
                    st.rerun()

        # Display chat history in main area
        with st.expander("Chat History", expanded=True):
            for message in chat["messages"]:
                if message["role"] == "user":
                    with st.chat_message(message["role"], avatar="🧑"):
                        st.markdown(message["content"])
                else:
                    with st.chat_message(message["role"], avatar="🤖"):
                        st.markdown(message["content"])

        # Check prompt limit before showing input
        can_send, message = can_send_prompt(st.session_state.username)
        if not can_send:
            st.error(message)
            return

        # Input box for user message
        if prompt := st.chat_input("Type your message here..."):
            # Double check limit (in case of race conditions)
            can_send, message = can_send_prompt(st.session_state.username)
            if not can_send:
                st.error(message)
                return
            
            # Update prompt count
            success, result = update_prompt_count(st.session_state.username)
            if not success:
                st.error(f"Failed to update prompt count: {result}")
                return
            
            # Display user message
            with st.chat_message("user", avatar="🧑"):
                st.markdown(prompt)
            chat["messages"].append({"role": "user", "content": prompt})

            # Process response
            with st.chat_message("assistant", avatar="🤖"):
                response_placeholder = st.empty()
                stop_placeholder = st.empty()
                
                stop_button = stop_placeholder.button("Stop")
                if stop_button:
                    st.session_state.stop_signal = True
                
                if not st.session_state.stop_signal:
                    # Choose between regular conversation and PDF-based QA
                    if chat["vector_store"] is not None:
                        llm = ChatOpenAI(model="gpt-4", openai_api_key=OPENAI_API_KEY)
                        qa_chain = ConversationalRetrievalChain.from_llm(
                            llm=llm,
                            retriever=chat["vector_store"].as_retriever(),
                            memory=chat["qa_memory"],
                            return_source_documents=True,
                            chain_type="stuff"
                        )
                        
                        result = qa_chain({"question": prompt})
                        response = result["answer"]
                        chat["chat_history"].append((prompt, response))
                    else:
                        llm = ChatOpenAI(model="gpt-4", openai_api_key=OPENAI_API_KEY)
                        conversation = ConversationChain(
                            llm=llm,
                            memory=chat["memory"],
                            verbose=True
                        )
                        response = conversation.run(input=prompt)

                    # Display the response
                    response_placeholder.markdown(response)
                    chat["messages"].append({"role": "assistant", "content": response})
                    stop_placeholder.empty()
                else:
                    response_placeholder.markdown("Response generation stopped.")
                    st.session_state.stop_signal = False
                    stop_placeholder.empty()

            # Save the updated chat
            save_chat(st.session_state.username, st.session_state.current_chat, chat)
            
            # Force refresh to update prompt count display
            st.rerun()

    else:
        st.title("Welcome to MemoryBot🤖")
        st.write("Select a chat from the sidebar or start a new one.")

if __name__ == "__main__":
    main()