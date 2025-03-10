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

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Set Streamlit page configuration
st.set_page_config(page_title='🧠MemoryBot🤖', layout='wide')

# Initialize session states
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

# Function to start a new chat
def new_chat():
    chat_id = len(st.session_state.chats) + 1
    st.session_state.chats[chat_id] = {
        "title": f"Chat {chat_id}",
        "messages": [],
        "memory": ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        ),
        "vector_store": None,
        "chat_history": []
    }
    st.session_state.current_chat = chat_id

# Function to rename the current chat
def rename_chat(new_title):
    if st.session_state.current_chat is not None:
        st.session_state.chats[st.session_state.current_chat]["title"] = new_title

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
                # Add edit button for each chat
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
                col1, col2 = st.columns(2)  # Changed to equal columns
                with col1:
                    if st.button("Save", key=f"save_{chat_id}", use_container_width=True):  # Added container width
                        st.session_state.chats[chat_id]["title"] = new_title
                        st.session_state[f"edit_mode_{chat_id}"] = False
                        st.rerun()  # Added rerun to refresh the UI
                with col2:
                    if st.button("Cancel", key=f"cancel_{chat_id}", use_container_width=True):  # Added container width
                        st.session_state[f"edit_mode_{chat_id}"] = False
                        st.rerun()  # Added rerun to refresh the UI
        
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
                memory=ConversationBufferMemory(memory_key="history", return_messages=True)
            )

        # Display assistant response with stop button
        with st.chat_message("assistant", avatar="🤖"):
            response_placeholder = st.empty()
            stop_placeholder = st.empty()
            
            stop_button = stop_placeholder.button("Stop")
            if stop_button:
                st.session_state.stop_signal = True
            
            if not st.session_state.stop_signal:
                if chat["vector_store"] is not None:
                    response = response
                else:
                    response = conversation.run(input=prompt)
                response_placeholder.markdown(response)
                chat["messages"].append({"role": "assistant", "content": response})
                stop_placeholder.empty()
            else:
                response_placeholder.markdown("Response generation stopped.")
                st.session_state.stop_signal = False
                stop_placeholder.empty()
else:
    st.title("Welcome to 🧠MemoryBot🤖")
    st.write("Select a chat from the sidebar or start a new one.")

