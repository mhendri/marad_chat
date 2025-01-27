import os
import openai
import streamlit as st
from datetime import datetime
from streamlit.logger import get_logger
from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader


logger = get_logger('Langchain-Chatbot')
VECTOR_STORE_PATH = "vector_store.faiss"


#decorator
def enable_chat_history(func):
    if os.environ.get("OPENAI_API_KEY"):

        # to clear chat history after swtching chatbot
        current_page = func.__qualname__
        if "current_page" not in st.session_state:
            st.session_state["current_page"] = current_page
        if st.session_state["current_page"] != current_page:
            try:
                st.cache_resource.clear()
                del st.session_state["current_page"]
                del st.session_state["messages"]
            except:
                pass

        # to show chat history on ui
        if "messages" not in st.session_state:
            st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]
        for msg in st.session_state["messages"]:
            st.chat_message(msg["role"]).write(msg["content"])

    def execute(*args, **kwargs):
        func(*args, **kwargs)
    return execute

def display_msg(msg, author):
    """Method to display message on the UI

    Args:
        msg (str): message to display
        author (str): author of the message -user/assistant
    """
    st.session_state.messages.append({"role": author, "content": msg})
    st.chat_message(author).write(msg)

def choose_custom_openai_key():
    openai_api_key = st.sidebar.text_input(
        label="OpenAI API Key",
        type="password",
        placeholder="sk-...",
        key="SELECTED_OPENAI_API_KEY"
        )
    if not openai_api_key:
        st.error("Please add your OpenAI API key to continue.")
        st.info("Obtain your key from this link: https://platform.openai.com/account/api-keys")
        st.stop()

    model = "gpt-4o"
    try:
        client = openai.OpenAI(api_key=openai_api_key)
        available_models = [{"id": i.id, "created":datetime.fromtimestamp(i.created)} for i in client.models.list() if str(i.id).startswith("gpt")]
        available_models = sorted(available_models, key=lambda x: x["created"])
        available_models = [i["id"] for i in available_models]

        model = st.sidebar.selectbox(
            label="Model",
            options=available_models,
            key="SELECTED_OPENAI_MODEL"
        )
    except openai.AuthenticationError as e:
        st.error(e.body["message"])
        st.stop()
    except Exception as e:
        print(e)
        st.error("Something went wrong. Please try again later.")
        st.stop()
    return model, openai_api_key

def configure_llm():
    available_llms = ["gpt-4o","llama3.1:8b","llama3.2:3b","use your openai api key"]
    llm_opt = st.sidebar.radio(
        label="LLM",
        options=available_llms,
        key="SELECTED_LLM"
        )

    if llm_opt == "llama3.1:8b":
        llm = ChatOllama(model="llama3.1", base_url=st.secrets["OLLAMA_ENDPOINT"])
    elif llm_opt == "llama3.2:3b":
        llm = ChatOllama(model="llama3.2", base_url=st.secrets["OLLAMA_ENDPOINT"])
    elif llm_opt == "gpt-4o":
        llm = ChatOpenAI(model_name=llm_opt, temperature=0, streaming=True, api_key=st.secrets["OPENAI_API_KEY"])
    else:
        model, openai_api_key = choose_custom_openai_key()
        llm = ChatOpenAI(model_name=model, temperature=0, streaming=True, api_key=openai_api_key)
    return llm

def print_qa(cls, question, answer):
    log_str = "\nUsecase: {}\nQuestion: {}\nAnswer: {}\n" + "------"*10
    logger.info(log_str.format(cls.__name__, question, answer))


def sync_st_session():
    for k, v in st.session_state.items():
        st.session_state[k] = v

@st.cache_resource
def configure_embedding_model():
    embedding_model = OpenAIEmbeddings()
    return embedding_model

def load_or_preload_documents(docs, directory="corpus"):
        """Load existing vector store or preload PDF documents from the specified directory."""
        if os.path.exists(VECTOR_STORE_PATH):
            st.info("Loading vector store from disk...")
            vector_store = FAISS.load_local(VECTOR_STORE_PATH, configure_embedding_model(), allow_dangerous_deserialization=True)
            st.success("Vector store loaded successfully.")
            return vector_store

        if not os.path.exists(directory):
            st.warning(f"Directory '{directory}' does not exist. Please ensure the folder contains your PDFs.")
            return None

        st.info("No existing vector store found. Loading and indexing PDF documents. This may take a few moments...")

        docs = []
        for file_name in os.listdir(directory):
            # Construct full file path
            file_path = os.path.join(directory, file_name)
            print(file_path)
            if os.path.isfile(file_path):  # Ensure it's a file and not a directory
                loader = PyPDFLoader(file_path)
                docs.extend(loader.load())

        # Split documents and store in vector db
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        splits = text_splitter.split_documents(docs)
        vector_store = FAISS.from_documents(splits, configure_embedding_model())

        # Save the vector store to disk
        vector_store.save_local(VECTOR_STORE_PATH)
        st.success(f"Successfully preloaded {len(docs)} PDF documents into the vector store and saved it to disk.")
        return vector_store