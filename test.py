import streamlit as st
import logging
import os
import tempfile
import shutil
import pdfplumber
import ollama

from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama.chat_models import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever
from typing import List, Tuple, Dict, Any, Optional

# Set protobuf environment variable to avoid error messages
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

# Streamlit page configuration
st.set_page_config(
    page_title="PDF RAG Chat App",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def extract_text_from_pdf(file_upload) -> str:
    """Extract all text from the uploaded PDF."""
    logger.info("Extracting text from PDF")
    with pdfplumber.open(file_upload) as pdf:
        text = "".join(page.extract_text() for page in pdf.pages)
    return text


def create_vector_db(file_upload) -> Chroma:
    """Create a vector database from the extracted PDF content."""
    logger.info("Creating vector database")
    temp_dir = tempfile.mkdtemp()

    path = os.path.join(temp_dir, file_upload.name)
    with open(path, "wb") as f:
        f.write(file_upload.getvalue())
        logger.info(f"PDF saved at {path}")
        loader = UnstructuredPDFLoader(path)
        documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(documents)
    logger.info("PDF split into chunks")

    embeddings = OpenAIEmbeddings()  # Replace with your embedding model
    vector_db = Chroma.from_documents(chunks, embeddings, collection_name="pdf_rag")
    logger.info("Vector database created")

    shutil.rmtree(temp_dir)
    return vector_db


def process_question(question: str, vector_db: Chroma, model: str) -> str:
    """Query the vector database and generate a response using the selected model."""
    logger.info(f"Processing question: {question} with model {model}")
    retriever = vector_db.as_retriever()
    llm = ChatOpenAI(model=model)  # Replace with Llama 3.2 integration

    chain = RetrievalQA.from_chain_type(
        retriever=retriever,
        llm=llm,
        chain_type="stuff",
        chain_prompt=PromptTemplate(
            template="""You are a helpful assistant. Use the provided context to answer the question.
Context: {context}
Question: {question}"""
        ),
    )
    response = chain.run({"question": question})
    return response


def main():
    """Main function to run the Streamlit app."""
    st.title("📚 PDF Chat with RAG")
    st.sidebar.markdown("### Upload a PDF and Chat!")

    # Session state for the vector database
    if "vector_db" not in st.session_state:
        st.session_state["vector_db"] = None

    # File upload
    uploaded_file = st.sidebar.file_uploader("Upload your PDF", type=["pdf"])

    if uploaded_file:
        with st.spinner("Processing PDF..."):
            st.session_state["vector_db"] = create_vector_db(uploaded_file)
            st.success("PDF processed successfully!")

    # Chat interface
    if st.session_state["vector_db"]:
        st.subheader("Chat with your document")
        user_query = st.text_input("Enter your question:")

        if user_query:
            with st.spinner("Getting response..."):
                response = process_question(
                    user_query, st.session_state["vector_db"], model="gpt-4"
                )  # Replace model with Llama 3.2
                st.write("### Response")
                st.write(response)

    # Option to clear session state
    if st.sidebar.button("Clear Data"):
        st.session_state.clear()
        st.success("Session cleared. Upload a new PDF to start again.")


if __name__ == "__main__":
    main()
