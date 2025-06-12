import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI
import os

# Set your OpenRouter API Key here
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Load and read PDF
st.title("RAG Chatbot using OpenRouter.ai")
pdf = st.file_uploader("Upload your PDF", type="pdf")

if pdf:
    pdf_reader = PdfReader(pdf)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() or ""

    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)

    # Embed and store in vector DB (FAISS)
    embeddings = HuggingFaceEmbeddings()
    vectorstore = FAISS.from_texts(chunks, embedding=embeddings)

    # Accept user query
    query = st.text_input("Ask a question about the PDF:")
    if query:
        docs = vectorstore.similarity_search(query=query, k=3)

        # Load LLM from OpenRouter
        llm = ChatOpenAI(
            openai_api_base="https://openrouter.ai/api/v1",
            openai_api_key=os.environ["OPENAI_API_KEY"],
            model="mistralai/mixtral-8x7b-instruct",  # You can also use llama-3 or others
            temperature=0.5
        )

        chain = load_qa_chain(llm, chain_type="stuff")
        response = chain.run(input_documents=docs, question=query)

        st.write("### Answer:")
        st.write(response)
