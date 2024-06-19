import streamlit as st
import PyPDF2
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_pdf_text(pdf_file_paths):
    text = ""
    for pdf_path in pdf_file_paths:
        try:
            with open(pdf_path, "rb") as fh:
                reader = PyPDF2.PdfReader(fh)
                for page in reader.pages:
                    text += page.extract_text()
        except FileNotFoundError:
            print(f"File not found: {pdf_path}")
        except Exception as e:
            print(f"An error occurred with file {pdf_path}: {e}")
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def main():
    # st.set_page_config("Chat PDF")
    # st.header("Chat with PDF using Gemini💁")

    # user_question = st.text_input("Ask a Question from the PDF Files")

    # if user_question:
    #     user_input(user_question)

    # with st.sidebar:
        # st.title("Menu:")
        # pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        # pdf_docs = "DSA.pdf"
        # if st.button("Submit & Process"):
        #     with st.spinner("Processing..."):
    pdf_file_paths = [
        r"C:\Users\chira\OneDrive\Desktop\Hackathon\DSA.pdf",
        r"C:\Users\chira\OneDrive\Desktop\Hackathon\book2.pdf",
        r"C:\Users\chira\OneDrive\Desktop\Hackathon\book4.pdf",
        r"C:\Users\chira\OneDrive\Desktop\Hackathon\book5.pdf",
        r"C:\Users\chira\OneDrive\Desktop\Hackathon\book6.pdf",
        r"C:\Users\chira\OneDrive\Desktop\Hackathon\book7.pdf",
        r"C:\Users\chira\OneDrive\Desktop\Hackathon\book8.pdf",
        r"C:\Users\chira\OneDrive\Desktop\Hackathon\book9.pdf",
        r"C:\Users\chira\OneDrive\Desktop\Hackathon\book10.pdf",
    ]

    raw_text = get_pdf_text(pdf_file_paths)
    if raw_text:
        text_chunks = get_text_chunks(raw_text)
        get_vector_store(text_chunks)
        print("Processing complete.")

    # raw_text = get_pdf_text(pdf_docs)
    # text_chunks = get_text_chunks(raw_text)
    # get_vector_store(text_chunks)
                # st.success("Done")

if __name__ == "__main__":
    main()
