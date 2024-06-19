import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import random
import copy

load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))






def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        pdf_reader= PdfReader(pdf)
        for page in pdf_reader.pages:
            text+= page.extract_text()
    return  text



def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def query_vector_store(vector_store, user_input):

    results = vector_store.similarity_search(user_input, k=5)  # Retrieve top 5 relevant chunks
    return results

def get_conversational_chain():

    prompt_template = """
    Answer the question as detailed as possible from the provided context(minimum 50 words), make sure to provide all the details, don't provide the wrong answer. You may use your own knowledge but make sure that the information is relevant\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)

    prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain


def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()

    
    response = chain(
        {"context":docs, "question": user_question, "input_documents": docs}
        , return_only_outputs=True)

    print(response)
    st.write("Reply: ", response["output_text"])

def generate_random_question(relevant_chunks):
    if not relevant_chunks:
        return "No relevant information found."
    
    selected_chunk = random.choice(relevant_chunks).page_content
    output_parser = StrOutputParser()
    # Use the language model to generate a question
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)

    prompt_template = PromptTemplate(input_variables=['selected_chunk'], template='Generate a detailed interview question based on {selected_chunk}. Only give the question. DO NOT GIVE THE EXPECTED ANSWER. Question should not be longer than 40 words.You can use the following examples : UserInput: Queue. SystemOuput: Explain the first in first out data structure using a real life example. UserInput: tree. SystemOuput: Explain insertion in a binart search tree.')
    chain = prompt_template | model | output_parser
    response = chain.invoke({'selected_chunk':selected_chunk})

    print(response)
    # question = response  # Adjust based on the response structure of your LLM

    return response


def main():
    st.set_page_config(page_title="Chat PDF")
    st.header("Chat with PDF using Gemini💁") 
    
    user_subject = st.text_input("What topic would you like to test today")
    
    if user_subject and 'question' not in st.session_state:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        relevant_chunks = query_vector_store(vector_store, user_subject)
        question = generate_random_question(relevant_chunks)
        st.session_state.question = question

    if 'question' in st.session_state:
        st.write(st.session_state.question)
        
        user_answer = st.text_input("Write your answer here:", key="user_answer")
        
        if user_answer:
            answer = user_input(st.session_state.question)
            st.write(answer)

if __name__ == "__main__":
    main()