import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS



def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        reader = PdfReader(pdf)
        for page in reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(raw_text):
    splitter = CharacterTextSplitter(
        separator = "\n",
        chunk_size = 1000,
        chunk_overlap=200,
        length_function=len)
    text_chunks = splitter.split_text(raw_text)
    return text_chunks

def get_vecotor_store(text_chunks):
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_texts(embedding=embeddings, texts=text_chunks)
    # vector_store.fit(text_chunks)
    return vector_store


def main():
    load_dotenv()
    st.set_page_config(page_title = "Chat with multiple pdf", page_icon = ":books:")
    st.header("Chat with multiple PDFs :books:")
    st.text_input("Ask a question about documents")
    with st.sidebar:
        st.subheader("Upload PDFs")
        pdf_docs = st.file_uploader("Choose a file click on process", type = "pdf" ,accept_multiple_files = True)
        if st.button("Process"):
            with st.spinner("Processing"):
            # get pdf text
                raw_text = get_pdf_text(pdf_docs)

            # get the text chuncks
                text_chunks = get_text_chunks(raw_text)
                st.write(text_chunks)

            # create vector store

            vector_store = get_vecotor_store(text_chunks)








if __name__=='__main__':
    main()