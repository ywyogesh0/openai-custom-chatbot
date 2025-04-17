import streamlit as st
from langchain_text_splitters import RecursiveCharacterTextSplitter

# name of the chat-bot
st.header("Custom Open AI ChatBot")

# define sidebar to upload TXT file
with st.sidebar:
    st.title("TXT FILE")
    file = st.file_uploader("Upload TXT File To Train Open AI Model", type=["txt"])

# decode bytes into text after reading TXT file
# divide the file content into chunks
if file is not None:
    txt_content = file.read().decode("utf-8")
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n"],
        chunk_size=100,
        chunk_overlap=20,
        length_function=len
    )
    chunks = text_splitter.split_text(txt_content)
    st.write(chunks)
