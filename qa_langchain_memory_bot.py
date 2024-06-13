import pdfplumber
import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from streamlit_chat import message
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Constants
system_message = "You are a helpful assistant."
openai_api_key = os.getenv("OPENAI_API_KEY")

def extract_text_from_pdf(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text()
    return text

def get_conversation():
    template = """The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know.

Current conversation:
{history}
Human: {input}
AI:"""
    prompt = ChatPromptTemplate.from_template(template)
    
    llm = ChatOpenAI(temperature=0, api_key=openai_api_key)
    memory = ConversationBufferMemory(memory_key="history", return_messages=True)
    
    conversation = ConversationChain(
        llm=llm, prompt=prompt, memory=memory, verbose=True
    )
    return conversation

st.markdown('''# PDF Question Answering System 📚

Here is a Streamlit-powered PDF Question Answering System used to extract answers from PDF documents.

## Features 🚀

- **Intuitive Interface**: User-friendly Streamlit interface. 🖥️
- **Advanced NLP**: Leverages state-of-the-art NLP models for accurate answers. 🧠
''')

uploaded_pdf = st.file_uploader("Upload a PDF for Q&A", type=["pdf"])
if uploaded_pdf is not None:
    temp_pdf_path = "temp_uploaded.pdf"
    with open(temp_pdf_path, "wb") as f:
        f.write(uploaded_pdf.getbuffer())

    extracted_text = extract_text_from_pdf(temp_pdf_path)

    if len(extracted_text) > 2048:
        extracted_text = extracted_text[:2048]

    question = st.text_input("Ask a question based on the uploaded PDF:")
    if question:
        try:
            if "conversation" not in st.session_state:
                st.session_state.conversation = get_conversation()

            conversation = st.session_state.conversation
            response = conversation.run({"input": question})
            
            st.write(response)

            # Display the conversation history
            chat_history = conversation.memory.chat_memory.messages
            for i, entry in enumerate(chat_history, 1):  # Display all messages in history
                if entry.type == 'system':
                    message(entry.content, key=str(i))
                else:
                    message(entry.content, is_user=(entry.type == 'human'), key=str(i))

        except Exception as e:
            st.error(f"Failed to generate an answer: {str(e)}")
else:
    st.write("Upload a PDF file to get started.")
