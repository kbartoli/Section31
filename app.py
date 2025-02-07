import streamlit as st
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import FAISS
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.runnables.history import RunnableWithMessageHistory
import os

from dotenv import load_dotenv
load_dotenv()

#os.environ["HF_API"]=os.getenv("HF_API")



## set up Streamlit
st.title("Conversational RAG with PDF upload and chat history")
st.write("Upload PDF and Chat with the content")

# input Groq API key
api_key = st.text_input("API KEY:", type= "password")
HF_key = st.st.text_input("HF KEY:", type= "password")
embeddings = HuggingFaceEmbeddings(api_key = HF_key, model_name="all-MiniLM-L6-v2")

if api_key and HF_key:
    llm=ChatGroq(groq_api_key=api_key, model_name="gemma2-9b-it")
    #chat interface
    session_id = st.text_input("session ID", value = "defaul_session")
    #statefully manage chat history
    if "store" not in st.session_state:
        st.session_state.store={}

    uploaded_files= st.file_uploader("Choose a pdf file", type="pdf", accept_multiple_files=True)

    #process uploaded pdf
    if uploaded_files:
        documents=[]
        for uploaded_file in uploaded_files:
            temppdf=f"./temp.pdf"
            with open(temppdf,"wb") as file:
                file.write(uploaded_file.getvalue())
                file_name=uploaded_file.name

            loader = PyPDFLoader(temppdf)
            docs= loader.load()
            documents.extend(docs)

        # split text and embeddings w Hunngingface
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
        splits = text_splitter.split_documents(documents)
        vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
        retriever = vectorstore.as_retriever()
    
        contextualize_q_system_prompt=(
            "given a chat history and the latest user question which might reference context in the chat history,formulate a standalone question which can be understood without the chat history."
            "do not answer the question, just reformulate it if neeeded and otherwise return it as is."
        )

        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

        history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

        #answer question prompt
        system_prompt = (
            "You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question"
            "If you do not know the answer, say that you dont know. Use thress sentences maximum and keep the answer concise"
            "\n\n"
            "{context}"
        )
        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}")
            ]
        )

        question_answer_chain = create_stuff_documents_chain(llm,qa_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

        def get_session_history(session:str)->BaseChatMessageHistory:
            if session_id not in st.session_state.store:
                st.session_state.store[session_id]=ChatMessageHistory()
            return st.session_state.store[session_id]
        
        conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain, get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer"
        )

        user_input = st.text_input("Your question: ")
        if user_input:
            session_history = get_session_history(session_id)
            response = conversational_rag_chain.invoke(
                {"input": user_input},
                config ={
                    "configurable": {"session_id":session_id}
                }, #constructs a key "abc123" in store
            )

            st.write(st.session_state.store)
            st.write("Assistant:", response["answer"])
            st.write("Chat History:", session_history.messages)

else:
    st.warning("Please enter API key")



