import streamlit as st
from retrieval_service import RetrievalService
from llm_service import LLMService

st.set_page_config(layout="wide")
st.title("RAG-based Legal AI Chatbot")

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []
if "last_query" not in st.session_state:
    st.session_state["last_query"] = ""

llm_model = st.sidebar.selectbox(
    "Select LLM Model:",
    ["Groq", "Ollama", "Hugging Face"],
    index=0,
)

model_mapping = {
    "Ollama": "deepseek-r1:8b",
    "Groq": "llama-3.3-70b-versatile",
    "Hugging Face": "deepseek-ai/deepseek-v2",
}

for q, a in reversed(st.session_state["chat_history"]):
    st.chat_message("user").write(q)
    st.chat_message("assistant").write(a)

with st.container():
    col1, col2 = st.columns([4, 1])
    with col1:
        query = st.text_input(
            "Ask a legal question about Pakistani law:",
            key="user_query",
            label_visibility="collapsed",
            on_change=lambda: st.session_state.update(send_pressed=True)
        )
    with col2:
        send_pressed = st.button("Send")

if send_pressed or (query and query != st.session_state["last_query"]):
    if query.strip():
        st.session_state["last_query"] = query
        
        retriever = RetrievalService()
        context = retriever.retrieve_context(query)

        llm_service = LLMService(
            model_name=model_mapping[llm_model],
            chat_history=st.session_state["chat_history"],
            question=query,
            context=context,
        )
        
        if llm_model == "Ollama":
            response = llm_service.ollama_execution()
        elif llm_model == "Groq":
            response = llm_service.groq_execution()
        else:
            response = llm_service.hugging_face_execution()
        
        st.session_state["chat_history"].append((query, response))
        st.rerun()