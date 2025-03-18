from dotenv import load_dotenv

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
import asyncio

from astream_events_handler import invoke_our_graph   # Utility function to handle events from astream_events from graph

load_dotenv()

st.title("GigaChat Agentic Reasoner 🤔")
prompt = st.chat_input()

st.markdown("### История разговора не хранится! Один вопрос - один ответ!")

# Initialize chat messages in session state
if "messages" not in st.session_state:
    # st.session_state["messages"] = [AIMessage(content="Задайте свой вопрос?")]
    st.session_state["messages"] = []


for msg in st.session_state.messages:
    if isinstance(msg, AIMessage):
        st.chat_message("assistant").write(msg.content)
    elif isinstance(msg, HumanMessage):
        st.chat_message("user").write(msg.content)


if prompt:
    st.session_state.messages.append(HumanMessage(content=prompt))
    st.chat_message("user").write(prompt)

    with st.chat_message("assistant"):
        placeholder = st.container()
        response = asyncio.run(invoke_our_graph(st.session_state.messages, placeholder))
        st.session_state.messages.append(AIMessage(response))
