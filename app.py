from dotenv import load_dotenv

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
import asyncio

from astream_events_handler import invoke_our_graph   # Utility function to handle events from astream_events from graph

load_dotenv()

st.title("GigaChat Agentic Reasoner 🤔")
prompt = st.chat_input()

# st.markdown("#### Chat Streaming and Tool Calling using Astream Events")

# # Initialize the expander state
# if "expander_open" not in st.session_state:
#     st.session_state.expander_open = True

# Capture user input from chat input


# # Toggle expander state based on user input
# if prompt is not None:
#     st.session_state.expander_open = False  # Close the expander when the user starts typing

# # # st write magic
# # with st.expander(label="Simple Chat Streaming and Tool Calling using LangGraph's Astream Events", expanded=st.session_state.expander_open):
# #     """
# #     In this example, we're going to be creating our own events handler to stream our [_LangGraph_](https://langchain-ai.github.io/langgraph/)
# #     invocations with via [`astream_events (v2)`](https://langchain-ai.github.io/langgraph/how-tos/streaming-from-final-node/).
# #     This one is does not use any callbacks or external streamlit libraries and is asynchronous.
# #     we've implemented `on_llm_new_token`, a method that run on every new generation of a token from the ChatLLM model, and
# #     `on_tool_start` a method that runs on every tool call invocation even multiple tool calls, and `on_tool_end` giving final result of tool call.
# #     """

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
