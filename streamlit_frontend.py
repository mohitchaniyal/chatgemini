from urllib import response
import streamlit as st
from langgraph_backend import chatbot
from langchain_core.messages import HumanMessage
import uuid
from langgraph_backend import checkpointer


# ---------- Helper Functions ----------
def generate_uuid():
    return str(uuid.uuid4())

def reset_chat():
    st.session_state['message_history'] = []
    st.session_state['thread_id'] = generate_uuid()
    add_thread(st.session_state['thread_id'])

def add_thread(thread_id):
    if thread_id not in st.session_state['chat_threads']:
        st.session_state['chat_threads'].append(thread_id)


# ---------- Session State Setup ----------
if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []

if "chat_threads" not in st.session_state:
    st.session_state['chat_threads'] = list(set(
        state.config["configurable"]['thread_id'] for state in checkpointer.list(None)
    ))

if 'thread_id' not in st.session_state:
    st.session_state['thread_id'] = generate_uuid()
    add_thread(st.session_state['thread_id'])


# ---------- Sidebar ----------
with st.sidebar:
    st.markdown(
        "<h2 style='text-align:center;'>LangGraph Chatbot</h2>",
        unsafe_allow_html=True
    )
    st.divider()

    if st.button("New Chat", use_container_width=True):
        reset_chat()

    st.markdown("###Previous Chats")
    for thread in st.session_state['chat_threads']:
        if st.button(f"🗂 {thread[:8]}...", use_container_width=True):
            state = chatbot.get_state({"configurable": {"thread_id": thread}})
            st.session_state['thread_id'] = thread

            if state.values:
                messages = state.values["messages"]
                st.session_state['message_history'] = [
                    {"role": 'user' if isinstance(msf, HumanMessage) else 'assistant',
                     "content": getattr(msf, "content", "")}
                    for msf in messages
                ]
            else:
                st.session_state['message_history'] = []


# ---------- Chat Messages ----------
st.markdown(
    """
    <style>
    .stChatMessage {
        padding: 12px;
        border-radius: 12px;
        margin-bottom: 10px;
    }
    .stChatMessage.user {
        background-color: #DCF8C6;
        color: #000;
    }
    .stChatMessage.assistant {
        background-color: #F1F0F0;
        color: #000;
    }
    </style>
    """,
    unsafe_allow_html=True
)

for msg in st.session_state['message_history']:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# ---------- Chat Input ----------
st.user_input = st.chat_input("Type your message...")
if st.user_input:
    st.session_state['message_history'].append({"role": "user", "content": st.user_input})
    with st.chat_message("user"):
        st.markdown(st.user_input)

    config = {"configurable": {"thread_id": st.session_state['thread_id']}}
    stream = chatbot.stream(
        {"messages": [HumanMessage(content=st.user_input)]},
        config=config,
        stream_mode='messages'
    )

    with st.chat_message("assistant"):
        ai_message = st.write_stream(message.content for message, metadata in stream)
        st.session_state['message_history'].append({"role": "assistant", "content": ai_message})
