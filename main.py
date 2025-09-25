from urllib import response
import streamlit as st
from langchain_core.messages import HumanMessage
import uuid
import os
import tempfile
import sqlite3

# Import your chatbot components
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_community.tools import DuckDuckGoSearchRun


# ---------- Chatbot Initialization Function ----------
@st.cache_resource
def initialize_chatbot(api_key):
    """Initialize the chatbot with the provided API key"""
    
    # Initialize tools
    search_tool = DuckDuckGoSearchRun()
    tools = [search_tool]
    
    # Initialize LLM with the API key
    llm = ChatGoogleGenerativeAI(
        model='gemini-2.0-flash-exp',
        google_api_key=api_key
    )
    llm_with_tools = llm.bind_tools(tools)
    
    # Define state
    class ChatState(TypedDict):
        messages: Annotated[list[BaseMessage], add_messages]
    
    def chat_node(state: ChatState):
        messages = state['messages']
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}
    
    # Create temporary database for this session
    temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
    conn = sqlite3.connect(temp_db.name, check_same_thread=False)
    checkpointer = SqliteSaver(conn=conn)
    
    # Build the graph
    graph = StateGraph(ChatState)
    tool_node = ToolNode(tools=tools)
    
    # Add nodes
    graph.add_node("chat_node", chat_node)
    graph.add_node("tools", tool_node)
    
    # Add edges
    graph.add_edge(START, "chat_node")
    graph.add_conditional_edges("chat_node", tools_condition)
    graph.add_edge("tools", "chat_node")
    graph.add_edge("chat_node", END)
    
    # Compile the graph
    chatbot = graph.compile(checkpointer=checkpointer)
    
    return chatbot, checkpointer


# ---------- Helper Functions ----------
def generate_uuid():
    return str(uuid.uuid4())

def reset_chat():
    st.session_state['message_history'] = []
    st.session_state['thread_id'] = generate_uuid()
    add_thread(st.session_state['thread_id'])

def add_thread(thread_id):
    if 'chat_threads' in st.session_state and thread_id not in st.session_state['chat_threads']:
        st.session_state['chat_threads'].append(thread_id)


# ---------- Main App ----------
st.set_page_config(page_title="LangGraph Chatbot", page_icon="🤖", layout="wide")

# API Key Input
if 'api_key' not in st.session_state:
    st.session_state['api_key'] = ''

if not st.session_state['api_key']:
    st.title("ChatGemini")
    st.markdown("### Please enter your Google Gemini API Key to continue")
    
    with st.form("api_key_form"):
        api_key_input = st.text_input(
            "Google Gemini API Key",
            type="password",
            placeholder="Enter your API key here...",
            help="You can get your API key from https://aistudio.google.com/app/apikey"
        )
        submit_button = st.form_submit_button("Connect", use_container_width=True)
        
        if submit_button:
            if api_key_input.strip():
                try:
                    # Test the API key by initializing the chatbot
                    with st.spinner("Validating API key..."):
                        chatbot, checkpointer = initialize_chatbot(api_key_input.strip())
                        
                    # If successful, store the API key and components
                    st.session_state['api_key'] = api_key_input.strip()
                    st.session_state['chatbot'] = chatbot
                    st.session_state['checkpointer'] = checkpointer
                    st.success("API key validated successfully!")
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Error validating API key: {str(e)}")
                    st.error("Please check your API key and try again.")
            else:
                st.error("Please enter a valid API key")
    
    # Instructions
    with st.expander("ℹ How to get your API key"):
        st.markdown("""
        1. Go to [Google AI Studio](https://aistudio.google.com/app/apikey)
        2. Sign in with your Google account
        3. Click "Create API Key"
        4. Copy the generated API key
        5. Paste it in the field above
        """)
    
    st.stop()

# Initialize chatbot if not already done
if 'chatbot' not in st.session_state:
    try:
        with st.spinner("Initializing chatbot..."):
            chatbot, checkpointer = initialize_chatbot(st.session_state['api_key'])
            st.session_state['chatbot'] = chatbot
            st.session_state['checkpointer'] = checkpointer
    except Exception as e:
        st.error(f"Error initializing chatbot: {str(e)}")
        st.session_state['api_key'] = ''  # Reset API key to show input form again
        st.rerun()

# Get chatbot and checkpointer from session state
chatbot = st.session_state['chatbot']
checkpointer = st.session_state['checkpointer']

# ---------- Session State Setup ----------
if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []

if "chat_threads" not in st.session_state:
    try:
        st.session_state['chat_threads'] = list(set(
            state.config["configurable"]['thread_id'] for state in checkpointer.list(None)
        ))
    except:
        st.session_state['chat_threads'] = []

if 'thread_id' not in st.session_state:
    st.session_state['thread_id'] = generate_uuid()
    add_thread(st.session_state['thread_id'])


# ---------- Sidebar ----------
with st.sidebar:
    st.markdown(
        "<h2 style='text-align:center;'>ChatGemini</h2>",
        unsafe_allow_html=True
    )
    
    # API Key Status
    # st.success("Connected")
    if st.button("Change API Key", use_container_width=True):
        # Clear the API key and related session state
        for key in ['api_key', 'chatbot', 'checkpointer']:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()
    
    st.divider()

    if st.button("➕ New Chat", use_container_width=True):
        reset_chat()

    st.markdown("### Previous Chats")
    for thread in st.session_state['chat_threads']:
        if st.button(f"🗂 {thread[:8]}...", use_container_width=True):
            try:
                state = chatbot.get_state({"configurable": {"thread_id": thread}})
                st.session_state['thread_id'] = thread

                if state.values:
                    messages = state.values["messages"]
                    st.session_state['message_history'] = [
                        {"role": 'user' if isinstance(msg, HumanMessage) else 'assistant',
                         "content": getattr(msg, "content", "")}
                        for msg in messages
                    ]
                else:
                    st.session_state['message_history'] = []
            except Exception as e:
                st.error(f"Error loading chat: {str(e)}")


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
user_input = st.chat_input("Type your message...")
if user_input:
    st.session_state['message_history'].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    config = {"configurable": {"thread_id": st.session_state['thread_id']}}
    
    try:
        stream = chatbot.stream(
            {"messages": [HumanMessage(content=user_input)]},
            config=config,
            stream_mode='messages'
        )

        with st.chat_message("assistant"):
            ai_message = st.write_stream(message.content for message, metadata in stream)
            st.session_state['message_history'].append({"role": "assistant", "content": ai_message})
            
    except Exception as e:
        st.error(f"Error generating response: {str(e)}")
        st.error("Please try again or check your API key.")