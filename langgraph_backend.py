from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
import sqlite3
# from langgraph_backend import checkpointer
from langgraph.prebuilt import ToolNode,tools_condition
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool


search_tool = DuckDuckGoSearchRun()
tools = [search_tool]

# Load environment variables
load_dotenv()

# Initialize tools
search_tool = DuckDuckGoSearchRun()
tools = [search_tool]

# Initialize LLM and bind tools
llm = ChatGoogleGenerativeAI(model='gemini-2.0-flash-exp')  # Updated model name
llm_with_tools = llm.bind_tools(tools)

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def chat_node(state: ChatState):
    messages = state['messages']
    # Use the LLM with tools bound
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

# Database connection for checkpointing
conn = sqlite3.connect("chatbot.db", check_same_thread=False)
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

