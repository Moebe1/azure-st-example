import streamlit as st
import os
import math
import re
from typing import Optional, Literal, List, Dict, Any

############################################################
# NOTE TO READER:
# This code replaces the previous LATS code with a RefleXion
# approach (based on LangGraph) while keeping the same API and
# structure from your prior LATS-based example. We do not remove
# any features—token counting, Távily tool usage, streaming, etc.
# The difference is that we define a REFLEXION AGENT (rather than
# LATS) to generate, reflect on, and revise the answer.
############################################################

# =============================================================================
# Streamlit Imports & Setup
# =============================================================================
st.set_page_config(page_title="Reflexion Agent", page_icon="🤖")
from openai import AzureOpenAI, OpenAIError

# We still use the Távily search
from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper
from langchain_community.tools.tavily_search import TavilySearchResults

# LangGraph imports
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers.openai_tools import JsonOutputToolsParser
from langchain_core.prompt_values import ChatPromptValue
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import chain as chain_runnable, RunnableConfig
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field, ValidationError
from typing_extensions import TypedDict

# =============================================================================
# Configuration - Azure OpenAI + Tavily
# =============================================================================
AZURE_OPENAI_API_KEY = st.secrets.get("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = st.secrets.get("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_VERSION = st.secrets.get("AZURE_OPENAI_API_VERSION")
TAVILY_API_KEY = st.secrets.get("TAVILY_API_KEY", "")

os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY  # Távily uses environment var

# List of available Azure deployments
AVAILABLE_MODELS = [
    "o1-mini",
    "gpt-4o",
    "gpt-4o-mini"
]

# AzureOpenAI client creation
client = AzureOpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_version=AZURE_OPENAI_API_VERSION
)

# =============================================================================
# Tools & Tavily
# =============================================================================
search_api = TavilySearchAPIWrapper()
tavily_tool = TavilySearchResults(api_wrapper=search_api)
tavily_tool.max_results = 5  # if allowed

# We'll define a structured "search" function that the reflexion agent can call:
def run_tavily(queries: list[str]) -> list[dict]:
    """
    Run Távily search for each query in 'queries', returning a list of results.
    """
    outputs = []
    for q in queries:
        r = tavily_tool._run(q)
        outputs.append({"query": q, "results": r})
    return outputs

class ReflexionToolInput(BaseModel):
    search_queries: List[str] = Field(description="Search queries to improve or refine the answer")

@StructuredTool.from_function
def reflexion_search(search_queries: List[str]) -> list[dict]:
    """
    Távily search invocation function used by the Reflexion agent.
    """
    return run_tavily(search_queries)

# We define a single tool node with name "ReflexionTool", which references the function:
tool_node = ToolNode(tools=[reflexion_search])

# =============================================================================
# Reflection Actor Data Models
# =============================================================================
class ReflectionCritique(BaseModel):
    missing: str = Field(description="Critique of what is missing from the answer.")
    superfluous: str = Field(description="Critique of any extraneous info in the answer.")

class ReflexionAnswer(BaseModel):
    """
    For the reflexion-based approach, we ask the model to produce:
    1) 'answer': the actual text
    2) 'reflection': ReflectionCritique
    3) 'search_queries': up to 3 queries for Távily
    """
    answer: str = Field(description="Revised or initial answer to the user's question (~250 words).")
    reflection: ReflectionCritique
    search_queries: List[str]

# We'll define prompt templates for "initial draft" and "revision"
SYSTEM_INSTRUCTIONS = """You are a 'Reflexion' AI assistant specialized in step-by-step refinement.
1. Provide an answer to the user's question (about 250 words).
2. Provide a reflection about missing info or superfluous content.
3. Provide 1-3 search queries to help refine or expand the answer if needed.
Make sure your final answer is relevant and addresses the question thoroughly but concisely."""

ACTOR_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_INSTRUCTIONS),
        MessagesPlaceholder(variable_name="messages"),
        (
            "user",
            "Now reflect and respond using the ReflexionAnswer function in valid JSON."
        ),
    ]
)

# We'll define a chain that calls the model to produce ReflexionAnswer
parser = JsonOutputToolsParser(return_id=True)

@chain_runnable
def reflexion_chain(inputs: dict, config: RunnableConfig):
    """
    Creates a single-turn response from the model, returning:
      - ai_msg in messages
      - parsed items as tool_calls
    so we do NOT break Azure Chat by including tool calls in the messages list.
    """
    model_name = config["configurable"].get("model_name", "gpt-4o")
    max_tokens = config["configurable"].get("max_tokens", 500)
    verbose = config["configurable"].get("verbose", False)

    # Build final prompt
    messages_val = ChatPromptValue(
        messages=ACTOR_PROMPT.format_prompt(messages=inputs["messages"]).to_messages()
    )

    # Azure call
    resp = client.chat.completions.create(
        model=model_name,
        messages=messages_val.to_messages(),  # must contain role & content for each item
        stream=False,
        n=1,
        max_tokens=max_tokens
    )

    # usage info
    usage_info = getattr(resp, "usage", None)
    if usage_info and "total_tokens_used" in st.session_state:
        st.session_state["total_tokens_used"] += usage_info.total_tokens

    if not resp or not resp.choices:
        # fallback
        return {
            "messages": [AIMessage(content="No response")],
            "tool_calls": []
        }

    ai_msg = resp.choices[0].message.to_dict()  # has role='assistant'
    parsed = parser.invoke(ai_msg)  # might contain tool calls

    # Instead of returning {"messages": [ai_msg] + parsed}, do:
    # We store the AI message in "messages" and the parser results in "tool_calls".
    # This ensures we do NOT feed invalid dicts to Azure next time.
    return {
        "messages": [ai_msg],
        "tool_calls": parsed
    }

# We'll re-use reflexion_chain for "revision_chain" if needed
revision_chain = reflexion_chain

# =============================================================================
# Defining the Graph for Reflexion
# =============================================================================
class ReflexionState(TypedDict):
    messages: List[Any]      # the conversation
    tool_calls: List[Any]    # any parser outputs from the last step

builder = StateGraph(ReflexionState)
builder.add_node("draft", reflexion_chain)
builder.add_node("tool", tool_node)
builder.add_node("revise", revision_chain)

# Edges
builder.add_edge(START, "draft")
builder.add_edge("draft", "tool")
builder.add_edge("tool", "revise")

# We'll define a simple loop condition: if the user wants multiple expansions or if we set a max
MAX_ITERATIONS = 3

def reflexion_loop(state: ReflexionState) -> str:
    # count how many times we've invoked the chain
    expansions = 0
    for call in state.get("tool_calls", []):
        if isinstance(call, dict) and call.get("type") == "tool_call":
            expansions += 1

    if expansions >= MAX_ITERATIONS:
        return END
    return "tool"

builder.add_conditional_edges("revise", reflexion_loop, ["tool", END])
reflexion_graph = builder.compile()

# =============================================================================
# Streamlit Handling
# =============================================================================
def get_langchain_agent(model_choice: str, system_prompt: str, verbose: bool):
    """
    Re-implements the get_langchain_agent but now as a Reflexion agent, 
    using the same signature so we do not break the existing code or API.
    """
    agent = reflexion_graph
    return agent

def main():
    st.title("Reflexion Agent with Távily Search + Azure + Token Counting")

    if "messages" not in st.session_state:
        st.session_state["messages"] = [
            {"role": "assistant", "content": "Hello! I'm a Reflexion-based AI. How can I help?"}
        ]
    if "total_tokens_used" not in st.session_state:
        st.session_state["total_tokens_used"] = 0

    if "tool_calls" not in st.session_state:
        st.session_state["tool_calls"] = []

    with st.sidebar:
        st.header("Configuration")
        model_choice = st.selectbox("Azure Model:", AVAILABLE_MODELS, index=0)
        system_prompt = st.text_area("System Prompt", "You are an AI assistant.")
        streaming_enabled = st.checkbox("Enable Streaming", value=False)
        verbosity_enabled = st.checkbox("Enable Verbose Mode", value=False)
        max_tokens = st.number_input("Max Tokens per Response", min_value=50, max_value=4096, value=500)
        st.write("**Total Tokens Used**:", st.session_state["total_tokens_used"])
        if st.button("Clear Conversation"):
            st.session_state["messages"] = [
                {"role": "assistant", "content": "Conversation cleared. How can I help now?"}
            ]
            st.session_state["tool_calls"] = []
            st.session_state["total_tokens_used"] = 0
            st.experimental_rerun()

    # Display conversation
    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    if prompt := st.chat_input("Ask a question..."):
        st.session_state["messages"].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        with st.chat_message("assistant"):
            msg_placeholder = st.empty()
            # Build a reflexion state
            reflexion_state: ReflexionState = {
                "messages": st.session_state["messages"],
                "tool_calls": st.session_state["tool_calls"]
            }

            agent = get_langchain_agent(model_choice, system_prompt, verbosity_enabled)
            config = {
                "configurable": {
                    "model_name": model_choice,
                    "verbose": verbosity_enabled,
                    "max_tokens": max_tokens
                },
                "callbacks": None
            }

            response_text = ""
            try:
                for step_output in agent.stream(reflexion_state, stream_mode="values", config=config):
                    # step_output is a dict like {"draft": <ReflexionState>} or {"tool": <ReflexionState>} ...
                    node_name, node_data = next(iter(step_output.items()))
                    reflexion_state = node_data  # updated state

                # reflexion_state["messages"] likely has the final AI message
                final_msg = ""
                for msg in reversed(reflexion_state["messages"]):
                    if isinstance(msg, dict) and msg.get("role") == "assistant":
                        final_msg = msg.get("content", "")
                        break
                final_msg = re.sub(r'[ \t]+$', '', final_msg, flags=re.MULTILINE)
                final_msg = re.sub(r'^\s*\n', '', final_msg)
                final_msg = re.sub(r'\n\s*$', '', final_msg)

                response_text = final_msg
                msg_placeholder.write(final_msg)
            except Exception as e:
                st.error(f"Error: {e}")
                response_text = "I encountered an error. Please try again."

        st.session_state["messages"].append({"role": "assistant", "content": response_text})
        # Also store updated tool_calls
        st.session_state["tool_calls"] = reflexion_state.get("tool_calls", [])

if __name__ == "__main__":
    main()