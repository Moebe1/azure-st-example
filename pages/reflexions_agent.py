import streamlit as st
import os
import math
import re
from typing import Optional, Literal, List, Dict, Any
import json

############################################################
# NOTE TO READER:
# This code uses a "Reflexion" approach (LangGraph), ensuring
# that we never produce invalid "messages" entries for Azure.
# We store tool calls in a separate "tool_calls" field.
# The key fix is giving each node (including the tool node)
# a transform that returns a dict with "messages" & "tool_calls".
############################################################

# =============================================================================
# Streamlit Imports & Setup
# =============================================================================
st.set_page_config(page_title="Reflexion Agent", page_icon="🤖")
from openai import AzureOpenAI, OpenAIError

# Távily
from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper
from langchain_community.tools.tavily_search import TavilySearchResults

# LangGraph
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

# Create AzureOpenAI client
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
    """Run Távily search for each query in 'queries', returning a list of results."""
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
    Returns a list of { query, results } dicts.
    """
    return run_tavily(search_queries)

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
    answer: str = Field(description="A ~250 word answer to the user's question.")
    reflection: ReflectionCritique
    search_queries: List[str]

# Prompt for the "actor"
SYSTEM_INSTRUCTIONS = """You are a 'Reflexion' AI assistant specialized in step-by-step refinement.
1. Provide an answer to the user's question (~250 words).
2. Provide a reflection about missing info or superfluous content.
3. Provide 1-3 search queries to help refine or expand the answer if needed.
Make sure your final answer is relevant and addresses the question thoroughly but concisely."""

ACTOR_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_INSTRUCTIONS),
        MessagesPlaceholder(variable_name="messages"),
        ("user", "Now reflect and respond using the ReflexionAnswer function in valid JSON.")
    ]
)

parser = JsonOutputToolsParser(return_id=True)

# We'll store both messages and "tool_calls" in the state, to avoid mixing them.

class ReflexionState(TypedDict):
    messages: List[dict]
    tool_calls: List[dict]


# =============================================================================
# CHAIN: reflexion_chain
# =============================================================================
@chain_runnable
def reflexion_chain(inputs: Dict, config: RunnableConfig):
    """
    We generate a single response from the model,
    returning { "messages": [...], "tool_calls": [...] }
    so that 'messages' are valid chat messages, each with role/content,
    and 'tool_calls' are the parser outputs that might lack 'role'.
    """
    model_name = config["configurable"].get("model_name", "gpt-4o")
    max_tokens = config["configurable"].get("max_tokens", 512)
    verbose = config["configurable"].get("verbose", False)

    # Build final prompt using the 'messages' from the state
    chat_val = ACTOR_PROMPT.format_prompt(messages=inputs["messages"])
    resp = client.chat.completions.create(
        model=model_name,
        messages=chat_val.to_messages(),
        n=1,
        max_tokens=max_tokens,
        stream=False
    )

    usage = getattr(resp, "usage", None)
    if usage and "total_tokens_used" in st.session_state:
        st.session_state["total_tokens_used"] += usage.total_tokens

    if not resp or not resp.choices:
        # fallback
        return {
            "messages": [ {"role": "assistant", "content": "No response"} ],
            "tool_calls": []
        }

    # 'ai_msg' is what we push into messages
    ai_msg = resp.choices[0].message.to_dict()  # {role=assistant, content=...}

    # parse the tool calls
    parsed = parser.invoke(ai_msg)  # might yield [ {"name":..., "type":"tool_call", ...}, ...]

    return {
        "messages": [ai_msg],  # only the single AI message
        "tool_calls": parsed   # store the tool calls separately
    }


# =============================================================================
# We define the tool node with a custom transform
# so that it merges the tool output (the Távily search results) back into the state
# as additional info, or an assistant message, etc.
# =============================================================================
def tool_transform(state: ReflexionState, tool_result: list):
    """
    We have the Távily search output as 'tool_result',
    which is a list of { "query":..., "results":... }.

    We'll wrap it as an assistant message or something so that
    it's all still valid for the next chain call.
    """
    # build a new "assistant" message that summarizes the search results
    # or just store them in the 'tool_calls' if we prefer. But we typically want to
    # add them to 'messages' so the next chain can see them.
    # We'll keep it simple: add a single assistant message with the JSON content.

    text_summary = f"Tool Results: {json.dumps(tool_result, indent=2)}"
    new_msg = {
        "role": "assistant",
        "content": text_summary
    }

    return {
        "messages": state["messages"] + [new_msg],
        "tool_calls": state["tool_calls"]
    }

tool_node = ToolNode(
    tools=[reflexion_search],
    # The transform merges the Távily result into the same shape
    transform=tool_transform
)


# =============================================================================
# We'll define "revision_chain" = reflexion_chain for second pass
# =============================================================================
revision_chain = reflexion_chain


# =============================================================================
# BUILD THE GRAPH
# =============================================================================
builder = StateGraph(ReflexionState)
builder.add_node("draft", reflexion_chain)
builder.add_node("tool", tool_node)
builder.add_node("revise", revision_chain)

builder.add_edge(START, "draft")
builder.add_edge("draft", "tool")
builder.add_edge("tool", "revise")

MAX_ITERATIONS = 3

def reflexion_loop(state: ReflexionState) -> str:
    # We'll see how many expansions we've done by counting 'tool_call'
    expansions = 0
    for tc in state.get("tool_calls", []):
        if tc.get("type") == "tool_call":
            expansions += 1
    if expansions >= MAX_ITERATIONS:
        return END
    return "tool"

builder.add_conditional_edges("revise", reflexion_loop, ["tool", END])
reflexion_graph = builder.compile()


# =============================================================================
# For user code
# =============================================================================
def get_langchain_agent(model_choice: str, system_prompt: str, verbose: bool):
    """
    Re-implements the get_langchain_agent but now as a Reflexion agent, 
    using the same signature so we do not break the existing code or API.
    """
    # We ignore system_prompt or incorporate it if we want
    return reflexion_graph


def main():
    st.title("Reflexion Agent with Távily Search + Azure + Token Counting")

    # initialize session state
    if "messages" not in st.session_state:
        st.session_state["messages"] = [
            {"role": "assistant", "content": "Hello! I'm a Reflexion-based AI. How can I help?"}
        ]
    if "tool_calls" not in st.session_state:
        st.session_state["tool_calls"] = []
    if "total_tokens_used" not in st.session_state:
        st.session_state["total_tokens_used"] = 0

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

    # display existing conversation
    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    if prompt := st.chat_input("Ask a question..."):
        st.session_state["messages"].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        with st.chat_message("assistant"):
            msg_placeholder = st.empty()
            # Build the reflexion state
            reflexion_state: ReflexionState = {
                "messages": st.session_state["messages"],
                "tool_calls": st.session_state["tool_calls"]
            }

            # run the graph
            agent = get_langchain_agent(model_choice, system_prompt, verbosity_enabled)
            config = {
                "configurable": {
                    "model_name": model_choice,
                    "verbose": verbosity_enabled,
                    "max_tokens": max_tokens
                }
            }

            response_text = ""
            try:
                # .stream(..., stream_mode="values") returns step outputs
                for step_output in agent.stream(reflexion_state, stream_mode="values", config=config):
                    node_name, node_data = next(iter(step_output.items()))
                    reflexion_state = node_data  # updated state

                # find the last assistant message
                final_msg = ""
                for msg_ in reversed(reflexion_state["messages"]):
                    if msg_.get("role") == "assistant":
                        final_msg = msg_.get("content", "")
                        break

                final_msg = re.sub(r'[ \t]+$', '', final_msg, flags=re.MULTILINE)
                final_msg = re.sub(r'^\s*\n', '', final_msg)
                final_msg = re.sub(r'\n\s*$', '', final_msg)

                response_text = final_msg
                msg_placeholder.write(final_msg)
            except Exception as e:
                st.error(f"Error: {e}")
                response_text = "I encountered an error. Please try again."

        # update session state
        st.session_state["messages"].append({"role": "assistant", "content": response_text})
        st.session_state["tool_calls"] = reflexion_state["tool_calls"]


if __name__ == "__main__":
    main()