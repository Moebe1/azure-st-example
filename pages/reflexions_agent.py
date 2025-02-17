import streamlit as st
import os
import math
import re
from typing import Optional, Literal, List, Dict, Any
import json

############################################################
# REFLEXION AGENT CODE (LangGraph), NO transform= param
# We use a separate "merge" node to handle the tool output.
############################################################

# =============================================================================
# Streamlit Imports & Setup
# =============================================================================
st.set_page_config(page_title="Reflexion Agent", page_icon="🤖")
from openai import AzureOpenAI, OpenAIError

# Távily imports
from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper
from langchain_community.tools.tavily_search import TavilySearchResults

# LangGraph
from langgraph.graph import StateGraph, START, END, Node
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

def run_tavily(queries: list[str]) -> list[dict]:
    """Run Távily search for each query in 'queries', returning a list of results."""
    outputs = []
    for q in queries:
        r = tavily_tool._run(q)
        outputs.append({"query": q, "results": r})
    return outputs

class ReflexionToolInput(BaseModel):
    search_queries: List[str] = Field(description="Search queries to improve the answer")

@StructuredTool.from_function
def reflexion_search(search_queries: List[str]) -> list[dict]:
    """
    Távily tool function that returns a list of { query, results }
    """
    return run_tavily(search_queries)

# =============================================================================
# Reflexion Actor data
# =============================================================================
class ReflectionCritique(BaseModel):
    missing: str
    superfluous: str

class ReflexionAnswer(BaseModel):
    answer: str
    reflection: ReflectionCritique
    search_queries: List[str]

SYSTEM_INSTRUCTIONS = """You are a 'Reflexion' AI assistant specialized in step-by-step refinement.
1. Provide a ~250 word answer to the user's question.
2. Provide reflection about missing info or superfluous content.
3. Provide 1-3 search queries to help refine or expand if needed.
Return valid JSON for the ReflexionAnswer model.
"""

ACTOR_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_INSTRUCTIONS),
        MessagesPlaceholder(variable_name="messages"),
        ("user", "Now reflect and respond using the ReflexionAnswer function in valid JSON.")
    ]
)

parser = JsonOutputToolsParser(return_id=True)

class ReflexionState(TypedDict):
    messages: List[dict]   # valid chat messages
    tool_calls: List[dict] # parser outputs (lack 'role')

# =============================================================================
# reflexion_chain
# =============================================================================
@chain_runnable
def reflexion_chain(inputs: Dict, config: RunnableConfig):
    """
    We pass 'messages' to Azure,
    parse the AI's JSON => store in 'tool_calls'
    """
    model_name = config["configurable"].get("model_name", "gpt-4o")
    max_tokens = config["configurable"].get("max_tokens", 512)
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

    ai_msg = resp.choices[0].message.to_dict()  # {role=assistant, content=...}
    parsed = parser.invoke(ai_msg)  # the tool calls

    return {
        "messages": [ai_msg],
        "tool_calls": parsed
    }

# =============================================================================
# Tool Node (no transform param!)
# We'll rely on the default behavior, which returns a dict like:
# { "messages": inputs["messages"], "tool_result": <your tool output> }
# We'll have to merge that in a separate node.
# =============================================================================
tool_node = ToolNode(tools=[reflexion_search])

# =============================================================================
# We define a "merge" node to wrap tool output in an assistant message
# =============================================================================
class MergeNode(Node):
    def run_node(self, state: ReflexionState, config: dict):
        """
        The default tool_node might produce something like:
          { "messages": <unchanged msg>, "tool_result": [ ... ] }
        We'll convert that tool_result into an assistant message, then
        combine with existing 'messages'.
        """
        prev_messages = state["messages"]
        tool_result = state.get("tool_result", None)
        tool_calls = state.get("tool_calls", [])

        if tool_result is None:
            # no new results, so return as-is
            return {
                "messages": prev_messages,
                "tool_calls": tool_calls
            }

        # wrap tool_result in an assistant message
        text_summary = "Tool Results:\n" + json.dumps(tool_result, indent=2)
        new_msg = {
            "role": "assistant",
            "content": text_summary
        }

        updated_msgs = prev_messages + [new_msg]
        return {
            "messages": updated_msgs,
            "tool_calls": tool_calls
        }

merge_node = MergeNode("merge_node")

# We'll define the revision chain the same as reflexion_chain
revision_chain = reflexion_chain

# =============================================================================
# BUILD THE GRAPH
# draft -> tool -> merge -> revise -> possibly loop or end
# =============================================================================
builder = StateGraph(ReflexionState)
builder.add_node("draft", reflexion_chain)
builder.add_node("tool", tool_node)
builder.add_node("merge", merge_node)
builder.add_node("revise", revision_chain)

builder.add_edge(START, "draft")
builder.add_edge("draft", "tool")
builder.add_edge("tool", "merge")
builder.add_edge("merge", "revise")

MAX_ITERATIONS = 3

def reflexion_loop(state: ReflexionState) -> str:
    expansions = 0
    # each time the model returns an item with "type": "tool_call", increment expansions
    for tc in state.get("tool_calls", []):
        if tc.get("type") == "tool_call":
            expansions += 1
    if expansions >= MAX_ITERATIONS:
        return END
    return "tool"

builder.add_conditional_edges("revise", reflexion_loop, ["tool", END])
reflexion_graph = builder.compile()

# =============================================================================
# The "get_langchain_agent" for code-compatibility
# =============================================================================
def get_langchain_agent(model_choice: str, system_prompt: str, verbose: bool):
    # ignoring system_prompt or incorporate it if you want
    return reflexion_graph

# =============================================================================
# The main streamlit app
# =============================================================================
def main():
    st.title("Reflexion Agent (No transform param)")

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
                }
            }

            final_answer = ""
            try:
                # run the graph with .stream(..., stream_mode="values")
                for step_output in agent.stream(reflexion_state, stream_mode="values", config=config):
                    node_name, node_data = next(iter(step_output.items()))
                    reflexion_state = node_data  # updated state

                # read last assistant message
                for msg_ in reversed(reflexion_state["messages"]):
                    if msg_.get("role") == "assistant":
                        final_answer = msg_.get("content", "")
                        break

                # clean whitespace
                final_answer = re.sub(r'[ \t]+$', '', final_answer, flags=re.MULTILINE)
                final_answer = re.sub(r'^\s*\n', '', final_answer)
                final_answer = re.sub(r'\n\s*$', '', final_answer)
                msg_placeholder.write(final_answer)

            except Exception as e:
                st.error(f"Error: {e}")
                final_answer = "I encountered an error, sorry."

        st.session_state["messages"].append({"role": "assistant", "content": final_answer})
        st.session_state["tool_calls"] = reflexion_state["tool_calls"]


if __name__ == "__main__":
    main()