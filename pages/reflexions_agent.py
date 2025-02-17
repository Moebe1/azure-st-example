import streamlit as st
import os
import re
import json
from typing import Dict, List, TypedDict

# Streamlit page config
st.set_page_config(page_title="Reflexion Agent", page_icon="🤖")

# Azure OpenAI
from openai import AzureOpenAI
from openai import OpenAIError

# Távily
from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper
from langchain_community.tools.tavily_search import TavilySearchResults

# LangGraph
from langgraph.graph import StateGraph, START, END
from langchain_core.runnables import chain as chain_runnable, RunnableConfig
from langchain_core.messages import AIMessage
from langchain_core.prompt_values import ChatPromptValue
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers.openai_tools import JsonOutputToolsParser

# ==============================
# Config & Távily Setup
# ==============================
AZURE_OPENAI_API_KEY = st.secrets.get("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = st.secrets.get("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_VERSION = st.secrets.get("AZURE_OPENAI_API_VERSION")
TAVILY_API_KEY = st.secrets.get("TAVILY_API_KEY", "")

os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY

AVAILABLE_MODELS = [
    "o1-mini",
    "gpt-4o",
    "gpt-4o-mini"
]

client = AzureOpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_version=AZURE_OPENAI_API_VERSION
)

search_api = TavilySearchAPIWrapper()
tavily_tool = TavilySearchResults(api_wrapper=search_api)
tavily_tool.max_results = 5

def run_tavily(queries: List[str]) -> List[Dict]:
    """Run Távily for each query, returning {query, results} objects."""
    outputs = []
    for q in queries:
        r = tavily_tool._run(q)
        outputs.append({"query": q, "results": r})
    return outputs


# ==============================
# Our reflexion state shape
# ==============================
class ReflexionState(TypedDict):
    messages: List[dict]   # each item: {"role": "...", "content": "..."}
    tool_calls: List[dict] # parser outputs or similar data lacking "role"


# ==============================
# The actor prompt
# ==============================
SYSTEM_INSTRUCTIONS = """You are a 'Reflexion' AI assistant specialized in step-by-step refinement.
1. Provide a ~250 word answer to the user's question.
2. Provide reflection about missing info or superfluous content.
3. Provide 1-3 search queries to help refine or expand if needed.
Return valid JSON for the 'ReflexionAnswer' model.
"""

REFLEXION_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_INSTRUCTIONS),
        MessagesPlaceholder(variable_name="messages"),
        (
            "user",
            "Now reflect and respond using the ReflexionAnswer function in valid JSON."
        )
    ]
)

json_parser = JsonOutputToolsParser(return_id=True)


# ==============================
# Node 1: reflexion_chain
# ==============================
@chain_runnable
def reflexion_chain(state: Dict, config: RunnableConfig):
    import logging
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.debug(f"reflexion_chain - state: {state}")
    logging.debug(f"reflexion_chain - config: {config}")
    """
    Calls Azure Chat. 
    Returns { 'messages': [assistant_msg], 'tool_calls': [parsed items] }
    """
    model_name = config["configurable"].get("model_name", "gpt-4o")
    max_tokens = config["configurable"].get("max_tokens", 512)

    # build the chat prompt
    prompt_val = REFLEXION_PROMPT.format_prompt(messages=state["messages"])

    # call azure
    resp = client.chat.completions.create(
        model=model_name,
        messages=prompt_val.to_messages(),
        stream=False,
        n=1,
        max_tokens=max_tokens
    )

    usage = getattr(resp, "usage", None)
    if usage and "total_tokens_used" in st.session_state:
        st.session_state["total_tokens_used"] += usage.total_tokens

    if not resp or not resp.choices:
        return {
            "messages": [{"role": "assistant", "content": "No response"}],
            "tool_calls": []
        }

    ai_msg = resp.choices[0].message.to_dict()  # { role='assistant', content='...' }
    parsed = json_parser.invoke(ai_msg)        # tool_call(s) if any

    return {
        "messages": [ai_msg],
        "tool_calls": parsed
    }


# ==============================
# Node 2: tool_chain
# ==============================
@chain_runnable
def tool_chain(state: Dict, config: RunnableConfig):
    import logging
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.debug(f"tool_chain - state: {state}")
    logging.debug(f"tool_chain - config: {config}")
    """
    If there's a 'type': 'tool_call' with 'search_queries',
    run Távily, and append the result as an assistant message.
    """
    new_messages = state["messages"]
    new_tool_calls = state["tool_calls"]

    # look for the first tool_call
    for item in new_tool_calls:
        if item.get("type") == "tool_call":
            # typically item["args"] might hold 'search_queries'
            args = item.get("args", {})
            queries = args.get("search_queries", [])
            if isinstance(queries, list) and queries:
                result = run_tavily(queries)
                # create a new assistant message summarizing
                text_summary = "Tool Results:\n" + json.dumps(result, indent=2)
                new_msg = {
                    "role": "assistant",
                    "content": text_summary
                }
                new_messages = new_messages + [new_msg]
            break

    return {
        "messages": new_messages,
        "tool_calls": new_tool_calls
    }


# We'll define revision_chain same as reflexion_chain
revision_chain = reflexion_chain

# ==============================
# Building the Graph
# ==============================
from langgraph.graph import StateGraph, START, END

builder = StateGraph(ReflexionState)
builder.add_node("draft", reflexion_chain)
builder.add_node("tool", tool_chain)
builder.add_node("revise", revision_chain)

builder.add_edge(START, "draft")
builder.add_edge("draft", "tool")
builder.add_edge("tool", "revise")

MAX_ITERATIONS = 3

def reflexion_loop(state: ReflexionState) -> str:
    import logging
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.debug(f"reflexion_loop - state: {state}")
    expansions = 0
    for tc in state.get("tool_calls", []):
        if tc.get("type") == "tool_call":
            expansions += 1
    if expansions >= MAX_ITERATIONS:
        return END
    return "tool"

builder.add_conditional_edges("revise", reflexion_loop, ["tool", END])

reflexion_graph = builder.compile()


# ==============================
# Externally, the same signature
# ==============================
def get_langchain_agent(model_choice: str, system_prompt: str, verbose: bool):
    # ignoring system_prompt or incorporate it if you want
    return reflexion_graph


# ==============================
# The main streamlit code
# ==============================
def main():
    st.title("Reflexion Agent - TypeError fix")

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
        max_tokens = st.number_input("Max Tokens per Response", min_value=50, max_value=4096, value=500)
        st.write("**Total Tokens Used**:", st.session_state["total_tokens_used"])

        if st.button("Clear Conversation"):
            st.session_state["messages"] = [
                {"role": "assistant", "content": "Conversation cleared. How can I help now?"}
            ]
            st.session_state["tool_calls"] = []
            st.session_state["total_tokens_used"] = 0
            st.experimental_rerun()

    # Display existing conversation
    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    if prompt := st.chat_input("Ask a question..."):
        # append new user message
        st.session_state["messages"].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        with st.chat_message("assistant"):
            msg_placeholder = st.empty()

            # Build the reflexion_state
            reflexion_state: ReflexionState = {
                "messages": st.session_state["messages"],
                "tool_calls": st.session_state["tool_calls"]
            }

            agent = get_langchain_agent(model_choice, system_prompt, verbose=False)
            config = {
                "configurable": {
                    "model_name": model_choice,
                    "max_tokens": max_tokens
                }
            }

            final_answer = ""
            try:
                for step_output in agent.stream(reflexion_state, stream_mode="values", config=config):
                    # step_output is a dict like {"draft": {messages:..., tool_calls:...}}
                    node_name, node_data = next(iter(step_output.items()))
                    reflexion_state = node_data  # updated

                # find last 'assistant' message
                for m in reversed(reflexion_state["messages"]):
                    if m.get("role") == "assistant":
                        final_answer = m["content"]
                        break

                # cleanup
                final_answer = re.sub(r'[ \t]+$', '', final_answer, flags=re.MULTILINE)
                final_answer = re.sub(r'^\s*\n', '', final_answer)
                final_answer = re.sub(r'\n\s*$', '', final_answer)
                msg_placeholder.write(final_answer)
            except Exception as e:
                st.error(f"Error: {e}")
                final_answer = "I encountered an error."

        st.session_state["messages"].append({"role": "assistant", "content": final_answer})
        st.session_state["tool_calls"] = reflexion_state["tool_calls"]


if __name__ == "__main__":
    main()