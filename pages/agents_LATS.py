import os
import math
import re
from typing import Optional, Literal
from collections import deque, defaultdict

import streamlit as st
from openai import AzureOpenAI, OpenAIError

# Tavily tool
from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper
from langchain_community.tools.tavily_search import TavilySearchResults

# -------------------------------------------------------------------------
# Azure OpenAI Configuration
# -------------------------------------------------------------------------
AZURE_OPENAI_API_KEY = st.secrets.get("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = st.secrets.get("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_VERSION = st.secrets.get("AZURE_OPENAI_API_VERSION")

AVAILABLE_MODELS = [
    "o1-mini",
    "gpt-4o",
    "gpt-4o-mini"
]

# Create Azure OpenAI client
client = AzureOpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_version=AZURE_OPENAI_API_VERSION
)

# -------------------------------------------------------------------------
# Tavily: optional for search
# -------------------------------------------------------------------------
tavily_api_key = st.secrets.get("TAVILY_API_KEY", "")
os.environ["TAVILY_API_KEY"] = tavily_api_key  # TávilySearchAPIWrapper picks it from environment
search_api = TavilySearchAPIWrapper()
tavily_tool = TavilySearchResults(api_wrapper=search_api)
tavily_tool.max_results = 5  # If allowed in your Tavily version

# -------------------------------------------------------------------------
# LATS Data Structures (Reflection, Node, TreeState)
# -------------------------------------------------------------------------
from pydantic import BaseModel, Field

class Reflection(BaseModel):
    reflections: str
    score: int = Field(ge=0, le=10)
    found_solution: bool

    def normalized_score(self) -> float:
        return self.score / 10.0


class Node:
    def __init__(
        self,
        messages: list[dict],
        reflection: Reflection,
        parent: Optional["Node"] = None,
    ):
        self.messages = messages
        self.reflection = reflection
        self.parent = parent
        self.children: list["Node"] = []

        self.value = 0.0
        self.visits = 0
        self.depth = parent.depth + 1 if parent else 1
        self._is_solved = reflection.found_solution

        if self._is_solved:
            self._mark_tree_as_solved()

        self.backpropagate(reflection.normalized_score())

    def __repr__(self):
        return f"<Node depth={self.depth}, value={self.value:.2f}, visits={self.visits}, solved={self._is_solved}/>"

    @property
    def is_solved(self) -> bool:
        return self._is_solved

    @property
    def is_terminal(self) -> bool:
        return len(self.children) == 0

    @property
    def height(self) -> int:
        """Max depth below this node."""
        if self.children:
            return 1 + max(child.height for child in self.children)
        return 1

    def _mark_tree_as_solved(self):
        node = self
        while node:
            node._is_solved = True
            node = node.parent

    def backpropagate(self, reward: float):
        node = self
        while node:
            node.visits += 1
            node.value = (node.value * (node.visits - 1) + reward) / node.visits
            node = node.parent

    def upper_confidence_bound(self, exploration_weight=1.0):
        if not self.parent:
            return self.value
        if self.visits == 0:
            return self.value
        avg_reward = self.value
        explore = math.sqrt(math.log(self.parent.visits) / self.visits)
        return avg_reward + exploration_weight * explore

    def get_best_solution(self) -> "Node":
        """Return best solution node from this subtree if it exists."""
        candidates = self._collect_descendants()
        # Maximize value if node is terminal & solved
        best = max(
            candidates,
            key=lambda nd: int(nd.is_terminal and nd.is_solved) * nd.value,
        )
        return best

    def _collect_descendants(self) -> list["Node"]:
        stack = [self]
        out = []
        while stack:
            node = stack.pop()
            out.append(node)
            if node.children:
                stack.extend(node.children)
        return out


class TreeState(dict):
    """
    e.g. {
      "input": "user question",
      "model_name": "gpt-4o",
      "verbose": True/False,
      "max_tokens": 500,   # or some limit
      "root": <Node>
    }
    """


# -------------------------------------------------------------------------
# Azure Chat + usage logging + max token param
# -------------------------------------------------------------------------
def azure_chat(
    messages: list[dict],
    model_name: str,
    stream=False,
    n=1,
    verbose=False,
    max_tokens=500
):
    """
    Minimal wrapper that calls the Azure Chat Completion API and logs usage.
    We also cap the generation with 'max_tokens'. 
    """
    if verbose:
        st.write(f"**Azure Chat Request** => model={model_name}, n={n}, stream={stream}, max_tokens={max_tokens}")
        st.write("Messages:")
        for idx, msg in enumerate(messages):
            st.write(f"{idx}. {msg['role'].upper()}: {msg.get('content', '')}")

    try:
        resp = client.chat.completions.create(
            model=model_name,
            messages=messages,
            stream=stream,
            n=n,
            max_tokens=max_tokens
        )
        usage = getattr(resp, "usage", None)
        if usage:
            prompt_toks = getattr(usage, "prompt_tokens", 0) or 0
            completion_toks = getattr(usage, "completion_tokens", 0) or 0
            total_toks = getattr(usage, "total_tokens", 0) or (prompt_toks + completion_toks)
            if "total_tokens_used" not in st.session_state:
                st.session_state["total_tokens_used"] = 0
            st.session_state["total_tokens_used"] += total_toks
            if verbose:
                st.write(f"**Token Usage**: Prompt={prompt_toks}, Completion={completion_toks}, Total={total_toks}")
                st.write(f"Cumulative: {st.session_state['total_tokens_used']}")
        return resp
    except OpenAIError as e:
        st.error(f"OpenAI API Error: {str(e)}")
        return None


def parse_tavily_tool_calls(assistant_msg: dict):
    tool_calls = []
    if "function_call" in assistant_msg:
        fc = assistant_msg["function_call"]
        if fc.get("name") == "tavily_search_results_json":
            args = fc.get("arguments", {})
            tool_calls.append({
                "name": "tavily_search_results_json",
                "args": args
            })
    return tool_calls

def run_tavily_tools(tool_calls: list[dict], verbose=False):
    out_msgs = []
    for tc in tool_calls:
        if verbose:
            st.write(f"**Calling Tavily** with query: {tc['args']}")
        query = tc["args"].get("query", "")
        # The model must specifically call "tavily_search_results_json" to invoke Távily
        results = tavily_tool._run(query)
        out_msgs.append({"role": "tool", "content": results})
    return out_msgs


# -------------------------------------------------------------------------
# Reflection Step: score candidate
# -------------------------------------------------------------------------
def reflection_step(user_input: str, candidate_msgs: list[dict], state: TreeState) -> Reflection:
    model_name = state["model_name"]
    verbose = state.get("verbose", False)
    max_tokens = state.get("max_tokens", 500)

    reflection_prompt = [
        {
            "role": "system",
            "content": "Reflect and grade the assistant's response. Provide a score (0-10) and whether it solved the query."
        },
        {
            "role": "user",
            "content": (
                f"User asked: {user_input}\n"
                f"Assistant response:\n{candidate_msgs[-1].get('content', '')}"
            )
        }
    ]
    resp = azure_chat(
        reflection_prompt,
        model_name=model_name,
        stream=False,
        n=1,
        verbose=verbose,
        max_tokens=max_tokens
    )
    if not resp or not resp.choices:
        return Reflection(reflections="No reflection", score=0, found_solution=False)

    text = resp.choices[0].message.content or ""
    score_match = re.search(r"score\s*[:=]\s*(\d+)", text.lower())
    found_solution = False
    sol_match = re.search(r"(?i)(found_solution|solution|complete)\s*[:=]\s*(true|false)", text)
    if sol_match:
        found_solution = sol_match.group(2).lower() == "true"

    score_val = 0
    if score_match:
        try:
            s = int(score_match.group(1))
            score_val = max(0, min(10, s))
        except ValueError:
            pass

    # If reflection score >= 8, consider it "solved"
    if score_val >= 8:
        found_solution = True

    if verbose:
        st.write(f"**Reflection** => Score={score_val}, FoundSolution={found_solution}")

    return Reflection(
        reflections=text.strip(),
        score=score_val,
        found_solution=found_solution
    )


# -------------------------------------------------------------------------
# LATS Steps
# -------------------------------------------------------------------------
def select_best_leaf(root: Node):
    node = root
    while node.children:
        node = max(node.children, key=lambda c: c.upper_confidence_bound())
    return node

def generate_initial_response(state: TreeState) -> TreeState:
    """
    Produce the initial candidate from user input => root node 
    with reflection.
    """
    user_input = state["input"]
    model_name = state["model_name"]
    verbose = state.get("verbose", False)
    max_tokens = state.get("max_tokens", 500)

    if verbose:
        st.write("**[generate_initial_response]**: Creating root node.")

    messages = [
        {"role": "system", "content": "You are an AI assistant with chain-of-thought."},
        {"role": "user", "content": user_input},
    ]

    resp = azure_chat(messages, model_name=model_name, stream=False, n=1, verbose=verbose, max_tokens=max_tokens)
    if not resp or not resp.choices:
        reflection = Reflection(reflections="No initial answer", score=0, found_solution=False)
        state["root"] = Node([], reflection)
        return state

    ai_msg = resp.choices[0].message.to_dict()
    tool_calls = parse_tavily_tool_calls(ai_msg)
    tool_msgs = run_tavily_tools(tool_calls, verbose=verbose)
    final_msgs = messages + [ai_msg] + tool_msgs

    reflection = reflection_step(user_input, final_msgs, state)
    root = Node(final_msgs, reflection)
    state["root"] = root
    return state

def generate_candidates(leaf_msgs: list[dict], state: TreeState, N=5):
    user_input = state["input"]
    model_name = state["model_name"]
    verbose = state.get("verbose", False)
    max_tokens = state.get("max_tokens", 500)

    if verbose:
        st.write(f"**generate_candidates** => N={N}")

    system_prompt = {"role": "system", "content": "Please continue the conversation step by step."}
    conversation = [system_prompt] + leaf_msgs

    resp = azure_chat(
        conversation,
        model_name=model_name,
        stream=False,
        n=N,
        verbose=verbose,
        max_tokens=max_tokens
    )
    if not resp or not resp.choices:
        return []
    return [c.message.to_dict() for c in resp.choices]

def expand(state: TreeState) -> TreeState:
    root = state["root"]
    leaf = select_best_leaf(root)
    verbose = state.get("verbose", False)

    if verbose:
        st.write(f"**[expand]** => Best leaf so far: {leaf}")

    expansions = generate_candidates(leaf.messages, state, N=5)
    if verbose:
        st.write(f"Generated {len(expansions)} expansions.")

    for e_msg in expansions:
        tool_calls = parse_tavily_tool_calls(e_msg)
        tool_msgs = run_tavily_tools(tool_calls, verbose=verbose)
        cand_msgs = leaf.messages + [e_msg] + tool_msgs
        reflection = reflection_step(state["input"], cand_msgs, state)
        node = Node(cand_msgs, reflection, parent=leaf)
        leaf.children.append(node)

    return state

def should_loop(state: TreeState) -> Literal["expand", "END"]:
    root = state["root"]
    if root.is_solved or root.height >= 5:
        return "END"
    return "expand"

def run_lats(state: TreeState):
    # start
    state = generate_initial_response(state)
    step_count = 0
    while True:
        decision = should_loop(state)
        if decision == "END":
            break
        state = expand(state)
        step_count += 1
        # Extra safeguard
        if step_count > 10:
            break
    return state


# -------------------------------------------------------------------------
# Streamlit UI
# -------------------------------------------------------------------------
def main():
    st.set_page_config(page_title="Azure OpenAI LATS (Token Cap, Távily, Refresh)", page_icon="🧠")
    st.title("LATS with Azure + Tavily (Token Cap)")

    # Chat-like conversation
    if "messages" not in st.session_state:
        st.session_state["messages"] = [
            {"role": "assistant", "content": "Hello! I'm an Azure-based AI. How can I help?"}
        ]

    if "total_tokens_used" not in st.session_state:
        st.session_state["total_tokens_used"] = 0

    with st.sidebar:
        st.header("Configuration")
        model_choice = st.selectbox("Azure Model:", AVAILABLE_MODELS, index=0)
        verbosity_enabled = st.checkbox("Enable Verbose Mode", value=False)
        max_tokens = st.number_input("Max Tokens per Response", min_value=50, max_value=4096, value=500)
        st.write("**Total Tokens Used**:", st.session_state["total_tokens_used"])
        if st.button("Clear Conversation"):
            # Clear the messages & token usage, then rerun
            st.session_state["messages"] = [
                {"role": "assistant", "content": "Conversation cleared. How can I help now?"}
            ]
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

        # Container for final LATS result
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            # Build LATS state
            state = TreeState(
                input=prompt,
                model_name=model_choice,
                verbose=verbosity_enabled,
                max_tokens=max_tokens
            )

            with st.spinner("Thinking with LATS..."):
                try:
                    final_state = run_lats(state)
                    best_solution = final_state["root"].get_best_solution()
                    final_msg = ""
                    # The last AI message in best_solution
                    for msg in reversed(best_solution.messages):
                        if msg["role"] == "assistant":
                            final_msg = msg["content"]
                            break
                    # Clean whitespace
                    final_msg = re.sub(r'[ \t]+$', '', final_msg, flags=re.MULTILINE)
                    final_msg = re.sub(r'^\s*\n', '', final_msg)
                    final_msg = re.sub(r'\n\s*$', '', final_msg)
                    message_placeholder.write(final_msg)
                    st.session_state["messages"].append({"role": "assistant", "content": final_msg})
                except Exception as e:
                    st.error(f"Error during LATS: {e}")
                    st.session_state["messages"].append(
                        {"role": "assistant", "content": "Oops, something went wrong with LATS."}
                    )

if __name__ == "__main__":
    main()