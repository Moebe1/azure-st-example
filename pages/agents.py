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
# We are using exactly the pattern from your example code:
AZURE_OPENAI_API_KEY = st.secrets["AZURE_OPENAI_API_KEY"]
AZURE_OPENAI_ENDPOINT = st.secrets["AZURE_OPENAI_ENDPOINT"]
AZURE_OPENAI_API_VERSION = st.secrets["AZURE_OPENAI_API_VERSION"]

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

# -------------------------------------------------------------------------
# Tavily: optional for search
# -------------------------------------------------------------------------
tavily_api_key = st.secrets.get("TAVILY_API_KEY", "")
search_api = search_api = TavilySearchAPIWrapper()
tavily_tool = TavilySearchResults(api_wrapper=search_api, max_results=5)

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
        messages: list[dict],   # We'll store them as a list of dict for consistency with Azure usage
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

        # Update node & ancestors with reflection score
        self.backpropagate(self.reflection.normalized_score())

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
        # Prefer terminal & solved nodes
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

def select_best_leaf(root: Node) -> Node:
    """Starting from root, pick the highest UCT child until a leaf."""
    node = root
    while node.children:
        node = max(node.children, key=lambda c: c.upper_confidence_bound())
    return node

# We'll store the root node + user input in a dict
class TreeState(dict):
    pass


# -------------------------------------------------------------------------
# Azure LLM Helpers
# -------------------------------------------------------------------------
def azure_chat(messages: list[dict], model_name: str, stream=False, n=1):
    """
    Minimal wrapper that calls the Azure Chat Completion API.
    - messages: list of dicts e.g. [{"role": "user", "content": "Hello"}]
    - model_name: typically "gpt-4o" or "o1-mini" in your case
    - stream: whether to use streaming
    - n: number of completions to generate
    Returns a list of 'choices' in the standard openai format.
    """
    try:
        resp = client.chat.completions.create(
            model=model_name,
            messages=messages,
            stream=stream,
            n=n
        )
        return resp
    except OpenAIError as e:
        st.error(f"OpenAI API Error: {str(e)}")
        return None

def parse_tavily_tool_calls(assistant_msg: dict):
    """
    If the assistant message includes a function call to Távily, parse it out 
    and return a list of tool calls: [{"name":..., "args":...}, ...]
    For simplicity, we check assistant_msg["function_call"] if present.
    Or parse the 'content' if it includes structured data.
    """
    # This is a placeholder; adapt logic for your actual usage.
    tool_calls = []
    if "function_call" in assistant_msg:
        fc = assistant_msg["function_call"]
        if fc.get("name") == "tavily_search_results_json":
            tool_calls.append({"name": "tavily_search_results_json", "args": fc.get("arguments", {})})
    return tool_calls

def run_tavily_tools(tool_calls: list[dict]):
    """Calls TávilySearchResults for each tool call. Returns list of new messages."""
    out_msgs = []
    for tc in tool_calls:
        # Using the TávilySearchResults tool
        results = tavily_tool._run(tc["args"]["query"])
        # We'll store the result in a new message with role=tool for convenience
        out_msgs.append(
            {
                "role": "tool",
                "content": results
            }
        )
    return out_msgs


# -------------------------------------------------------------------------
# Reflection Step: Score or critique a candidate
# -------------------------------------------------------------------------
def reflection_step(user_input: str, candidate_msgs: list[dict], model_name: str) -> Reflection:
    """
    Calls Azure to reflect on the candidate. Return a Reflection object.
    In your approach, you can define any reflection prompt you like.
    """
    # We'll define a simple reflection system prompt
    reflection_prompt = [
        {
            "role": "system",
            "content": "Reflect and grade the assistant's response. Provide a score (0-10) and whether it solved the query."
        },
        {
            "role": "user",
            "content": f"User asked: {user_input}\nAssistant response:\n{candidate_msgs[-1].get('content', '')}"
        }
    ]

    resp = azure_chat(reflection_prompt, model_name=model_name)
    if not resp or not resp.choices:
        return Reflection(reflections="No reflection", score=0, found_solution=False)

    # We'll do a naive parse: look for "Score: x" (0-10) and "Solution: True/False"
    text = resp.choices[0].message.content or ""
    # Extract score
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

    reflection = Reflection(
        reflections=text.strip(),
        score=score_val,
        found_solution=found_solution
    )
    return reflection


# -------------------------------------------------------------------------
# Step: Generate the initial candidate (root node)
# -------------------------------------------------------------------------
def generate_initial_response(state: TreeState) -> TreeState:
    user_input = state["input"]
    model_name = state["model_name"]

    # system message for context
    messages = [
        {
            "role": "system",
            "content": "You are an AI assistant capable of step-by-step reasoning. Provide a helpful response."
        }
    ]
    # user message
    messages.append({"role": "user", "content": user_input})

    # 1. Call Azure Chat for initial candidate
    resp = azure_chat(messages, model_name=model_name, stream=False, n=1)
    if not resp or not resp.choices:
        # Fallback
        reflection = Reflection(reflections="No initial answer", score=0, found_solution=False)
        state["root"] = Node(messages=[], reflection=reflection, parent=None)
        return state

    ai_msg = resp.choices[0].message.to_dict()  # Convert to a dict: {role=..., content=..., function_call=...}

    # 2. Parse tool usage if any
    tool_calls = parse_tavily_tool_calls(ai_msg)
    tool_msgs = run_tavily_tools(tool_calls)

    final_msgs = messages + [ai_msg] + tool_msgs

    # 3. Reflection
    reflection = reflection_step(user_input, final_msgs, model_name)

    root_node = Node(final_msgs, reflection=reflection, parent=None)
    state["root"] = root_node
    return state


# -------------------------------------------------------------------------
# Step: Expand. Generate N new candidates from the best leaf.
# -------------------------------------------------------------------------
def generate_candidates(user_input: str, prefix_msgs: list[dict], model_name: str, N=5):
    """
    Calls Azure to produce N candidate expansions. Return a list of message dicts.
    """
    # We'll just do the same system+user format, but add prefix_msgs as assistant context
    # so the model can continue from that state.
    # Some might prefer a separate "role=system" specifying "Propose next steps" etc.
    system_prompt = {
        "role": "system",
        "content": "Continue reasoning from the conversation so far. Provide the next response."
    }
    # We'll treat prefix_msgs as conversation context.
    conv = [system_prompt] + prefix_msgs
    resp = azure_chat(conv, model_name=model_name, stream=False, n=N)
    if not resp or not resp.choices:
        return []

    return [choice.message.to_dict() for choice in resp.choices]


def expand(state: TreeState) -> TreeState:
    root = state["root"]
    user_input = state["input"]
    model_name = state["model_name"]

    # 1) select best leaf
    leaf = select_best_leaf(root)
    trajectory = leaf.messages  # all messages up to now

    # 2) generate N expansions
    expansions = generate_candidates(user_input, trajectory, model_name, N=5)
    child_nodes = []
    for e_msg in expansions:
        # parse tool calls
        tool_calls = parse_tavily_tool_calls(e_msg)
        tool_msgs = run_tavily_tools(tool_calls)

        cand_msgs = trajectory + [e_msg] + tool_msgs
        reflect = reflection_step(user_input, cand_msgs, model_name)
        node = Node(cand_msgs, reflection=reflect, parent=leaf)
        leaf.children.append(node)
        child_nodes.append(node)

    return state


# -------------------------------------------------------------------------
# Step: Should we keep searching or stop?
# -------------------------------------------------------------------------
def should_loop(state: TreeState) -> Literal["expand", "END"]:
    root = state["root"]
    # If solved or depth>5, end
    if root.is_solved or root.height >= 5:
        return "END"
    return "expand"


# -------------------------------------------------------------------------
# Build a minimal LATS-like state graph
# -------------------------------------------------------------------------
def run_lats(state: TreeState):
    """
    Minimal driver function:
    - start => generate_initial_response => check loop => expand => check loop => expand => ...
    - stops when solved or depth limit reached
    """
    # start
    state = generate_initial_response(state)
    step_count = 0
    while True:
        choice = should_loop(state)
        if choice == "END":
            break
        # else expand
        state = expand(state)
        step_count += 1
        if step_count > 10:
            # safety
            break
    return state


# -------------------------------------------------------------------------
# Streamlit Integration
# -------------------------------------------------------------------------
def main():
    st.set_page_config(page_title="Azure OpenAI LATS", page_icon="🧠")
    st.title("LATS with Azure OpenAI (no standard OPENAI_API_KEY needed)")

    # Chat-like conversation
    if "messages" not in st.session_state:
        st.session_state["messages"] = [
            {"role": "assistant", "content": "Hello! I'm an Azure-based AI. How can I help?"}
        ]

    with st.sidebar:
        st.header("Configuration")
        model_choice = st.selectbox("Azure Model Deployment:", AVAILABLE_MODELS, index=0)
        streaming = st.checkbox("Enable Streaming", value=False)
        st.markdown("---")
        if st.button("Clear Conversation"):
            st.session_state["messages"] = [
                {"role": "assistant", "content": "Conversation cleared. How can I help you now?"}
            ]

    # Display conversation
    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    if prompt := st.chat_input("Ask a question..."):
        # Add user message
        st.session_state["messages"].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        # Container for the final LATS result
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            # Build a LATS state
            state = TreeState(input=prompt, model_name=model_choice)

            with st.spinner("Thinking with LATS..."):
                try:
                    final_state = run_lats(state)
                    # best solution
                    best_solution = final_state["root"].get_best_solution()
                    # The last AI message in that trajectory
                    final_msg = ""
                    for msg in reversed(best_solution.messages):
                        if msg["role"] == "assistant":
                            final_msg = msg["content"]
                            break

                    # Cleanup whitespace
                    final_msg = re.sub(r'[ \t]+$', '', final_msg, flags=re.MULTILINE)
                    final_msg = re.sub(r'^\s*\n', '', final_msg)
                    final_msg = re.sub(r'\n\s*$', '', final_msg)

                    response_placeholder.write(final_msg)
                    st.session_state["messages"].append({"role": "assistant", "content": final_msg})
                except Exception as e:
                    st.error(f"Error during LATS: {e}")
                    st.session_state["messages"].append(
                        {"role": "assistant", "content": "Oops, something went wrong with LATS."}
                    )

if __name__ == "__main__":
    main()