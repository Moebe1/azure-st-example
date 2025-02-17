import os
import math
from collections import deque, defaultdict
from typing import Optional, Literal

import streamlit as st
from getpass import getpass

# ─────────────────────────────────────────────────────────────────────────────
# 1. INSTALL / IMPORT LANGCHAIN + LANGGRAPH + TAVILY
#    Make sure you have them installed:
#    pip install --upgrade langchain langgraph langchain_openai tavily-python
# ─────────────────────────────────────────────────────────────────────────────

# langgraph imports
from langgraph.graph import START, END, StateGraph
from langgraph.prebuilt import ToolNode

# langchain (core) + additional
from langchain_core.output_parsers.openai_tools import (
    JsonOutputToolsParser,
    PydanticToolsParser,
)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.prompt_values import ChatPromptValue
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.runnables import chain as as_runnable, RunnableConfig

# langchain_openai for Azure or normal
from langchain_openai import ChatOpenAI

# Tools: Tavily search
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper

# Pydantic
from pydantic import BaseModel, Field

# =============================================================================
# Configuration - Azure OpenAI + Tavily
# =============================================================================
AZURE_OPENAI_API_KEY = st.secrets.get("AZURE_OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = st.secrets.get("AZURE_OPENAI_ENDPOINT") or os.environ.get("OPENAI_API_ENDPOINT")
AZURE_OPENAI_API_VERSION = st.secrets.get("AZURE_OPENAI_API_VERSION") or os.environ.get("AZURE_OPENAI_API_VERSION")
TAVILY_API_KEY = st.secrets.get("TAVILY_API_KEY") or os.environ.get("TAVILY_API_KEY")

# List of available models (map them to your actual Azure deployments or GPT names)
AVAILABLE_MODELS = [
    "o1-mini",
    "gpt-4o",
    "gpt-4o-mini"
]

# ─────────────────────────────────────────────────────────────────────────────
# 2. Data Structures for LATS (Node, Reflection, TreeState)
# ─────────────────────────────────────────────────────────────────────────────
class Reflection(BaseModel):
    """Critique, reflections, and a numeric score for the candidate response."""
    reflections: str = Field(
        description="Critique, reflections on sufficiency, correctness, etc."
    )
    score: int = Field(
        description="Score from 0-10 on the quality of the candidate response.",
        gte=0,
        lte=10,
    )
    found_solution: bool = Field(
        description="Whether the response has fully solved the question or task."
    )

    def as_message(self) -> BaseMessage:
        """Convert reflection data to a human-style message for the next iteration."""
        return HumanMessage(
            content=f"Reflection:\n{self.reflections}\nScore: {self.score}"
        )

    @property
    def normalized_score(self) -> float:
        return self.score / 10.0


class Node:
    """Each Node in the LATS tree stores messages + reflection data, plus children."""
    def __init__(
        self,
        messages: list[BaseMessage],
        reflection: Reflection,
        parent: Optional["Node"] = None,
    ):
        self.messages = messages
        self.reflection = reflection
        self.parent = parent
        self.children: list["Node"] = []

        self.value = 0.0
        self.visits = 0
        self.depth = parent.depth + 1 if parent is not None else 1
        self._is_solved = reflection.found_solution if reflection else False

        # If the node is solved, propagate 'solved' up the tree
        if self._is_solved:
            self._mark_tree_as_solved()

        # Update the node's value upward
        self.backpropagate(reflection.normalized_score)

    def __repr__(self):
        return f"<Node value={self.value}, visits={self.visits}, reflection={self.reflection}/>"
    
    @property
    def is_solved(self) -> bool:
        return self._is_solved

    @property
    def is_terminal(self) -> bool:
        """No children => a leaf node in the tree."""
        return len(self.children) == 0

    @property
    def best_child_score(self):
        """Return the child with the highest value (preferring solved)."""
        if not self.children:
            return None
        return max(self.children, key=lambda child: int(child.is_solved) * child.value)

    @property
    def height(self) -> int:
        """Max depth below this node (for measuring how 'deep' we rolled out)."""
        if self.children:
            return 1 + max([child.height for child in self.children])
        return 1

    def _mark_tree_as_solved(self):
        """Propagate 'solved' status up the chain so we can stop early if desired."""
        n = self
        while n:
            n._is_solved = True
            n = n.parent

    def backpropagate(self, reward: float):
        """Update the score of this node and all ancestors."""
        node = self
        while node:
            node.visits += 1
            # Weighted average over visits
            node.value = (node.value * (node.visits - 1) + reward) / node.visits
            node = node.parent

    def get_messages(self, include_reflections: bool = True):
        if include_reflections:
            return self.messages + [self.reflection.as_message()]
        return self.messages

    def get_trajectory(self, include_reflections: bool = True) -> list[BaseMessage]:
        """Backtrack up the tree to build a message list describing the path."""
        path = []
        node = self
        while node:
            # Reverse to put reflection after the candidate messages
            segment = node.get_messages(include_reflections=include_reflections)[::-1]
            path.extend(segment)
            node = node.parent
        return path[::-1]

    def get_best_solution(self) -> "Node":
        """Return the best node that is a solution in this subtree."""
        all_descendants = self._get_all_descendants()
        # Include self
        all_nodes = [self] + all_descendants
        # We only rank nodes that are terminal and solved
        best_node = max(
            all_nodes,
            key=lambda n: int(n.is_terminal and n.is_solved) * n.value,
        )
        return best_node

    def _get_all_descendants(self) -> list["Node"]:
        results = []
        queue = deque([self])
        while queue:
            node = queue.popleft()
            results.extend(node.children)
            for child in node.children:
                queue.append(child)
        return results

# A simple typed dictionary for the search state
class TreeState(dict):
    # "root": Node
    # "input": str
    pass


# ─────────────────────────────────────────────────────────────────────────────
# 3. Reflection + Generation + Expansion Runnables
# ─────────────────────────────────────────────────────────────────────────────

# 3A. Reflection Chain
reflection_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "Reflect and grade the assistant response to the user question."),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="candidate"),
    ]
)

# We'll parse the reflection as a "Reflection" Pydantic model
from langchain_core.runnables import chain as chain_runnable
from langchain_core.output_parsers.openai_tools import PydanticToolsParser

reflection_llm = ChatOpenAI(
    azure_deployment=model_choice,
    openai_api_key=AZURE_OPENAI_API_KEY,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    openai_api_version=AZURE_OPENAI_API_VERSION,
        temperature=0.1
    )

reflection_parser = PydanticToolsParser(tools=[Reflection])

reflection_chain = (
    reflection_prompt
    | reflection_llm.bind_tools(tools=[Reflection], tool_choice="Reflection").with_config(run_name="Reflection")
    | reflection_parser
)

@chain_runnable
def reflection_fn(inputs) -> Reflection:
    tool_choices = reflection_chain.invoke(inputs)
    reflection = tool_choices[0]
    # If the final message is not from the AI, it's not a solution
    candidate = inputs["candidate"]
    if len(candidate) == 0 or not isinstance(candidate[-1], AIMessage):
        reflection.found_solution = False
    return reflection

# 3B. Tools: Tavily Search
search = TavilySearchAPIWrapper(api_key=TAVILY_API_KEY, max_results=5)
tavily_tool = TavilySearchResults(api_wrapper=search, max_results=5)
tool_node = ToolNode(tools=[tavily_tool])

# 3C. Prompt to get "initial" or "expanded" answers
#     We'll keep the structure from the notebook example: user + existing messages
init_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are an AI assistant. Additional system prompt: {system_prompt}"),
        ("user", "{user_input}"),
        MessagesPlaceholder(variable_name="messages", optional=True),
    ]
)

gen_parser = JsonOutputToolsParser(return_id=True)

# We'll define a function that calls the chat model to produce N expansions
@chain_runnable
def generate_candidates(inputs, config: RunnableConfig):
    """Generate N candidate next steps from the current node's messages."""
    N = config["configurable"].get("N", 5)  # default 5 expansions
    llm = ChatOpenAI(
        model_name="gpt-4o",
        temperature=0.9,
        max_tokens=1000,
        n=N,
    ).bind_tools(tools=[tavily_tool])

    prompt_val = ChatPromptValue(
        messages=init_prompt.format_prompt(
            system_prompt=inputs["system_prompt"],
            user_input=inputs["user_input"],
            messages=inputs["messages"],
        ).to_messages()
    )
    chat_result = llm.generate([prompt_val.to_messages()])
    # Return multiple candidate AIMessage objects
    return [gen.message for gen in chat_result.generations[0]]

# ─────────────────────────────────────────────────────────────────────────────
# 4. Graph Steps: "start" (generate root) and "expand"
# ─────────────────────────────────────────────────────────────────────────────

def select_best_leaf(root: Node) -> Node:
    """Select the leaf node with highest UCT from root to leaf."""
    # If no children, return root
    if not root.children:
        return root

    node = root
    while node.children:
        node = max(node.children, key=lambda child: child.upper_confidence_bound())
    return node

def generate_initial_response(state: TreeState) -> TreeState:
    """
    Generate the initial candidate from the user input (root node).
    """
    user_input = state["input"]
    system_prompt = state.get("system_prompt", "")

    # Step 1: Produce initial answer
    # Single generation
    llm = ChatOpenAI(
        model_name="gpt-4o",
        temperature=0.9,
        max_tokens=1200,
    ).bind_tools(tools=[tavily_tool])
    prompt_val = init_prompt.format_prompt(
        system_prompt=system_prompt,
        user_input=user_input,
        messages=[],
    )
    result_msg = llm.invoke(prompt_val.to_messages())
    parsed = gen_parser.invoke(result_msg)

    # Step 2: Possibly call the Tavily tool
    tool_resps = []
    if parsed:
        for tcall in parsed:
            tool_resp = tool_node.invoke(
                {
                    "messages": [
                        AIMessage(
                            content="",
                            tool_calls=[
                                {
                                    "name": tcall["type"],
                                    "args": tcall["args"],
                                    "id": tcall["id"],
                                }
                            ],
                        )
                    ]
                }
            )
            tool_resps.append(tool_resp["messages"][0])

    # Combine
    output_messages = [result_msg] + tool_resps

    # Step 3: Reflect
    reflection = reflection_fn.invoke(
        {
            "input": user_input,
            "candidate": output_messages,
        }
    )

    root_node = Node(output_messages, reflection=reflection, parent=None)
    state["root"] = root_node
    return state

def expand(state: TreeState, config: RunnableConfig) -> TreeState:
    """
    From the best leaf, generate N candidate expansions. Score them, attach them.
    """
    root = state["root"]
    best_leaf = select_best_leaf(root)
    # Build the message trajectory
    messages = best_leaf.get_trajectory()
    user_input = state["input"]
    system_prompt = state.get("system_prompt", "")

    # Generate expansions (N candidates)
    expansions = generate_candidates.invoke(
        {
            "system_prompt": system_prompt,
            "user_input": user_input,
            "messages": messages,
        },
        config,
    )

    # For each candidate, parse tool calls + reflect
    child_nodes = []
    for candidate_msg in expansions:
        tool_calls = gen_parser.invoke(candidate_msg)
        tool_output_msgs = []
        if tool_calls:
            for tcall in tool_calls:
                tool_res = tool_node.invoke(
                    {
                        "messages": [
                            AIMessage(
                                content="",
                                tool_calls=[
                                    {
                                        "name": tcall["type"],
                                        "args": tcall["args"],
                                        "id": tcall["id"],
                                    }
                                ],
                            )
                        ]
                    }
                )
                tool_output_msgs.append(tool_res["messages"][0])

        # Combine
        candidate_msgs = [candidate_msg] + tool_output_msgs
        reflection = reflection_fn.invoke(
            {"input": user_input, "candidate": candidate_msgs}
        )
        node = Node(candidate_msgs, reflection=reflection, parent=best_leaf)
        best_leaf.children.append(node)
        child_nodes.append(node)

    return state


def should_loop(state: TreeState) -> Literal["expand", END]:
    """Return 'expand' or END depending on whether we found a solution or reached depth."""
    root = state["root"]
    if root.is_solved:
        return END
    # If we've rolled out to depth > 5, stop
    if root.height >= 5:
        return END
    return "expand"

# Build the LATS graph
from langgraph.graph import StateGraph

builder = StateGraph(TreeState)
builder.add_node("start", generate_initial_response)
builder.add_node("expand", expand)

builder.add_edge(START, "start")
builder.add_conditional_edges(
    "start",
    should_loop,
    ["expand", END],
)
builder.add_conditional_edges(
    "expand",
    should_loop,
    ["expand", END],
)

lats_graph = builder.compile()


# ─────────────────────────────────────────────────────────────────────────────
# 5. Streamlit App
# ─────────────────────────────────────────────────────────────────────────────

def main():
    st.set_page_config(page_title="Agents - LATS", page_icon="🤖")
    st.title("LATS Agents with LangGraph")

    # Initialize session for chat history
    if "messages" not in st.session_state:
        st.session_state["messages"] = [
            {"role": "assistant", "content": "Hello! How can I help you today?"}
        ]

    # Sidebar config
    with st.sidebar:
        st.header("Configuration")
        model_choice = st.selectbox("Select Model:", AVAILABLE_MODELS, index=0)
        default_system_prompt = "You are an AI assistant."
        system_prompt = st.text_area("System Prompt", default_system_prompt)
        streaming_enabled = st.checkbox("Enable Streaming", value=False)
        verbosity_enabled = st.checkbox("Enable Verbose Mode", value=False)
        st.write("---")
        st.write("Tavily API Key:", TAVILY_API_KEY if TAVILY_API_KEY else "Not Set")
        st.write("OpenAI API Key:", AZURE_OPENAI_API_KEY if AZURE_OPENAI_API_KEY else "Not Set")

    # Display conversation
    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    # Chat input
    if user_input := st.chat_input("Ask a question..."):
        st.session_state["messages"].append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.write(user_input)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()

            # We run the LATS Graph using the user's question as 'input'
            # We'll store the final answer in 'response_text'
            response_text = ""
            try:
                # Prepare the LATS state
                initial_state: TreeState = {
                    "input": user_input,
                    "system_prompt": system_prompt,
                }

                # We'll step through the graph. For short queries, it might not expand much.
                last_output = None
                for step in lats_graph.stream(initial_state):
                    # step is a dict, like {"start": <TreeState>} or {"expand": <TreeState>}
                    last_output = step

                    if verbosity_enabled:
                        # Show the intermediate step name + partial state
                        step_name, step_data = next(iter(step.items()))
                        message_placeholder.markdown(
                            f"**Step**: {step_name}, **Tree Depth**: {step_data['root'].height}"
                        )
                # done
                if last_output:
                    # final state's root
                    step_name, final_state = next(iter(last_output.items()))
                    solution_node = final_state["root"].get_best_solution()
                    # The last AIMessage is presumably the final "solution"
                    # We'll not show reflection messages here, just the final text
                    final_trajectory = solution_node.get_trajectory(include_reflections=False)
                    # The last message in that trajectory from the AI
                    # (some steps might have multiple AI messages, but typically the last is the final "answer")
                    for msg in reversed(final_trajectory):
                        if isinstance(msg, AIMessage):
                            response_text = msg.content
                            break
                message_placeholder.write(response_text or "No solution found.")
            except Exception as exc:
                st.error(f"Error: {exc}")
                response_text = (
                    "I encountered an error processing your request. Please try again."
                )

        st.session_state["messages"].append({"role": "assistant", "content": response_text})

if __name__ == "__main__":
    # If not provided via secrets, you can do something like this:
    # if not os.environ.get("TAVILY_API_KEY"):
    #     os.environ["TAVILY_API_KEY"] = getpass("TAVILY_API_KEY")
    # if not os.environ.get("OPENAI_API_KEY"):
    #     os.environ["OPENAI_API_KEY"] = getpass("OPENAI_API_KEY")
    main()