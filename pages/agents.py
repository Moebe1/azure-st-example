import streamlit as st
from langchain_openai import AzureChatOpenAI
from langchain.tools import Tool
from openai import OpenAIError

# Import the new LangGraph API
# This is part of the experimental module in LangChain
from langchain.experimental.langgraph import create_react_agent

# =============================================================================
# Configuration - Azure OpenAI
# =============================================================================
AZURE_OPENAI_API_KEY = st.secrets["AZURE_OPENAI_API_KEY"]
AZURE_OPENAI_ENDPOINT = st.secrets["AZURE_OPENAI_ENDPOINT"]
AZURE_OPENAI_API_VERSION = st.secrets["AZURE_OPENAI_API_VERSION"]

# List of available models
AVAILABLE_MODELS = [
    "o1-mini",
    "gpt-4o",
    "gpt-4o-mini"
]

# =============================================================================
# Function: Initialize LangGraph ReAct Agent
# =============================================================================
def get_langchain_agent(model_choice, system_prompt, verbose):
    """
    Creates a LangGraph-based ReAct agent that uses AzureChatOpenAI and 
    user-defined tools. We incorporate a chain-of-thought style system prompt
    to encourage detailed reasoning, although the actual detail may depend on
    the model and Azure OpenAI policy.
    """
    try:
        # Instantiate the AzureChatOpenAI LLM
        llm = AzureChatOpenAI(
            azure_deployment=model_choice,
            openai_api_key=AZURE_OPENAI_API_KEY,
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            openai_api_version=AZURE_OPENAI_API_VERSION,
            streaming=True if model_choice != "o1-mini" else False,
            temperature=0.9,
            max_tokens=3000
        )

        # Define any tools you want the agent to use
        tools = [
            Tool(
                name="Example Tool",
                func=lambda x: f"Processed: {x}",
                description="An example tool."
            )
        ]

        # Combine the user-provided system prompt with chain-of-thought instructions
        # This becomes the 'system_message' for the new ReAct agent.
        full_system_message = f"""{system_prompt}

You are a helpful AI assistant. 
Please show your detailed reasoning steps in bullet-point form followed by a final answer.
Include:
1) Key questions/concepts
2) Perspectives or facts
3) Examples or analogies
4) A synthesized conclusion

Be sure to reveal each step of your thought process. Do not hide or summarize them away.
"""

        # Create the ReAct agent using LangGraph
        agent_graph = create_react_agent(
            llm=llm,
            tools=tools,
            system_message=full_system_message,
            # Optional: You could add 'verbose=verbose' if the function supports it
        )

        return agent_graph

    except OpenAIError as e:
        st.error(f"LangChain Agent Initialization Error: {str(e)}")
        return None

# =============================================================================
# Main Streamlit App - Agents Page
# =============================================================================
def main():
    st.set_page_config(page_title="Agents", page_icon="🤖")
    st.title("LangChain Agents (LangGraph Version)")

    # Initialize session state for messages if not exists
    if "messages" not in st.session_state:
        st.session_state["messages"] = [
            {"role": "assistant", "content": "Hello! How can I help you today?"}
        ]

    # Sidebar: Model selection, system prompt, streaming toggle, token counting toggle, verbosity toggle
    with st.sidebar:
        st.header("Configuration")
        model_choice = st.selectbox("Select the primary model:", AVAILABLE_MODELS, index=0)
        default_system_prompt = "You are an AI assistant."
        system_prompt = st.text_area("Set System Prompt", default_system_prompt)
        streaming_enabled = st.checkbox("Enable Streaming", value=False)
        token_counting_enabled = st.checkbox("Enable Token Counting", value=False)
        verbosity_enabled = st.checkbox("Enable Verbose Mode", value=False)

        # Handle o1-mini streaming limitation
        if model_choice == "o1-mini" and streaming_enabled:
            st.warning("Streaming is not supported for o1-mini. Falling back to non-streaming.")

    # Initialize the new LangGraph-based ReAct agent
    agent_graph = get_langchain_agent(model_choice, system_prompt, verbosity_enabled)
    if not agent_graph:
        return

    # Display existing conversation
    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    # Chat input box
    if prompt := st.chat_input("Ask a question..."):
        st.session_state["messages"].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        # Container for the assistant's response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            response_text = ""

            if verbosity_enabled:
                with st.expander("View Agent's Reasoning", expanded=False):
                    st.write("**Model:**", model_choice)
                    st.markdown("---")
                    st.write("**LangGraph ReAct Process:**")

            with st.spinner("Thinking..."):
                try:
                    # In LangGraph, you typically call `invoke()` or `stream_invoke()`
                    # on the agent to get the final answer. We'll handle streaming or not.
                    if streaming_enabled and model_choice != "o1-mini":
                        # If create_react_agent supports streaming, we can do stream_invoke
                        # Otherwise, we can do a manual chunk approach.
                        # We'll show a general approach (may vary with exact library version):
                        for chunk in agent_graph.stream_invoke(prompt):
                            response_text += chunk
                            message_placeholder.markdown(response_text.strip())
                    else:
                        # Non-streaming approach
                        response_text = agent_graph.invoke(prompt)
                        message_placeholder.write(response_text)
                except Exception as e:
                    error_message = str(e)
                    st.error(f"An error occurred: {error_message}")
                    response_text = ("I apologize, but I encountered an error processing your request. "
                                     "Please try again or rephrase your query.")

        st.session_state["messages"].append({"role": "assistant", "content": response_text})

if __name__ == "__main__":
    main()