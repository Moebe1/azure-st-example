import streamlit as st
from langchain_community.chat_models import ChatOpenAI
from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool
from openai import OpenAIError

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
# Function: Initialize LangChain Agent
# =============================================================================
def get_langchain_agent(model_choice, system_prompt, verbose):
    try:
        llm = ChatOpenAI(
            openai_api_key=AZURE_OPENAI_API_KEY,
            openai_api_base=AZURE_OPENAI_ENDPOINT,
            openai_api_version=AZURE_OPENAI_API_VERSION,
            model=model_choice,
            streaming=True if model_choice != "o1-mini" else False
        )
        tools = [Tool(name="Example Tool", func=lambda x: f"Processed: {x}", description="An example tool.")]
        agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=verbose)
        return agent
    except OpenAIError as e:
        st.error(f"LangChain Agent Initialization Error: {str(e)}")
        return None

# =============================================================================
# Main Streamlit App - Agents Page
# =============================================================================
def main():
    st.set_page_config(page_title="Agents", page_icon="🤖")
    st.title("LangChain Agents")

    # Sidebar: Model selection, system prompt, streaming toggle, token counting toggle, verbosity toggle
    with st.sidebar:
        st.header("Configuration")
        model_choice = st.selectbox("Select the primary model:", AVAILABLE_MODELS, index=0)
        system_prompt = st.text_area("Set System Prompt", "You are an AI assistant.")
        streaming_enabled = st.checkbox("Enable Streaming", value=False)
        token_counting_enabled = st.checkbox("Enable Token Counting", value=False)
        verbosity_enabled = st.checkbox("Enable Verbose Mode", value=False)

        # Handle 01-mini streaming limitation
        if model_choice == "o1-mini" and streaming_enabled:
            st.warning("Streaming is not supported for o1-mini. Falling back to non-streaming.")

    # Initialize agent
    agent = get_langchain_agent(model_choice, system_prompt, verbosity_enabled)
    if not agent:
        return

    # Chat input box
    if prompt := st.chat_input("Ask a question..."):
        st.session_state["messages"].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            response_text = ""

            if verbosity_enabled:
                st.write(f"**Processing query using model:** {model_choice}")

            if streaming_enabled and model_choice != "o1-mini":
                with st.spinner("Thinking..."):
                    response_generator = agent.run(prompt)
                    for chunk in response_generator:
                        response_text += chunk
                        message_placeholder.write(response_text)
            else:
                with st.spinner("Thinking..."):
                    response_text = agent.run(prompt)
                    message_placeholder.write(response_text)

            if verbosity_enabled:
                st.write("**Response generated successfully.**")

        st.session_state["messages"].append({"role": "assistant", "content": response_text})

if __name__ == "__main__":
    main()