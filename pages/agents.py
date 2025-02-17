import streamlit as st
from langchain_openai import AzureChatOpenAI
from langchain.agents import (
    ZeroShotAgent,
    AgentExecutor
)
from langchain.tools import Tool
from openai import OpenAIError
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

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
# Function: Initialize LangChain Agent (Option B with valid llm_chain)
# =============================================================================
def get_langchain_agent(model_choice, system_prompt, verbose):
    """
    Creates an agent that uses a custom prefix with chain-of-thought style 
    instructions (embedded in the prompt), while retaining the ability to 
    use tools in a ReAct-like manner.
    """
    try:
        # Instantiate the AzureChatOpenAI LLM
        llm = AzureChatOpenAI(
            azure_deployment=model_choice,
            openai_api_key=AZURE_OPENAI_API_KEY,
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            openai_api_version=AZURE_OPENAI_API_VERSION,
            streaming=True if model_choice != "o1-mini" else False
        )

        # Define any tools you want the agent to use
        tools = [
            Tool(
                name="Example Tool",
                func=lambda x: f"Processed: {x}",
                description="An example tool."
            )
        ]

        # ---------------------------------------------------------------------
        # 1) CREATE A CUSTOM PROMPT TEMPLATE (PREFIX + SUFFIX)
        # ---------------------------------------------------------------------
        CUSTOM_PREFIX = """You are a helpful AI assistant specialized in step-by-step reasoning (Chain-of-Thought).
Please reason through the problem carefully and derive the answer.

Use the following format:
Step-by-Step Reasoning:
1) Analyze the question and identify key concepts or themes.
2) Consider diverse perspectives, including relevant theories, facts, or approaches.
3) Provide detailed explanations and illustrative examples to clarify the reasoning.
4) Synthesize the information logically to form a coherent understanding.
5) Conclude with a concise and accurate answer.

Final Answer:
"""

        CUSTOM_SUFFIX = "Begin!"

        # Build a PromptTemplate for the agent
        custom_prompt = PromptTemplate(
            input_variables=["input", "agent_scratchpad"],
            template=f"{CUSTOM_PREFIX}{{agent_scratchpad}}\n\n{{input}}\n\n{CUSTOM_SUFFIX}"
        )

        # ---------------------------------------------------------------------
        # 2) CREATE THE LLM CHAIN (ZeroShotAgent REQUIRES llm_chain)
        # ---------------------------------------------------------------------
        llm_chain = LLMChain(
            llm=llm,
            prompt=custom_prompt,
            verbose=verbose
        )

        # Prepare a list of tool names for the agent to recognize
        allowed_tools = [tool.name for tool in tools]

        # ---------------------------------------------------------------------
        # 3) BUILD THE ZERO-SHOT AGENT USING THE LLM CHAIN
        # ---------------------------------------------------------------------
        agent_instance = ZeroShotAgent(
            llm_chain=llm_chain,
            allowed_tools=allowed_tools
        )

        # ---------------------------------------------------------------------
        # 4) WRAP THE AGENT IN AN AGENTEXECUTOR
        # ---------------------------------------------------------------------
        agent = AgentExecutor.from_agent_and_tools(
            agent=agent_instance,
            tools=tools,
            verbose=verbose,
            max_iterations=5,
            handle_parsing_errors=True
        )

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

    # Initialize agent using custom CoT prompt approach
    agent = get_langchain_agent(model_choice, system_prompt, verbosity_enabled)
    if not agent:
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

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            response_text = ""

            if verbosity_enabled:
                with st.expander("View Agent's Reasoning", expanded=False):
                    st.write("**Model:** ", model_choice)
                    st.markdown("---")
                    st.write("**Agent's Thought Process:**")

            with st.spinner("Thinking..."):
                try:
                    if streaming_enabled and model_choice != "o1-mini":
                        # Streaming approach
                        response_generator = agent.run(prompt)
                        unified_expander = st.expander("Agent's Reasoning and Response", expanded=True) if verbosity_enabled else None
                        reasoning_text = ""
                        for chunk in response_generator:
                            response_text += chunk
                            if verbosity_enabled:
                                reasoning_text += chunk
                            message_placeholder.markdown(response_text.strip())
                        if verbosity_enabled and unified_expander:
                            refined_reasoning = " ".join(reasoning_text.splitlines()).strip()
                            unified_expander.markdown(refined_reasoning)
                    else:
                        # Non-streaming approach
                        response_text = agent.run(prompt)
                        if verbosity_enabled:
                            reasoning_expander = st.expander("View Agent's Reasoning", expanded=True)
                            reasoning_expander.write(response_text)
                        message_placeholder.write(response_text)
                except Exception as e:
                    error_message = str(e)
                    st.error(f"An error occurred: {error_message}")
                    response_text = "I apologize, but I encountered an error processing your request. Please try again or rephrase your query."

        st.session_state["messages"].append({"role": "assistant", "content": response_text})

if __name__ == "__main__":
    main()