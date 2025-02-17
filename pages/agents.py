import streamlit as st
from langchain_openai import AzureChatOpenAI
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
        from langchain.prompts import PromptTemplate
        from langchain.chains import LLMChain

        # Define Chain-of-Thought (CoT) prompt template
        chain_of_thought_template = PromptTemplate(
            input_variables=["input_question"],
            template="""
You are a helpful AI assistant specialized in step-by-step reasoning (Chain-of-Thought).
Please reason through the problem carefully and derive the answer.

Problem: {input_question}

Let's break down the steps to solve this:

Step-by-Step Reasoning:
1) Analyze the question and identify key concepts or themes.
2) Consider diverse perspectives, including relevant theories, facts, or approaches.
3) Provide detailed explanations and illustrative examples or analogies to clarify the reasoning.
4) Synthesize the information logically to form a unified and coherent understanding.
5) Derive a concise and accurate conclusion based on the reasoning process.

Final Answer:
"""
        )

        # Create LLMChain with CoT prompt
        llm = AzureChatOpenAI(
            azure_deployment=model_choice,
            openai_api_key=AZURE_OPENAI_API_KEY,
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            openai_api_version=AZURE_OPENAI_API_VERSION,
            streaming=True if model_choice != "o1-mini" else False
        )
        cot_chain = LLMChain(llm=llm, prompt=chain_of_thought_template, verbose=verbose)
        tools = [Tool(name="Example Tool", func=lambda x: f"Processed: {x}", description="An example tool.")]
        agent = initialize_agent(
            tools,
            llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=verbose,
            handle_parsing_errors=True,
            max_iterations=5  # Increased to allow more reasoning steps
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

        # Handle 01-mini streaming limitation
        if model_choice == "o1-mini" and streaming_enabled:
            st.warning("Streaming is not supported for o1-mini. Falling back to non-streaming.")

    # Initialize agent
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
                        response_generator = agent.run(prompt)
                        unified_expander = st.expander("Agent's Reasoning and Response", expanded=True) if verbosity_enabled else None
                        reasoning_text = ""
                        for chunk in response_generator:
                            response_text += chunk
                            if verbosity_enabled:
                                reasoning_text += chunk
                            message_placeholder.markdown(response_text.strip())
                        if verbosity_enabled and unified_expander:
                            # Post-process reasoning text to form a unified, polished paragraph
                            refined_reasoning = " ".join(reasoning_text.splitlines()).strip()
                            unified_expander.markdown(refined_reasoning)
                    else:
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