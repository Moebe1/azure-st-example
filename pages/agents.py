import streamlit as st
from openai import AzureOpenAI, OpenAIError
import re

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

# Create an Azure OpenAI client
client = AzureOpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_version=AZURE_OPENAI_API_VERSION
)

# =============================================================================
# Function: Get OpenAI Response
# =============================================================================
def get_openai_response(messages, model_name, stream=False):
    """
    Fetches a response from Azure OpenAI using the OpenAI Python library.
    Handles both streaming and non-streaming responses.
    """
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            stream=stream
        )
        return response
    except OpenAIError as e:
        st.error(f"OpenAI API Error: {str(e)}")
        return None

# =============================================================================
# Main Streamlit App - Agents Page
# =============================================================================
def main():
    st.set_page_config(page_title="Agents", page_icon="🤖")
    st.title("Azure OpenAI Agents")

    # Initialize session state for conversation
    if "messages" not in st.session_state:
        st.session_state["messages"] = [
            {"role": "assistant", "content": "Hello! How can I assist you today?"}
        ]

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
            assistant_text = ""
            usage_info = None

            if verbosity_enabled:
                st.write(f"**Processing query using model:** {model_choice}")

            # Decide if we do streaming or non-streaming
            if streaming_enabled and model_choice != "o1-mini":
                with st.spinner("Thinking..."):
                    response_generator = get_openai_response(st.session_state["messages"], model_choice, stream=True)
                    if not response_generator:
                        return  # If there's an error, stop here

                    for update in response_generator:
                        if update and hasattr(update, "choices") and update.choices:
                            chunk = update.choices[0].delta.content or ""
                            assistant_text += chunk
                            message_placeholder.write(assistant_text)

            else:
                with st.spinner("Thinking..."):
                    response = get_openai_response(st.session_state["messages"], model_choice, stream=False)
                    if not response:
                        return  # If there's an error, stop here

                    if response.choices and response.choices[0].message:
                        assistant_text = response.choices[0].message.content or ""
                    message_placeholder.write(assistant_text)

            if verbosity_enabled:
                st.write("**Response generated successfully.**")

        st.session_state["messages"].append({"role": "assistant", "content": assistant_text})

if __name__ == "__main__":
    main()