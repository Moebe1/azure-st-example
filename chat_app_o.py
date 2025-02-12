import streamlit as st
from openai import AzureOpenAI, OpenAIError
import re

# =============================================================================
# Configuration - Azure OpenAI
# =============================================================================
AZURE_OPENAI_API_KEY = st.secrets["AZURE_OPENAI_API_KEY"]
AZURE_OPENAI_ENDPOINT = st.secrets["AZURE_OPENAI_ENDPOINT"]
AZURE_OPENAI_API_VERSION = st.secrets["AZURE_OPENAI_API_VERSION"]

# List of deployments you have in Azure OpenAI
AVAILABLE_MODELS = [
    "o1-mini",
    "gpt-4o"
    # Add or remove models here as needed
]

# Create an Azure OpenAI client (api_version applies to all deployments)
client = AzureOpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_version=AZURE_OPENAI_API_VERSION
)

# =============================================================================
# Function: Get response from Azure OpenAI
# =============================================================================
def get_openai_response(messages, model_name):
    """
    Fetches a response from Azure OpenAI using the OpenAI Python library.
    Return the full response object so we can access 'usage' fields.
    """
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=messages
        )
        return response
    except OpenAIError as e:
        st.error(f"OpenAI API Error: {str(e)}")
        return None

# =============================================================================
# Main Streamlit App
# =============================================================================
def main():
    st.set_page_config(page_title="Azure OpenAI Chat", page_icon="💬")
    st.title("Azure OpenAI Chat Interface")

    # Initialize session state for conversation
    if "messages" not in st.session_state:
        st.session_state["messages"] = [
            {"role": "assistant", "content": "Hello! How can I help you today?"}
        ]

    # Initialize total tokens used in the session
    if "total_tokens_used" not in st.session_state:
        st.session_state["total_tokens_used"] = 0

    # Sidebar: Model selection, token counting toggle, and clear button
    with st.sidebar:
        st.header("Configuration")
        st.write("Ensure your Azure OpenAI API key and endpoint are correct.")
        model_choice = st.selectbox("Select the Azure deployment:", AVAILABLE_MODELS, index=0)

        # Toggle for token counting
        token_counting_enabled = st.checkbox("Enable Token Counting", value=False)

        if st.button("Clear Conversation"):
            st.session_state["messages"] = [
                {"role": "assistant", "content": "Conversation cleared. How can I help you now?"}
            ]
            st.session_state["total_tokens_used"] = 0

    # Display existing conversation
    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    # Chat input box at the bottom
    if prompt := st.chat_input("Type your message here…"):
        # 1) Append the user's message to conversation
        st.session_state["messages"].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        # 2) Get assistant response from the selected model (full response object)
        response = get_openai_response(st.session_state["messages"], model_choice)
        if not response:
            return  # If there's an error, we stop here

        # Extract assistant text from the response
        assistant_text = response.choices[0].message.content
        # Minimal whitespace cleanup
        assistant_text = re.sub(r'[ \t]+$', '', assistant_text, flags=re.MULTILINE)
        assistant_text = re.sub(r'^\s*\n', '', assistant_text)
        assistant_text = re.sub(r'\n\s*$', '', assistant_text)

        # 3) Append the assistant's message to the conversation and display
        st.session_state["messages"].append({"role": "assistant", "content": assistant_text})
        with st.chat_message("assistant"):
            st.write(assistant_text)

        # 4) If token counting is enabled, pull usage info from the response
        if token_counting_enabled and hasattr(response, "usage") and response.usage:
            prompt_tokens = response.usage.prompt_tokens or 0
            completion_tokens = response.usage.completion_tokens or 0
            total_tokens = response.usage.total_tokens or 0

            # Accumulate total tokens in session state
            st.session_state["total_tokens_used"] += total_tokens

            # Display token usage
            st.write(
                f"**Tokens Used**: "
                f"Prompt={prompt_tokens}, "
                f"Completion={completion_tokens}, "
                f"Total={total_tokens} "
                f"(Session Total={st.session_state['total_tokens_used']})"
            )

if __name__ == "__main__":
    main()
