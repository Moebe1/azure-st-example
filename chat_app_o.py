import streamlit as st
from openai import AzureOpenAI, OpenAIError
import re

# =============================================================================
# Configuration - Azure OpenAI
# =============================================================================
AZURE_OPENAI_API_KEY = st.secrets["AZURE_OPENAI_API_KEY"]
AZURE_OPENAI_ENDPOINT = st.secrets["AZURE_OPENAI_ENDPOINT"]
AZURE_OPENAI_DEPLOYMENT = st.secrets["AZURE_OPENAI_DEPLOYMENT"]
AZURE_OPENAI_API_VERSION = st.secrets["AZURE_OPENAI_API_VERSION"]

# Create Azure OpenAI client
client = AzureOpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_version=AZURE_OPENAI_API_VERSION
)

# =============================================================================
# Function: Get response from Azure OpenAI
# =============================================================================
def get_openai_response(messages):
    """
    Fetches a response from Azure OpenAI using the OpenAI Python library.
    Expects messages to be a list of dicts, e.g. [{'role': 'user', 'content': 'Hello'}].
    """
    try:
        response = client.chat.completions.create(
            model=AZURE_OPENAI_DEPLOYMENT,  # "model" is used for Azure's new endpoint
            messages=messages
        )
        return response.choices[0].message.content
    except OpenAIError as e:
        st.error(f"OpenAI API Error: {str(e)}")
        return None

# =============================================================================
# Main Streamlit App
# =============================================================================
def main():
    st.set_page_config(page_title="Azure OpenAI Chat", page_icon="💬")
    st.title("Azure OpenAI Chat Interface")

    # Initialize conversation in session state
    if "messages" not in st.session_state:
        # Provide a default greeting from the assistant
        st.session_state["messages"] = [
            {"role": "assistant", "content": "Hello! How can I help you today?"}
        ]

    # Sidebar
    with st.sidebar:
        st.header("Configuration")
        st.write("Ensure your Azure OpenAI API key and endpoint are correct.")
        if st.button("Clear Conversation"):
            st.session_state["messages"] = [
                {"role": "assistant", "content": "Conversation cleared. How can I help you now?"}
            ]

    # Display the conversation so far
    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    # Chat input box at the bottom
    if prompt := st.chat_input("Type your message here…"):
        # 1) Append the user message to the conversation
        st.session_state["messages"].append({"role": "user", "content": prompt})

        # Display user message
        with st.chat_message("user"):
            st.write(prompt)

        # 2) Get the assistant's response from Azure OpenAI
        assistant_text = get_openai_response(st.session_state["messages"])
        if assistant_text is None:
            return  # No valid response, early exit

        # Minimal trailing/leading whitespace cleanup
        assistant_text = re.sub(r'[ \t]+$', '', assistant_text, flags=re.MULTILINE)
        assistant_text = re.sub(r'^\s*\n', '', assistant_text)
        assistant_text = re.sub(r'\n\s*$', '', assistant_text)

        # 3) Append the assistant message to the session and display
        st.session_state["messages"].append({"role": "assistant", "content": assistant_text})
        with st.chat_message("assistant"):
            st.write(assistant_text)

if __name__ == "__main__":
    main()
