import streamlit as st
from openai import AzureOpenAI, OpenAIError
import re

# =============================================================================
# Configuration - Azure OpenAI
# =============================================================================
AZURE_OPENAI_API_KEY = st.secrets["AZURE_OPENAI_API_KEY"]
AZURE_OPENAI_ENDPOINT = st.secrets["AZURE_OPENAI_ENDPOINT"]
AZURE_OPENAI_API_VERSION = st.secrets["AZURE_OPENAI_API_VERSION"]

# List your available Azure OpenAI deployment names
AVAILABLE_MODELS = [
    "o1-mini",
    "gpt-4o"
]

# Create an Azure OpenAI client (using API key, not Azure Identity)
client = AzureOpenAI(
    api_key=AZURE_OPENAI_API_KEY,      # Use API Key
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_version=AZURE_OPENAI_API_VERSION
)

# =============================================================================
# Function: Get streaming response from Azure OpenAI
# =============================================================================
def get_openai_streaming_response(messages, model_name, max_tokens=1024):
    """
    Returns a generator that yields partial content from Azure OpenAI Chat.
    """
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            stream=True,       # IMPORTANT: streaming mode
            max_tokens=max_tokens,
            temperature=1.0,
            top_p=1.0,
        )
        # The returned response is a generator, not a single response object
        return response
    except OpenAIError as e:
        st.error(f"OpenAI API Error: {str(e)}")
        return None

# =============================================================================
# Main Streamlit App
# =============================================================================
def main():
    st.set_page_config(page_title="Azure OpenAI Chat", page_icon="💬")
    st.title("Azure OpenAI Chat Interface (Streaming)")

    # Initialize session state for conversation
    if "messages" not in st.session_state:
        st.session_state["messages"] = [
            {"role": "assistant", "content": "Hello! How can I help you today?"}
        ]

    # Sidebar: Model Selection + Clear Conversation
    with st.sidebar:
        st.header("Configuration")
        st.write("You are using an API Key (no Azure Identity).")
        model_choice = st.selectbox("Select the Azure deployment:", AVAILABLE_MODELS, index=0)

        if st.button("Clear Conversation"):
            st.session_state["messages"] = [
                {"role": "assistant", "content": "Conversation cleared. How can I help you now?"}
            ]

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

        # 2) Create a placeholder for the assistant's streaming response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response_text = ""

            # 3) Stream the response
            response_generator = get_openai_streaming_response(
                st.session_state["messages"],
                model_choice,
                max_tokens=1024
            )
            if not response_generator:
                return

            for update in response_generator:
                # Each 'update' is part of the streaming conversation
                if update.choices and update.choices[0].delta:
                    chunk = update.choices[0].delta.get("content", "")
                    full_response_text += chunk
                    # Update the placeholder with the partial text
                    message_placeholder.markdown(full_response_text)

            # 4) Clean up whitespace (optional)
            final_assistant_text = re.sub(r'[ \t]+$', '', full_response_text, flags=re.MULTILINE)
            final_assistant_text = re.sub(r'^\s*\n', '', final_assistant_text)
            final_assistant_text = re.sub(r'\n\s*$', '', final_assistant_text)

        # 5) Store the final text in session state
        st.session_state["messages"].append({"role": "assistant", "content": final_assistant_text})

if __name__ == "__main__":
    main()
