import streamlit as st
from openai import AzureOpenAI, OpenAIError
import re
from azure.identity import DefaultAzureCredential, get_bearer_token_provider

# =============================================================================
# Configuration - Azure OpenAI
# =============================================================================
AZURE_OPENAI_API_KEY = st.secrets["AZURE_OPENAI_API_KEY"]
AZURE_OPENAI_ENDPOINT = st.secrets["AZURE_OPENAI_ENDPOINT"]
AZURE_OPENAI_API_VERSION = st.secrets["AZURE_OPENAI_API_VERSION"]

# If you are using Azure AD token flow:
default_credential = DefaultAzureCredential()
token_provider = get_bearer_token_provider(
    default_credential,
    "https://cognitiveservices.azure.com/.default",
)

AVAILABLE_MODELS = [
    "o1-mini",
    "gpt-4o"  # example
]

# Create an Azure OpenAI client with Azure AD token
client = AzureOpenAI(
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    azure_ad_token_provider=token_provider,
    api_version=AZURE_OPENAI_API_VERSION
)

# =============================================================================
# Function: Get streaming response from Azure OpenAI
# =============================================================================
def get_openai_streaming_response(messages, model_name):
    """
    Returns a generator that yields partial content from Azure OpenAI Chat.
    Set stream=True to get the streaming response.
    """
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            stream=True,       # IMPORTANT: streaming
            max_tokens=4096,
            temperature=1.0,
            top_p=1.0,
        )
        return response  # This is an iterator / generator
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
        # 1) Append user message to the conversation
        st.session_state["messages"].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        # 2) Set up a placeholder for the assistant's response so we can stream updates
        with st.chat_message("assistant"):
            message_placeholder = st.empty()  # Will hold partial text as we stream
            full_response_text = ""           # Accumulates the streamed content

            # 3) Call the streaming function
            response_generator = get_openai_streaming_response(st.session_state["messages"], model_choice)
            if not response_generator:
                return

            # 4) Iterate over partial responses
            for update in response_generator:
                if update and update.choices and update.choices[0].delta:
                    chunk = update.choices[0].delta.get("content", "")
                    full_response_text += chunk
                    # Update the placeholder with the partial text
                    message_placeholder.markdown(full_response_text)

            # 5) Minimal whitespace cleanup for final text (optional)
            final_assistant_text = re.sub(r'[ \t]+$', '', full_response_text, flags=re.MULTILINE)
            final_assistant_text = re.sub(r'^\s*\n', '', final_assistant_text)
            final_assistant_text = re.sub(r'\n\s*$', '', final_assistant_text)

        # 6) Store final text back into session state
        st.session_state["messages"].append({"role": "assistant", "content": final_assistant_text})


if __name__ == "__main__":
    main()
