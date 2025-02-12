import streamlit as st
from openai import AzureOpenAI, OpenAIError
import re

# =============================================================================
# Configuration - Azure OpenAI
# =============================================================================
AZURE_OPENAI_API_KEY = st.secrets["AZURE_OPENAI_API_KEY"]
AZURE_OPENAI_ENDPOINT = st.secrets["AZURE_OPENAI_ENDPOINT"]
AZURE_OPENAI_API_VERSION = st.secrets["AZURE_OPENAI_API_VERSION"]

AVAILABLE_MODELS = [
    "o1-mini",
    "gpt-4o"
]

client = AzureOpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_version=AZURE_OPENAI_API_VERSION
)

def get_openai_response(messages, model_name):
    """
    Fetches a response from Azure OpenAI using the OpenAI Python library.
    `model_name` must match one of your Azure deployment names.
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

def main():
    st.set_page_config(page_title="Azure OpenAI Chat", page_icon="💬")
    st.title("Azure OpenAI Chat Interface")

    # Initialize session state for conversation
    if "messages" not in st.session_state:
        st.session_state["messages"] = [
            {"role": "assistant", "content": "Hello! How can I help you today?"}
        ]
    
    # We can track total tokens across the entire session
    if "total_tokens_used" not in st.session_state:
        st.session_state["total_tokens_used"] = 0

    # Sidebar
    with st.sidebar:
        st.header("Configuration")
        model_choice = st.selectbox("Select the Azure deployment:", AVAILABLE_MODELS, index=0)
        
        # Toggle for token counting
        token_counting_enabled = st.checkbox("Enable Token Counting", value=False)
        
        # Clear conversation button
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
        # 1) User message
        st.session_state["messages"].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        # 2) Assistant response
        response = get_openai_response(st.session_state["messages"], model_choice)
        if response is None:
            return

        # Extract assistant text
        assistant_text = response.choices[0].message.content
        assistant_text = re.sub(r'[ \t]+$', '', assistant_text, flags=re.MULTILINE)
        assistant_text = re.sub(r'^\s*\n', '', assistant_text)
        assistant_text = re.sub(r'\n\s*$', '', assistant_text)

        st.session_state["messages"].append({"role": "assistant", "content": assistant_text})
        with st.chat_message("assistant"):
            st.write(assistant_text)

        # 3) Token counting if enabled
        if token_counting_enabled:
            usage = getattr(response, "usage", None)
            if usage:
                prompt_tokens = usage.get("prompt_tokens", 0)
                completion_tokens = usage.get("completion_tokens", 0)
                total_tokens = usage.get("total_tokens", 0)

                # Add to session total
                st.session_state["total_tokens_used"] += total_tokens

                st.write(
                    f"**Tokens Used**: "
                    f"Prompt={prompt_tokens}, "
                    f"Completion={completion_tokens}, "
                    f"Total={total_tokens} "
                    f"(Session Total={st.session_state['total_tokens_used']})"
                )
            else:
                st.warning("Token usage data not returned by the API.")

if __name__ == "__main__":
    main()
