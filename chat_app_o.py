import streamlit as st
from openai import AzureOpenAI, OpenAIError
import re
import json
from datetime import datetime
import os

# =============================================================================
# Configuration - Azure OpenAI
# =============================================================================
AZURE_OPENAI_API_KEY = st.secrets["AZURE_OPENAI_API_KEY"]
AZURE_OPENAI_ENDPOINT = st.secrets["AZURE_OPENAI_ENDPOINT"]
AZURE_OPENAI_API_VERSION = st.secrets["AZURE_OPENAI_API_VERSION"]

# Token limits per model
MODEL_TOKEN_LIMITS = {
    "o1-mini": 4096,
    "gpt-4o": 8192,
    "gpt-4o-mini": 8192
}

# List of deployments you have in Azure OpenAI
AVAILABLE_MODELS = list(MODEL_TOKEN_LIMITS.keys())

# Create an Azure OpenAI client (api_version applies to all deployments)
client = AzureOpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_version=AZURE_OPENAI_API_VERSION
)

# =============================================================================
# Function: Non-Streaming Response
# =============================================================================
def get_openai_response(messages, model_name):
    """
    Fetches a response (non-streaming) from Azure OpenAI using the OpenAI Python library.
    Returns the full response object so we can access 'usage' fields.
    """
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            stream=False
        )
        return response
    except OpenAIError as e:
        st.error(f"OpenAI API Error: {str(e)}")
        return None

# =============================================================================
# Function: Streaming Response
# =============================================================================
def get_openai_streaming_response(messages, model_name):
    """
    Returns a generator that yields partial content from Azure OpenAI Chat.
    We set stream=True to get a streaming response.
    """
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            stream=True
        )
        return response  # This is a generator
    except OpenAIError as e:
        st.error(f"OpenAI API Error: {str(e)}")
        return None

# =============================================================================
# Main Streamlit App
# =============================================================================
def main():
    st.set_page_config(page_title="Azure OpenAI Chat", page_icon="💬", layout="wide")
    st.title("Azure OpenAI Chat Interface")

    # Initialize session states
    if "messages" not in st.session_state:
        st.session_state["messages"] = [
            {
                "role": "assistant",
                "content": "Hello! How can I help you today?",
                "timestamp": datetime.now().isoformat()
            }
        ]
    if "total_tokens_used" not in st.session_state:
        st.session_state["total_tokens_used"] = 0
    if "current_session" not in st.session_state:
        st.session_state["current_session"] = "new_session"

    # Sidebar: Configuration and Session Management
    with st.sidebar:
        st.header("Configuration")
        
        # Session Management
        st.subheader("Session Management")
        sessions = list_sessions()
        session_options = ["new_session"] + sessions
        selected_session = st.selectbox(
            "Select Session",
            session_options,
            index=session_options.index(st.session_state["current_session"])
        )

        # Handle session changes
        if selected_session != st.session_state["current_session"]:
            if selected_session == "new_session":
                st.session_state["messages"] = [
                    {"role": "assistant", "content": "Hello! How can I help you today?"}
                ]
                st.session_state["total_tokens_used"] = 0
            else:
                messages, total_tokens = load_session(selected_session)
                if messages:
                    st.session_state["messages"] = messages
                    st.session_state["total_tokens_used"] = total_tokens
            st.session_state["current_session"] = selected_session

        # Save Session Button
        if st.session_state["current_session"] == "new_session":
            new_session_name = st.text_input("Session Name", value="", key="new_session_name")
            if st.button("Save Session") and new_session_name.strip():
                save_session(
                    new_session_name,
                    st.session_state["messages"],
                    st.session_state["total_tokens_used"]
                )
                st.success(f"Session '{new_session_name}' saved!")
                st.session_state["current_session"] = new_session_name
                st.experimental_rerun()

        st.divider()
        
        # Model and Feature Configuration
        st.subheader("Chat Configuration")
        model_choice = st.selectbox("Select the Azure deployment:", AVAILABLE_MODELS, index=0)
        streaming_enabled = st.checkbox("Enable Streaming", value=False)
        token_counting_enabled = st.checkbox("Enable Token Counting", value=False)

        # Clear conversation
        if st.button("Clear Conversation"):
            st.session_state["messages"] = [
                {
                    "role": "assistant",
                    "content": "Conversation cleared. How can I help you now?",
                    "timestamp": datetime.now().isoformat()
                }
            ]
            st.session_state["total_tokens_used"] = 0

    # Main chat interface
    chat_container = st.container()
    with chat_container:
        # Display existing conversation
        for msg in st.session_state["messages"]:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

    # Chat input box at the bottom
    if prompt := st.chat_input("Type your message here…"):
        # 1) Append the user's message to conversation
        st.session_state["messages"].append({
            "role": "user",
            "content": prompt,
            "timestamp": datetime.now().isoformat()
        })
        with st.chat_message("user"):
            st.markdown(prompt)

        # Prepare messages with system prompt
        messages_with_system = [
            {"role": "system", "content": st.session_state["system_prompt"]}
        ] + st.session_state["messages"]

        # Prepare a container for the assistant's response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            assistant_text = ""
            usage_info = None

            # 2) Decide if we do streaming or non-streaming
            if streaming_enabled:
                # STREAMING approach
                response_generator = get_openai_streaming_response(messages_with_system, model_choice)
                if not response_generator:
                    return  # If there's an error, stop here

                # 2A) Iterate over chunks in the streaming response
                for update in response_generator:
                    if update and hasattr(update, "choices") and update.choices:
                        # Safely handle the possibility that content can be None
                        chunk = ""
                        if update.choices[0].delta and hasattr(update.choices[0].delta, "content"):
                            chunk = update.choices[0].delta.content or ""  # Convert None to empty string

                        assistant_text += chunk
                        message_placeholder.write(assistant_text)

                        # If usage is attached (often only on the final chunk)
                        if hasattr(update, "usage") and update.usage:
                            usage_info = update.usage

            else:
                # NON-STREAMING approach
                response = get_openai_response(messages_with_system, model_choice)
                if not response:
                    return  # If there's an error, stop here

                # Extract assistant text from the response
                if response.choices and response.choices[0].message:
                    assistant_text = response.choices[0].message.content or ""
                usage_info = getattr(response, "usage", None)
                # Immediately display the final text
                message_placeholder.markdown(assistant_text)

            # 3) Cleanup whitespace
            assistant_text = re.sub(r'[ \t]+$', '', assistant_text, flags=re.MULTILINE)
            assistant_text = re.sub(r'^\s*\n', '', assistant_text)
            assistant_text = re.sub(r'\n\s*$', '', assistant_text)

        # 4) Append the assistant's final message to the conversation
        st.session_state["messages"].append({"role": "assistant", "content": assistant_text})

        # Auto-save current session if it's not a new session
        if st.session_state["current_session"] != "new_session":
            save_session(
                st.session_state["current_session"],
                st.session_state["messages"],
                st.session_state["total_tokens_used"]
            )

        # 5) Token counting if enabled and usage is present
        if token_counting_enabled and usage_info:
            prompt_tokens = getattr(usage_info, "prompt_tokens", 0) or 0
            completion_tokens = getattr(usage_info, "completion_tokens", 0) or 0
            total_tokens = getattr(usage_info, "total_tokens", 0) or 0
            st.session_state["total_tokens_used"] += total_tokens
            st.rerun()  # Update the display with new token count

if __name__ == "__main__":
    main()
