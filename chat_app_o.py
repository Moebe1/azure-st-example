import streamlit as st
from openai import AzureOpenAI, OpenAIError
import re
import json
from datetime import datetime
import os
import markdown
import base64
from pathlib import Path

# =============================================================================
# Configuration - Azure OpenAI
# =============================================================================
AZURE_OPENAI_API_KEY = st.secrets["AZURE_OPENAI_API_KEY"]
AZURE_OPENAI_ENDPOINT = st.secrets["AZURE_OPENAI_ENDPOINT"]
AZURE_OPENAI_API_VERSION = st.secrets["AZURE_OPENAI_API_VERSION"]

# List of deployments you have in Azure OpenAI
AVAILABLE_MODELS = [
    "o1-mini",
    "gpt-4o",
    "gpt-4o-mini"
    # Add or remove models here as needed
]

# Create an Azure OpenAI client (api_version applies to all deployments)
client = AzureOpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_version=AZURE_OPENAI_API_VERSION
)

# =============================================================================
# System Prompts
# =============================================================================
DEFAULT_SYSTEM_PROMPTS = {
    "General Assistant": "You are a helpful assistant ready to help with any task.",
    "Technical Expert": "You are a technical expert focused on providing detailed, accurate technical information and code examples.",
    "Creative Writer": "You are a creative writing assistant focused on helping with writing and storytelling.",
    "Professional": "You are a professional assistant focused on business communication and formal interactions."
}

# =============================================================================
# Chat Session Management
# =============================================================================
SESSIONS_DIR = "chat_sessions"
EXPORTS_DIR = "chat_exports"
os.makedirs(SESSIONS_DIR, exist_ok=True)
os.makedirs(EXPORTS_DIR, exist_ok=True)

def save_session(session_name, messages, total_tokens=0, system_prompt=""):
    """Save the current chat session to a file"""
    session_data = {
        "messages": messages,
        "total_tokens": total_tokens,
        "timestamp": datetime.now().isoformat(),
        "system_prompt": system_prompt
    }
    file_path = os.path.join(SESSIONS_DIR, f"{session_name}.json")
    with open(file_path, "w") as f:
        json.dump(session_data, f, indent=2)

def load_session(session_name):
    """Load a chat session from a file"""
    file_path = os.path.join(SESSIONS_DIR, f"{session_name}.json")
    try:
        with open(file_path, "r") as f:
            data = json.load(f)
            return data["messages"], data.get("total_tokens", 0), data.get("system_prompt", "")
    except FileNotFoundError:
        return None, 0, ""

def list_sessions():
    """List all available chat sessions"""
    if not os.path.exists(SESSIONS_DIR):
        return []
    sessions = [f[:-5] for f in os.listdir(SESSIONS_DIR) if f.endswith('.json')]
    return sorted(sessions, reverse=True)

def export_chat(messages, format="markdown"):
    """Export chat history to various formats"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if format == "markdown":
        content = "# Chat Export\n\n"
        content += f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        for msg in messages:
            role = msg["role"].capitalize()
            content += f"## {role}\n\n{msg['content']}\n\n"
        
        file_path = os.path.join(EXPORTS_DIR, f"chat_export_{timestamp}.md")
        with open(file_path, "w") as f:
            f.write(content)
        return file_path
    return None

def get_download_link(file_path):
    """Generate a download link for a file"""
    with open(file_path, "rb") as f:
        bytes_data = f.read()
    b64 = base64.b64encode(bytes_data).decode()
    filename = Path(file_path).name
    return f'<a href="data:application/octet-stream;base64,{b64}" download="{filename}">Download {filename}</a>'

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
    # Theme settings
    if "theme" not in st.session_state:
        st.session_state["theme"] = "light"
    
    # Apply theme
    if st.session_state["theme"] == "dark":
        st.markdown("""
        <style>
        .stApp {
            background-color: #1E1E1E;
            color: #FFFFFF;
        }
        .stMarkdown {
            color: #FFFFFF;
        }
        </style>
        """, unsafe_allow_html=True)

    st.set_page_config(
        page_title="Azure OpenAI Chat",
        page_icon="💬",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    st.title("Azure OpenAI Chat Interface")

    # Initialize session states
    if "messages" not in st.session_state:
        st.session_state["messages"] = [
            {"role": "assistant", "content": "Hello! How can I help you today?"}
        ]
    if "total_tokens_used" not in st.session_state:
        st.session_state["total_tokens_used"] = 0
    if "current_session" not in st.session_state:
        st.session_state["current_session"] = "new_session"
    if "system_prompt" not in st.session_state:
        st.session_state["system_prompt"] = DEFAULT_SYSTEM_PROMPTS["General Assistant"]

    # Sidebar: Configuration and Session Management
    with st.sidebar:
        st.header("Configuration")
        
        # Theme Toggle
        theme = st.selectbox(
            "Theme",
            ["light", "dark"],
            index=0 if st.session_state["theme"] == "light" else 1
        )
        if theme != st.session_state["theme"]:
            st.session_state["theme"] = theme
            st.experimental_rerun()
        
        # System Prompt Selection
        st.subheader("System Prompt")
        system_prompt_type = st.selectbox(
            "Personality",
            list(DEFAULT_SYSTEM_PROMPTS.keys())
        )
        custom_prompt = st.text_area(
            "Custom System Prompt",
            value=st.session_state["system_prompt"],
            help="Define custom behavior for the assistant"
        )
        if custom_prompt != st.session_state["system_prompt"]:
            st.session_state["system_prompt"] = custom_prompt

        st.divider()
        
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
                st.session_state["system_prompt"] = DEFAULT_SYSTEM_PROMPTS["General Assistant"]
            else:
                messages, total_tokens, system_prompt = load_session(selected_session)
                if messages:
                    st.session_state["messages"] = messages
                    st.session_state["total_tokens_used"] = total_tokens
                    st.session_state["system_prompt"] = system_prompt
            st.session_state["current_session"] = selected_session

        # Save Session Button
        if st.session_state["current_session"] == "new_session":
            new_session_name = st.text_input("Session Name", value="", key="new_session_name")
            if st.button("Save Session") and new_session_name.strip():
                save_session(
                    new_session_name,
                    st.session_state["messages"],
                    st.session_state["total_tokens_used"],
                    st.session_state["system_prompt"]
                )
                st.success(f"Session '{new_session_name}' saved!")
                st.session_state["current_session"] = new_session_name
                st.experimental_rerun()

        # Export Options
        st.subheader("Export Chat")
        if st.button("Export as Markdown"):
            export_path = export_chat(st.session_state["messages"], "markdown")
            if export_path:
                st.markdown(get_download_link(export_path), unsafe_allow_html=True)

        st.divider()
        
        # Model and Feature Configuration
        st.subheader("Chat Configuration")
        model_choice = st.selectbox("Select the Azure deployment:", AVAILABLE_MODELS, index=0)
        streaming_enabled = st.checkbox("Enable Streaming", value=False)
        token_counting_enabled = st.checkbox("Enable Token Counting", value=False)

        # Clear conversation
        if st.button("Clear Conversation"):
            st.session_state["messages"] = [
                {"role": "assistant", "content": "Conversation cleared. How can I help you now?"}
            ]
            st.session_state["total_tokens_used"] = 0

    # Main chat interface
    chat_container = st.container()
    with chat_container:
        # Display existing conversation
        for msg in st.session_state["messages"]:
            with st.chat_message(msg["role"]):
                # Convert markdown to HTML for better rendering
                html_content = markdown.markdown(msg["content"])
                st.markdown(html_content, unsafe_allow_html=True)

    # Chat input box at the bottom
    if prompt := st.chat_input("Type your message here…"):
        # 1) Append the user's message to conversation
        st.session_state["messages"].append({"role": "user", "content": prompt})
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
                        message_placeholder.markdown(assistant_text)

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
                st.session_state["total_tokens_used"],
                st.session_state["system_prompt"]
            )

        # 5) Token counting if enabled and usage is present
        if token_counting_enabled and usage_info:
            prompt_tokens = getattr(usage_info, "prompt_tokens", 0) or 0
            completion_tokens = getattr(usage_info, "completion_tokens", 0) or 0
            total_tokens = getattr(usage_info, "total_tokens", 0) or 0

            st.session_state["total_tokens_used"] += total_tokens

            st.write(
                f"**Tokens Used**: "
                f"Prompt={prompt_tokens}, "
                f"Completion={completion_tokens}, "
                f"Total={total_tokens} "
                f"(Session Total={st.session_state['total_tokens_used']})"
            )

if __name__ == "__main__":
    main()
