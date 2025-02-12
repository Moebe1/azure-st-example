import streamlit as st
from openai import AzureOpenAI, OpenAIError
import re
import markdown2  # Converts assistant's Markdown to HTML

# =============================================================================
# Configuration - Azure OpenAI
# =============================================================================
AZURE_OPENAI_API_KEY = st.secrets["AZURE_OPENAI_API_KEY"]
AZURE_OPENAI_ENDPOINT = st.secrets["AZURE_OPENAI_ENDPOINT"]
AZURE_OPENAI_DEPLOYMENT = st.secrets["AZURE_OPENAI_DEPLOYMENT"]
AZURE_OPENAI_API_VERSION = st.secrets["AZURE_OPENAI_API_VERSION"]

# Create an Azure OpenAI client
client = AzureOpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_version=AZURE_OPENAI_API_VERSION
)

# =============================================================================
# Function: Get response from Azure OpenAI
# =============================================================================
def get_openai_response(messages):
    """ Fetches a response from Azure OpenAI using the OpenAI Python library. """
    try:
        response = client.chat.completions.create(
            model=AZURE_OPENAI_DEPLOYMENT,  # ✅ Use "model" instead of "engine"
            messages=messages
        )
        return response.choices[0].message.content
    except OpenAIError as e:
        st.error(f"OpenAI API Error: {str(e)}")
        return None

# =============================================================================
# Helper: Format a Single Message in HTML
# =============================================================================
def format_message(sender: str, html_text: str, align: str = "left"):
    """ Formats user and assistant messages for clean Streamlit UI. """
    return f"""
<div style="text-align: {align}; margin-bottom: 1rem;">
  <strong>{sender}:</strong><br/>
  <div style="white-space: pre-wrap; margin-top: 0.5rem;">
{html_text}
  </div>
</div>
""".strip()

# =============================================================================
# Main Streamlit App
# =============================================================================
def main():
    st.set_page_config(page_title="Azure OpenAI Chat", page_icon="💬")
    st.title("Azure OpenAI Chat Interface")

    # Initialize session state for conversation
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "conversation_html" not in st.session_state:
        st.session_state.conversation_html = ""

    # Sidebar for clearing conversation
    with st.sidebar:
        st.header("Configuration")
        st.write("Ensure your Azure OpenAI API key and endpoint are correct.")
        if st.button("Clear Conversation"):
            st.session_state.messages = []
            st.session_state.conversation_html = ""

    # Placeholder for displaying conversation
    conversation_placeholder = st.empty()

    # Form for user input
    with st.form(key="chat_form", clear_on_submit=True):
        user_input = st.text_input("Your message", key="user_input")
        submit_button = st.form_submit_button("Send")

    # Render existing conversation
    conversation_placeholder.markdown(
        st.session_state.conversation_html, 
        unsafe_allow_html=True
    )

    # Process new user message
    if submit_button and user_input:
        # 1) Store user message
        st.session_state.messages.append({"role": "user", "content": user_input})

        # 2) Convert user message to HTML (to support Markdown if needed)
        user_html = markdown2.markdown(user_input, extras=["tables"])

        # 3) Append user message to conversation (no extra forced "\n")
        st.session_state.conversation_html += format_message("User", user_html, align="right")

        # Update UI
        conversation_placeholder.markdown(
            st.session_state.conversation_html, 
            unsafe_allow_html=True
        )

        # 4) Fetch assistant response from Azure OpenAI
        assistant_text = get_openai_response(st.session_state.messages)
        
        if not assistant_text:
            return

        # === Minimal Whitespace Cleanup ===
        # - Strip trailing spaces per line
        # - Remove leading/trailing empty lines but KEEP paragraphs
        assistant_text = re.sub(r'[ \t]+$', '', assistant_text, flags=re.MULTILINE)
        assistant_text = re.sub(r'^\s*\n', '', assistant_text)
        assistant_text = re.sub(r'\n\s*$', '', assistant_text)
        
        # 5) Convert assistant response to HTML (preserve tables, lists)
        assistant_html = markdown2.markdown(assistant_text, extras=["tables"])

        # 6) Append assistant message to conversation
        st.session_state.messages.append({"role": "assistant", "content": assistant_text})
        st.session_state.conversation_html += format_message("Assistant", assistant_html, align="left")

        # Final re-render
        conversation_placeholder.markdown(
            st.session_state.conversation_html, 
            unsafe_allow_html=True
        )

if __name__ == "__main__":
    main()