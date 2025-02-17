import streamlit as st
from openai import AzureOpenAI, OpenAIError
import re
from pydantic import BaseModel, Field, ValidationError
import datetime
import logging

# =============================================================================
# Suppress Streamlit Debug Messages
# =============================================================================
logging.getLogger("streamlit").setLevel(logging.ERROR)

# =============================================================================
# Configuration - Azure OpenAI
# =============================================================================
AZURE_OPENAI_API_KEY = st.secrets.get("AZURE_OPENAI_API_KEY", "Your-API-Key-Here")
AZURE_OPENAI_ENDPOINT = st.secrets.get("AZURE_OPENAI_ENDPOINT", "Your-Endpoint-Here")
AZURE_OPENAI_API_VERSION = st.secrets.get("AZURE_OPENAI_API_VERSION", "Your-API-Version-Here")

AVAILABLE_MODELS = [
    "o1-mini",
    "gpt-4o",
    "gpt-4o-mini"
]

client = AzureOpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_version=AZURE_OPENAI_API_VERSION
)

# =============================================================================
# Reflexions Agent Components
# =============================================================================
class Reflection(BaseModel):
    missing: str = Field(description="Critique of what is missing.")
    superfluous: str = Field(description="Critique of what is superfluous.")

class AnswerQuestion(BaseModel):
    answer: str = Field(description="~250 word detailed answer to the question.")
    reflection: Reflection = Field(description="Your reflection on the initial answer.")
    search_queries: list[str] = Field(description="1-3 search queries for researching improvements.")

class ReviseAnswer(AnswerQuestion):
    references: list[str] = Field(description="Citations motivating your updated answer.")

def get_openai_response(messages, model_name, streaming_enabled):
    try:
        if streaming_enabled:
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                stream=True
            )
            def token_generator():
                for chunk in response:
                    if "choices" in chunk and chunk["choices"]:
                        delta_content = chunk["choices"][0].get("delta", {}).get("content", "")
                        if delta_content:
                            yield delta_content
            return None, token_generator()
        else:
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                stream=False
            )
            reasoning_tokens = response.choices[0].message.content.split("\n")
            return response, reasoning_tokens
    except OpenAIError:
        st.error("An error occurred while communicating with the OpenAI API. Please try again.")
        return None, []

# =============================================================================
# Main Streamlit App
# =============================================================================
def main():
    st.set_page_config(page_title="Reflexions LangGraph Agent", page_icon="🤖")
    st.title("Reflexions LangGraph Agent")

    if "messages" not in st.session_state:
        st.session_state["messages"] = [
            {"role": "assistant", "content": "Hello! How can I assist you today?"}
        ]

    if "total_tokens_used" not in st.session_state:
        st.session_state["total_tokens_used"] = 0

    with st.sidebar:
        st.header("Configuration")
        model_choice = st.selectbox("Select the Azure deployment:", AVAILABLE_MODELS, index=0)
        streaming_enabled = st.checkbox("Enable Streaming", value=False)
        token_counting_enabled = st.checkbox("Enable Token Counting", value=False)
        if st.button("Clear Conversation"):
            st.session_state["messages"] = [
                {"role": "assistant", "content": "Conversation cleared. How can I help you now?"}
            ]
            st.session_state["total_tokens_used"] = 0

    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    if prompt := st.chat_input("Type your message here…"):
        st.session_state["messages"].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            assistant_text = ""
            usage_info = None

            with st.spinner("Thinking..."):
                response, reasoning_data = get_openai_response(st.session_state["messages"], model_choice, streaming_enabled)
                if not response and not streaming_enabled:
                    return

                reasoning_tokens = []
                if streaming_enabled:
                    reasoning_placeholder = st.empty()
                    for token in reasoning_data:
                        reasoning_tokens.append(token)
                        reasoning_placeholder.markdown("**Reasoning Tokens:**\n" + "".join(reasoning_tokens))
                else:
                    if response.choices and response.choices[0].message:
                        assistant_text = response.choices[0].message.content or ""
                    usage_info = getattr(response, "usage", None)
                    message_placeholder.write(assistant_text)

            assistant_text = re.sub(r'[ \t]+$', '', assistant_text, flags=re.MULTILINE)
            assistant_text = re.sub(r'^\s*\n', '', assistant_text)
            assistant_text = re.sub(r'\n\s*$', '', assistant_text)

        st.session_state["messages"].append({"role": "assistant", "content": assistant_text})

        if not streaming_enabled:
            st.markdown("**Reasoning Tokens:**\n" + "\n".join(reasoning_tokens))

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