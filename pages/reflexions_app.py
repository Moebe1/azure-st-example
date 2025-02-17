import streamlit as st
from openai import AzureOpenAI, OpenAIError
import re
from pydantic import BaseModel, Field
import logging

# =============================================================================
# Suppress Streamlit Debug Messages
# =============================================================================
logging.getLogger("streamlit").setLevel(logging.ERROR)

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
# Reflexions Agent Components
# =============================================================================
class Reflection(BaseModel):
    missing: str = Field(description="Critique of what is missing.")
    superfluous: str = Field(description="Critique of what is superfluous.")

class AnswerQuestion(BaseModel):
    answer: str = Field(description="~250 word detailed answer to the question.")
    reflection: Reflection = Field(description="Your reflection on the initial answer.")
    revised_answer: str = Field(description="Revised answer based on the reflection.")

# =============================================================================
# Reflexion Actor Logic
# =============================================================================
def get_openai_response(messages, model_name):
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
# Main Streamlit App
# =============================================================================
def main():
    st.set_page_config(page_title="Reflexions LangGraph Agent", page_icon="🤖")
    st.title("Reflexions LangGraph Agent")

    if "messages" not in st.session_state:
        st.session_state["messages"] = [
            {"role": "assistant", "content": "Hello! How can I assist you today?"}
        ]

    with st.sidebar:
        st.header("Configuration")
        model_choice = st.selectbox("Select the Azure deployment:", AVAILABLE_MODELS, index=0)

        if st.button("Clear Conversation"):
            st.session_state["messages"] = [
                {"role": "assistant", "content": "Conversation cleared. How can I help you now?"}
            ]

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

            # Initial response
            with st.spinner("Thinking..."):
                initial_prompt = f"""You are a helpful AI assistant. Answer the following question in about 250 words: {prompt}
                After providing the answer, reflect on what is missing or superfluous in your response.
                """
                messages = [{"role": "user", "content": initial_prompt}]
                response = get_openai_response(messages, model_choice)

                if response and response.choices and response.choices[0].message:
                    assistant_text = response.choices[0].message.content or ""
                    st.session_state["messages"].append({"role": "assistant", "content": assistant_text})
                    message_placeholder.write(assistant_text)

            # Reflection and revision
            with st.spinner("Reflecting and revising..."):
                reflection_prompt = f"""You are a helpful AI assistant. You previously answered the question: {prompt} with the following answer: {assistant_text}.
                Now, reflect on your answer. What is missing? What is superfluous? Provide a critique of your answer.
                Then, based on your reflection, revise your answer to improve it.
                """
                messages = [{"role": "user", "content": reflection_prompt}]
                response = get_openai_response(messages, model_choice)

                if response and response.choices and response.choices[0].message:
                    revised_assistant_text = response.choices[0].message.content or ""
                    st.session_state["messages"].append({"role": "assistant", "content": revised_assistant_text})
                    message_placeholder.write(f"**Revised Answer:**\n{revised_assistant_text}")

if __name__ == "__main__":
    main()