import streamlit as st
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.output_parsers.openai_tools import PydanticToolsParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from pydantic import BaseModel, Field, ValidationError
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper
import datetime
import re

# =============================================================================
# Configuration - LangGraph Reflexions Agent
# =============================================================================
TAVILY_API_KEY = st.secrets["TAVILY_API_KEY"]
search = TavilySearchAPIWrapper(api_key=TAVILY_API_KEY)
tavily_tool = TavilySearchResults(api_wrapper=search, max_results=5)

llm = ChatAnthropic(model="claude-3-5-sonnet-20240620")

class Reflection(BaseModel):
    missing: str = Field(description="Critique of what is missing.")
    superfluous: str = Field(description="Critique of what is superfluous.")

class AnswerQuestion(BaseModel):
    answer: str = Field(description="~250 word detailed answer to the question.")
    reflection: Reflection = Field(description="Your reflection on the initial answer.")
    search_queries: list[str] = Field(description="1-3 search queries for researching improvements.")

class ReviseAnswer(AnswerQuestion):
    references: list[str] = Field(description="Citations motivating your updated answer.")

actor_prompt_template = ChatPromptTemplate.from_messages([
    ("system", """You are an expert researcher. Current time: {time}
1. {first_instruction}
2. Reflect and critique your answer. Be severe to maximize improvement.
3. Recommend search queries to research information and improve your answer."""),
    MessagesPlaceholder(variable_name="messages"),
    ("user", "Reflect on the user's original question and the actions taken thus far.")
]).partial(time=lambda: datetime.datetime.now().isoformat())

initial_answer_chain = actor_prompt_template.partial(
    first_instruction="Provide a detailed ~250 word answer.",
    function_name=AnswerQuestion.__name__,
) | llm.bind_tools(tools=[AnswerQuestion])

revision_chain = actor_prompt_template.partial(
    first_instruction="Revise your original answer using the new information.",
    function_name=ReviseAnswer.__name__,
) | llm.bind_tools(tools=[ReviseAnswer])

validator = PydanticToolsParser(tools=[AnswerQuestion, ReviseAnswer])

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

            if streaming_enabled:
                with st.spinner("Thinking..."):
                    response_generator = initial_answer_chain.invoke({"messages": st.session_state["messages"]})
                    for update in response_generator:
                        assistant_text += update
                        message_placeholder.write(assistant_text)
            else:
                with st.spinner("Thinking..."):
                    response = initial_answer_chain.invoke({"messages": st.session_state["messages"]})
                    assistant_text = response["answer"]
                    message_placeholder.write(assistant_text)

            assistant_text = re.sub(r'[ \t]+$', '', assistant_text, flags=re.MULTILINE)
            assistant_text = re.sub(r'^\s*\n', '', assistant_text)
            assistant_text = re.sub(r'\n\s*$', '', assistant_text)

        st.session_state["messages"].append({"role": "assistant", "content": assistant_text})

        if token_counting_enabled and usage_info:
            st.write(f"Tokens Used: {usage_info}")

if __name__ == "__main__":
    main()