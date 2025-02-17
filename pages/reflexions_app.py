import streamlit as st
from openai import AzureOpenAI, OpenAIError
import re
from pydantic import BaseModel, Field
import logging
import numexpr
from bs4 import BeautifulSoup
import requests
from langchain_community.tools.tavily_search import TavilySearchResults

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
TAVILY_API_KEY = st.secrets["TAVILY_API_KEY"]

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
# Tool Definitions
# =============================================================================
def calculate(expression: str) -> str:
    """
    Evaluates a mathematical expression and returns the result.
    """
    try:
        result = numexpr.evaluate(expression)
        return str(result)
    except Exception as e:
        return f"Error: {str(e)}"

def summarize_document(url: str) -> str:
    """
    Summarizes the content of a document at a given URL.
    """
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
        soup = BeautifulSoup(response.content, "html.parser")
        text = ' '.join(soup.stripped_strings)

        # Summarize the text using OpenAI
        messages = [{"role": "user", "content": f"Summarize the following text:\n{text}"}]
        summary_response = client.chat.completions.create(
            model="gpt-4o",  # Or another suitable model
            messages=messages,
            stream=False
        )
        if summary_response and summary_response.choices and summary_response.choices[0].message:
            return summary_response.choices[0].message.content or ""
        else:
            return "Could not summarize the document."
    except requests.exceptions.RequestException as e:
        return f"Error fetching URL: {str(e)}"
    except Exception as e:
        return f"Error summarizing document: {str(e)}"

tavily_search = TavilySearchResults(api_key=TAVILY_API_KEY)

# =============================================================================
# Reflexion Actor Logic
# =============================================================================
def get_openai_response(messages, model_name):
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            stream=False,
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "tavily_search_results_json",
                        "description": "Useful for when you need to answer questions about current events. Input should be a search query.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "query": {
                                    "type": "string",
                                    "description": "The search query to use."
                                }
                            },
                            "required": ["query"]
                        }
                    }
                }
            ],
            tool_choice="auto"
        )
        return response
    except OpenAIError as e:
        st.error(f"OpenAI API Error: {str(e)}")
        return None

def process_response(response):
    assistant_text = ""
    if response and response.choices and response.choices[0].message:
        message = response.choices[0].message
        assistant_text = message.content or ""
        logging.info(f"LLM Response Content: {assistant_text}")

        # Check for tool calls
        if message.tool_calls:
            logging.info(f"Tool Calls: {message.tool_calls}")
            for tool_call in message.tool_calls:
                function_name = tool_call.function.name
                function_args = tool_call.function.arguments
                logging.info(f"Function Name: {function_name}, Arguments: {function_args}")

                try:
                    if function_name == "tavily_search_results_json":
                        query = eval(function_args)['query']
                        search_results = tavily_search.run(query)
                        if search_results and isinstance(search_results, list):
                            # Concatenate the content of all search results
                            combined_content = "\n".join([result.get("content", "") for result in search_results if isinstance(result, dict)])
                            assistant_text = f"Search results: {combined_content}"
                        else:
                            assistant_text = "\n\nCould not find relevant information in search results."
                except Exception as e:
                    assistant_text += f"\n\nError processing tool call: {function_name} - {str(e)}"
                    logging.error(f"Error processing tool call: {function_name} - {str(e)}")
    return assistant_text

# =============================================================================
# Main Streamlit App
# =============================================================================
def main():
    st.set_page_config(page_title="Reflexions Multi-Tool Agent", page_icon="🤖")
    st.title("Reflexions Multi-Tool Agent")

    # Configure logging
    logging.basicConfig(level=logging.INFO)

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
                initial_prompt = f"""You are a helpful AI assistant. You have access to the following tool:
                - tavily_search_results_json: Searches the web and returns results.

                When the user asks a question, use the tavily_search_results_json tool to search for relevant information.
                Then, based on the search results, provide a concise and accurate answer to the user's question.
                """
                messages = [{"role": "user", "content": initial_prompt}]
                response = get_openai_response(messages, model_choice)
                assistant_text = process_response(response)

                st.session_state["messages"].append({"role": "assistant", "content": assistant_text})
                message_placeholder.write(assistant_text)

            # Reflection and revision (removed for simplicity)

if __name__ == "__main__":
    main()