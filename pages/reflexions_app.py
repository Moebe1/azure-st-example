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
    prompt = """You are a helpful AI assistant. You have access to the following tool:
    - tavily_search_results_json: Searches the web and returns results.

    When the user asks a question, use the tavily_search_results_json tool to search for relevant information.
    Then, based on the search results, provide a concise and accurate answer to the user's question.
    It is very important that you synthesize the information from the tool calls into a complete and user-friendly answer.
    Do not just execute tools, but also process and present the findings in a clear and concise manner.

    For example, if the user asks: "What is the currency used in France? What is the current exchange rate of 1 unit of that currency to Australian Dollars?",
    you should use the tavily_search_results_json tool to search for the currency used in France, and then use the tavily_search_results_json tool again to search for the exchange rate of that currency to Australian Dollars.
    Then, you should synthesize the information into a complete and user-friendly answer, such as: "The currency used in France is the Euro (EUR). The current exchange rate is 1 EUR = 1.65 Australian Dollars."

    If the search results are ambiguous or contradictory, you should acknowledge the uncertainty and explain the conflicting information to the user, rather than simply presenting the raw data.

    Remember, you are a helpful assistant that provides complete and understandable answers, not just raw data.

    The final answer should be in the following format: "The currency used in France is [Currency Name]. The current exchange rate is 1 [Currency Name] = [Exchange Rate] Australian Dollars."

    Optimize the prompt for improved reasoning and information extraction from the search results, focusing on identifying the relevant information and discarding irrelevant details.
    """
    messages = [{"role": "system", "content": prompt}] + messages
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

def validate_response(response_content):
    """
    Validates the response content against task-specific criteria using the gpt-4o-mini model.
    Returns a tuple (is_valid, feedback) where is_valid is a boolean
    and feedback is a string describing any issues.
    """
    if not response_content:
        return False, "Response is empty."
    if len(response_content) > 1000:
        return False, "Response exceeds the maximum allowed length."

    # Use gpt-4o-mini for external evaluation
    try:
        evaluation_prompt = f"Evaluate the following response for correctness and completeness:\n\n{response_content}"
        evaluation_response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": evaluation_prompt}],
            stream=False
        )
        if evaluation_response and evaluation_response.choices and evaluation_response.choices[0].message:
            feedback = evaluation_response.choices[0].message.content or "No feedback provided."
            if "valid" in feedback.lower():
                return True, feedback
            else:
                return False, feedback
        else:
            return False, "External evaluator did not return a valid response."
    except OpenAIError as e:
        return False, f"Error during external evaluation: {str(e)}"

def process_response(response, user_question):
    assistant_text = ""
    max_iterations = 3  # Define the maximum number of iterations for improvement
    iteration = 0

    while iteration < max_iterations:
        if response and response.choices and response.choices[0].message:
            message = response.choices[0].message
            assistant_text = message.content or ""
            logging.info(f"LLM Response Content (Iteration {iteration + 1}): {assistant_text}")

            # Validate the response
            is_valid, feedback = validate_response(assistant_text)
            if is_valid:
                logging.info("Validation successful.")
                break  # Exit the loop if the response is valid
            else:
                logging.warning(f"Validation failed: {feedback}")
                assistant_text += f"\n\nValidation Feedback: {feedback}"

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
                                # Include the user's question and search results in the messages sent to the OpenAI API
                                messages = [
                                    {"role": "user", "content": user_question},
                                    {"role": "assistant", "content": f"Search results: {combined_content}"}
                                ]
                                response = client.chat.completions.create(
                                    model="gpt-4o",  # Or another suitable model
                                    messages=messages,
                                    stream=False
                                )
                                if response and response.choices and response.choices[0].message:
                                    assistant_text = response.choices[0].message.content or ""
                                else:
                                    assistant_text = "Could not synthesize the information."
                            else:
                                assistant_text = "\n\nCould not find relevant information in search results."
                    except Exception as e:
                        assistant_text += f"\n\nError processing tool call: {function_name} - {str(e)}"
                        logging.error(f"Error processing tool call: {function_name} - {str(e)}")
        iteration += 1

    if iteration == max_iterations:
        logging.warning("Maximum iterations reached without a valid response.")
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
                messages = st.session_state["messages"]
                response = get_openai_response(messages, model_choice)
                assistant_text = process_response(response, prompt)

                st.session_state["messages"].append({"role": "assistant", "content": assistant_text})
                message_placeholder.write(assistant_text)

            # Reflection and revision (removed for simplicity)

if __name__ == "__main__":
    main()