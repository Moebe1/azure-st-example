import streamlit as st
from openai import AzureOpenAI, OpenAIError
import re
from pydantic import BaseModel, Field
import logging
import numexpr
from bs4 import BeautifulSoup
import requests
from langchain_community.tools.tavily_search import TavilySearchResults

# Define the maximum number of iterations for improvement
max_iterations = 3

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

class ReviseAnswer(AnswerQuestion):
    """Revise your original answer to your question. Provide an answer and reflection."""
    pass

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
def get_openai_response(messages, model_name, use_revise_answer=False):
    """
    Gets a response from the OpenAI API, using the AnswerQuestion or ReviseAnswer tool.
    """
    prompt = f"""You are a helpful AI assistant. You have access to the following tool:
    - tavily_search_results_json: Searches the web and returns results.
    You must use the tool to answer the question. After using the tool, you MUST synthesize the information into a complete and user-friendly answer.
    """
    messages = [{"role": "system", "content": prompt}] + messages

    tools = [
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
    ]

    if use_revise_answer:
        function_name = "ReviseAnswer"
        function_description = "Revise your original answer to your question. Provide an answer and reflection."
        parameters = ReviseAnswer.model_json_schema()
    else:
        function_name = "AnswerQuestion"
        function_description = "Answer the question. Provide an answer and reflection."
        parameters = AnswerQuestion.model_json_schema()

    function_tool = {
        "type": "function",
        "function": {
            "name": function_name,
            "description": function_description,
            "parameters": parameters,
        }
    }
    tools.append(function_tool)

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            stream=False,
            tools=tools,
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

def process_response(response, user_question, model_choice):
    assistant_text = ""
    iteration = 0
    use_revise_answer = False

    while iteration < max_iterations:
        if response and response.choices and response.choices[0].message:
            message = response.choices[0].message
            tool_calls = message.tool_calls
            logging.info(f"LLM Response Content (Iteration {iteration + 1}): {message.content}")

            if tool_calls:
                tool_call = tool_calls[0]
                function_name = tool_call.function.name
                function_args = tool_call.function.arguments
                logging.info(f"Function Name: {function_name}, Arguments: {function_args}")

                try:
                    if function_name == "AnswerQuestion":
                        answer_data = AnswerQuestion.model_validate_json(function_args)
                        assistant_text = answer_data.answer
                        reflection = answer_data.reflection.dict()
                        st.session_state["reflections"].append(reflection)

                    elif function_name == "ReviseAnswer":
                        answer_data = ReviseAnswer.model_validate_json(function_args)
                        assistant_text = answer_data.answer
                        reflection = answer_data.reflection.dict()
                        st.session_state["reflections"].append(reflection)
                    
                    elif function_name == "tavily_search_results_json":
                        try:
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
                                response = get_openai_response(
                                    messages, model_choice, use_revise_answer
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
                    else:
                        assistant_text += f"\n\nUnknown function: {function_name}"
                except Exception as e:
                    assistant_text += f"\n\nError processing tool call: {str(e)}"
                    logging.error(f"Error processing tool call: {str(e)}")
            else:
                assistant_text = message.content or ""

            iteration += 1
            if iteration < max_iterations:
                # Prepare for the next iteration, using ReviseAnswer
                messages = st.session_state["messages"] + [{"role": "assistant", "content": assistant_text}]
                response = get_openai_response(messages, model_choice, use_revise_answer=True)
                use_revise_answer = True # Ensure ReviseAnswer is used in subsequent iterations
        else:
            assistant_text = "Could not get a valid response from the model."
            break

    if iteration == max_iterations:
        logging.warning("Maximum iterations reached.")

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
        st.session_state["reflections"] = []

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
                assistant_text = process_response(response, prompt, model_choice)
 
                st.session_state["messages"].append({"role": "assistant", "content": assistant_text})
                message_placeholder.markdown(assistant_text)

            # Reflection and revision (removed for simplicity)

if __name__ == "__main__":
    main()