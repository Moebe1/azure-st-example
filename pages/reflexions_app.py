import streamlit as st
from openai import AzureOpenAI, OpenAIError
import re
from pydantic import BaseModel, Field
import logging
import numexpr
from bs4 import BeautifulSoup
import requests
from langchain_community.tools.tavily_search import TavilySearchResults

# Define iteration limits
answer_iterations_limit = 3
reflection_iterations_limit = 2
max_iterations = answer_iterations_limit + reflection_iterations_limit # Total max iterations
iteration_type = "answer" # Start with answer generation

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
    Gets a response from the OpenAI API.
    """
    prompt = f"""You are a helpful AI assistant. You have access to the following tools:
    - tavily_search_results_json: Searches the web and returns results.
    - AnswerQuestion: Provides an answer to the question.
    - ReviseAnswer: Revises the original answer to the question.
    You must use the tavily_search_results_json tool to answer the question. After using the tool, you MUST synthesize the information into a complete and user-friendly answer using the AnswerQuestion or ReviseAnswer tool.
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
        },
        {
            "type": "function",
            "function": {
                "name": "AnswerQuestion",
                "description": "Answer the question. Provide an answer and reflection.",
                "parameters": AnswerQuestion.model_json_schema(),
            }
        },
        {
            "type": "function",
            "function": {
                "name": "ReviseAnswer",
                "description": "Revise your original answer to your question. Provide an answer and reflection.",
                "parameters": ReviseAnswer.model_json_schema(),
            }
        }
    ]

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            stream=False,
            tools=tools,
            tool_choice="auto"
        )
        logging.info(f"Raw OpenAI API Response: {response.model_dump_json()}") # FIXED LOGGING: Use model_dump_json()
        if response is None: # ADDED LOGGING
            logging.error("OpenAI API response is None") # ADDED LOGGING
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

def process_response(response, user_question, model_choice, status_placeholder):
    logging.info("Entering process_response function") # ADDED LOGGING
    assistant_text = ""
    iteration = 0
    use_revise_answer = False

    while iteration < max_iterations:
        if response and response.choices and response.choices[0].message:
            message = response.choices[0].message
            tool_calls = message.tool_calls or [] # Handle case with no tool_calls
            logging.info(f"LLM Response Content (Iteration {iteration + 1}): {message.content}")
            logging.info(f"Tool Calls: {tool_calls}") # ADDED LOGGING

            if tool_calls:
                search_queries = [] # Collect search queries
                
                for tool_call in tool_calls: # Iterate through all tool calls
                    function_name = tool_call.function.name
                    function_args = tool_call.function.arguments
                    logging.info(f"Function Name: {function_name}, Arguments: {function_args}")
                    status_placeholder.text(f"Using tool: {function_name}")

                    if function_name == "AnswerQuestion":
                        logging.info(f"AnswerQuestion function_args: {function_args}") # ADDED LOGGING
                        answer_data = AnswerQuestion.model_validate_json(function_args)
                        logging.info(f"AnswerQuestion answer_data: {answer_data}") # ADDED LOGGING
                        assistant_text = answer_data.answer
                        reflection = answer_data.reflection.dict()
                        st.session_state["reflections"].append(reflection)

                    elif function_name == "ReviseAnswer":
                        status_placeholder.text(f"Using tool: {function_name}")
                        logging.info(f"ReviseAnswer function_args: {function_args}") # ADDED LOGGING
                        answer_data = ReviseAnswer.model_validate_json(function_args)
                        logging.info(f"ReviseAnswer answer_data: {answer_data}") # ADDED LOGGING
                        assistant_text = answer_data.answer
                        reflection = answer_data.reflection.dict()
                        st.session_state["reflections"].append(reflection)
                    
                    elif function_name == "tavily_search_results_json":
                        status_placeholder.text(f"Using tool: {function_name}")
                        try:
                            query = eval(function_args)['query']
                            search_queries.append(query)
                        except Exception as e:
                            assistant_text += f"\n\nError processing tool call: {function_name} - {str(e)}"
                            logging.error(f"Error processing tool call: {function_name} - {str(e)}")
                            logging.exception(e) # ADDED LOGGING
                    else:
                        assistant_text += f"\n\nUnknown function: {function_name}"

                # Perform batched search if there are any search queries
                if search_queries:
                    combined_content = ""
                    for query in search_queries:
                        try:
                            search_results = tavily_search.run(query)
                            if search_results and isinstance(search_results, list):
                                # Concatenate the content of all search results
                                combined_content += "\n".join([result.get("content", "") for result in search_results if isinstance(result, dict)])
                                logging.info(f"Search Results for query '{query}': {combined_content}")
                            else:
                                assistant_text += f"\n\nCould not find relevant information in search results for query: {query}"
                        except Exception as e:
                            assistant_text += f"\n\nError during search for query '{query}': {str(e)}"
                            logging.error(f"Error during search for query '{query}': {str(e)}")

                    # Synthesize answer with combined search results
                    if combined_content:
                        messages = [
                            {"role": "user", "content": user_question},
                            {"role": "assistant", "content": f"Search results: {combined_content}"}
                        ]
                        try:
                            response = get_openai_response(messages, model_choice, use_revise_answer)
                            if response and response.choices and response.choices[0].message:
                                message = response.choices[0].message
                                tool_calls = message.tool_calls or [] # Handle case with no tool_calls
                                if tool_calls:
                                    for tool_call in tool_calls:
                                        logging.info(f"Tool Call Object: {tool_call}")
                                        logging.info(f"Tool Call Function Object: {tool_call.function}")
                                        function_name = tool_call.function.name
                                        function_args = tool_call.function.arguments
                                        status_placeholder.text(f"Using tool: {function_name}")

                                        if function_name == "tavily_search_results_json":
                                            try:
                                                query = eval(function_args)['query']
                                                search_queries.append(query)
                                            except Exception as e:
                                                assistant_text += f"\n\nError processing tool call: {function_name} - {str(e)}"
                                                logging.error(f"Error processing tool call: {function_name} - {str(e)}")
                                                logging.exception(e) # ADDED LOGGING
                                        elif function_name == "AnswerQuestion":
                                            status_placeholder.text(f"Using tool: {function_name}")
                                            answer_data = AnswerQuestion.model_validate_json(function_args)
                                            assistant_text = answer_data.answer
                                            reflection = answer_data.reflection.dict()
                                            st.session_state["reflections"].append(reflection)
                                        elif function_name == "ReviseAnswer":
                                            status_placeholder.text(f"Using tool: {function_name}")
                                            answer_data = ReviseAnswer.model_validate_json(function_args)
                                            assistant_text = answer_data.answer
                                            reflection = answer_data.reflection.dict()
                                            st.session_state["reflections"].append(reflection)
                                        else:
                                            assistant_text += f"\n\nUnknown function: {function_name}"

                                else:
                                    assistant_text = message.content or "Could not synthesize information from search results."
                            else:
                                assistant_text = "Could not synthesize the information."
                        except Exception as e:
                            assistant_text += f"\n\nError synthesizing answer: {str(e)}"
                            logging.error(f"Error synthesizing answer: {str(e)}")
                    else:
                        assistant_text += "\n\nCould not find relevant information in search results."
            else:
                assistant_text = message.content or ""

            iteration += 1
            if iteration < max_iterations:
                # Determine iteration type and update use_revise_answer accordingly
                if iteration <= answer_iterations_limit:
                    iteration_type = "answer"
                    use_revise_answer = True # Use ReviseAnswer for answer revisions
                else:
                    iteration_type = "reflection"
                    use_revise_answer = False # Do not use ReviseAnswer for reflection iterations, if needed

                # Prepare for the next iteration
                messages = st.session_state["messages"] + [{"role": "assistant", "content": assistant_text}]
                response = get_openai_response(messages, model_choice, use_revise_answer=use_revise_answer)
        else:
            assistant_text = "Could not get a valid response from the model."
            break

    if iteration == max_iterations:
        logging.warning("Maximum iterations reached.")

    logging.info(f"Assistant Text before return: {assistant_text}") # FIXED LOGGING
    if not assistant_text:
        assistant_text = "Reached maximum iterations without a final answer."
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

    for i, msg in enumerate(st.session_state["messages"]):
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    # After rendering all messages, create a single chat input
    with st.expander("Send a message", expanded=True):
        prompt = st.chat_input("Type your message here...", key="main_chat_input")
        if prompt:
            st.session_state["messages"].append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.write(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            status_placeholder = st.empty() # For ephemeral status updates
            assistant_text = ""

            # Initial response
            with st.spinner("Thinking..."):
                if prompt and prompt.strip():
                    messages = st.session_state["messages"]
                    status_placeholder.text("Generating response...")
                    logging.info("Calling get_openai_response") # ADDED LOGGING
                    response = get_openai_response(messages, model_choice)
                    logging.info("Returned from get_openai_response, calling process_response") # ADDED LOGGING
                    assistant_text = process_response(response, prompt, model_choice, status_placeholder)
                else:
                    assistant_text = ""

                st.session_state["messages"].append({"role": "assistant", "content": assistant_text})
                if assistant_text:
                    message_placeholder.markdown(assistant_text)
                    status_placeholder.empty() # Clear the status message

        # Display reflections in an expander, outside the chat message
        if len(st.session_state["reflections"]) > 0:
            with st.expander("Reflections"):
                reflections_output = ""
                for i, r in enumerate(st.session_state["reflections"], start=1):
                    reflections_output += f"**Reflection {i}:**  \n- Missing: {r.get('missing','')}  \n- Superfluous: {r.get('superfluous','')}  \n---  \n"
                # Add reflections to the chat with markdown
                st.markdown(reflections_output)
                logging.info(f"Reflections Output: {reflections_output}") # ADDED LOGGING

    # Reflection and revision (restored from reference)
    st.markdown("### Reflections")

if __name__ == "__main__":
    main()
