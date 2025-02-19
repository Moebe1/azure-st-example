import streamlit as st
from openai import AzureOpenAI, OpenAIError
import re
from pydantic import BaseModel, Field
import logging
import numexpr
from bs4 import BeautifulSoup
import requests
import time
from langchain_community.tools.tavily_search import TavilySearchResults

# Define iteration limits
answer_iterations_limit = 3
reflection_iterations_limit = 2
#max_iterations = answer_iterations_limit + reflection_iterations_limit # Total max iterations
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
@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_cached_openai_response(messages_str, model_name):
    """
    Cached wrapper for OpenAI API calls.
    """
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=eval(messages_str),  # Convert string back to list of dicts
            stream=False,
            tools=tools,
            tool_choice="auto"
        )
        return response
    except OpenAIError as e:
        st.error(f"OpenAI API Error: {str(e)}")
        return None

def get_openai_response(messages, model_name, use_revise_answer=False):
    """
    Gets a response from the OpenAI API with caching.
    """
    # Initialize response cache if not exists
    if "response_cache" not in st.session_state:
        st.session_state.response_cache = {}
        
    prompt = f"""You are a helpful AI assistant. You have access to the following tools:
    - tavily_search_results_json: Searches the web and returns results. Only use this for questions requiring current or factual information.
    - AnswerQuestion: Provides an answer to the question.
    - ReviseAnswer: Revises the original answer to the question.
    
    IMPORTANT SEARCH GUIDELINES:
    1. Only use search for questions requiring current information, facts, statistics, or specific data.
    2. Avoid searching for general knowledge, concepts, or how-to explanations.
    3. When searching, use specific and focused queries (max 3) rather than broad ones.
    4. Reuse previous search results when possible instead of making new searches.
    
    After using any tool, synthesize the information into a complete and user-friendly answer using the AnswerQuestion or ReviseAnswer tool. When performing calculations, present each step clearly using LaTeX formatting. Always enclose ONLY the mathematical expressions in \(...\) to ensure proper rendering. Do not include any surrounding text within the \(...\) environment. For example:
    "Perform the multiplication: \( 15 \times 3.2 = 48 \)."
    "Perform the division: \( \frac{100}{4} = 25 \)."
    "Add the results: \( 48 + 25 = 73 \)."
    The final result is 73. It is crucial to always use the \(...\) format for ONLY the mathematical expressions.
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

    # Convert messages to string for caching
    messages_str = str(messages)
    cache_key = f"{messages_str}_{model_name}"

    # Check if we have a cached response
    if cache_key in st.session_state.response_cache:
        logging.info("Using cached OpenAI response")
        return st.session_state.response_cache[cache_key]

    # If not in cache, get new response
    try:
        response = get_cached_openai_response(messages_str, model_name)
        if response is None:
            logging.error("OpenAI API response is None")
            return None
            
        # Cache the response
        st.session_state.response_cache[cache_key] = response
        logging.info("Cached new OpenAI response")
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

def needs_search(question: str) -> bool:
    """
    Determines if a question requires web search based on its content.
    Returns True if the question likely needs current or factual information,
    False for questions that can be answered with general knowledge.
    """
    # Questions that likely don't need search
    no_search_patterns = [
        r"^how (do|does|can|would|should)",  # How-to questions
        r"^what (is|are) the difference",     # Definition/comparison questions
        r"^explain",                          # Explanatory questions
        r"^describe",                         # Descriptive questions
        r"^calculate",                        # Mathematical questions
        r"\d[\+\-\*\/\=]",                   # Mathematical expressions
    ]
    
    # Questions that likely need search
    search_patterns = [
        r"latest|recent|current|new|update",  # Time-sensitive information
        r"(in|for|during) \d{4}",            # Specific year references
        r"statistics|data|numbers",           # Data-driven questions
        r"price|cost|market",                # Market-related information
        r"news|event",                       # Current events
    ]
    
    # Check if question matches no-search patterns
    for pattern in no_search_patterns:
        if re.search(pattern, question.lower()):
            logging.info(f"Question matches no-search pattern: {pattern}")
            return False
            
    # Check if question matches search patterns
    for pattern in search_patterns:
        if re.search(pattern, question.lower()):
            logging.info(f"Question matches search pattern: {pattern}")
            return True
            
    # Default to True to maintain existing behavior for unmatched patterns
    return True

def process_response(response, user_question, model_choice, status_placeholder):
    logging.info("Entering process_response function") # ADDED LOGGING
    assistant_text = ""
    iteration = 0
    use_revise_answer = False
    max_iterations = st.session_state.get("max_iterations", 5) # Default to 5 if not set
    
    # Initialize search cache and timestamp in session state if not exists
    if "search_cache" not in st.session_state:
        st.session_state.search_cache = {}
        st.session_state.search_cache_timestamps = {}
    
    # Clear expired cache entries (older than 1 hour)
    current_time = time.time()
    expired_queries = [
        query for query, timestamp in st.session_state.search_cache_timestamps.items()
        if current_time - timestamp > 3600  # 1 hour TTL
    ]
    for query in expired_queries:
        del st.session_state.search_cache[query]
        del st.session_state.search_cache_timestamps[query]
        logging.info(f"Cleared expired cache entry for query: {query}")

    while iteration < max_iterations:
        if response and response.choices and response.choices[0].message:
            message = response.choices[0].message
            tool_calls = message.tool_calls or [] # Handle case with no tool_calls
            logging.info(f"LLM Response Content (Iteration {iteration + 1}): {message.content}")
            logging.info(f"Tool Calls: {tool_calls}") # ADDED LOGGING

            search_queries = [] # Collect search queries
            latex_pattern = r"\\\((.*?)\\\)" # Improved regex to correctly capture LaTeX expressions
            if tool_calls:
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

                # Check if search is needed and limit number of searches
                if search_queries and needs_search(user_question):
                    combined_content = ""
                    # Batch process queries and limit to 3 queries
                    unique_queries = list(set(search_queries))[:3]  # Remove duplicates and limit to 3 queries
                    logging.info(f"Processing {len(unique_queries)} unique search queries")
                    for query in unique_queries:
                        try:
                            # Check cache first
                            if query in st.session_state.search_cache:
                                search_results = st.session_state.search_cache[query]
                                logging.info(f"Cache hit for query: {query}")
                            else:
                                search_results = tavily_search.run(query)
                                # Cache the results with timestamp
                                if search_results and isinstance(search_results, list):
                                    st.session_state.search_cache[query] = search_results
                                    st.session_state.search_cache_timestamps[query] = time.time()
                                    logging.info(f"Cached results for query: {query} at {time.strftime('%Y-%m-%d %H:%M:%S')}")

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
    
    # Use st.latex to format mathematical expressions
    if assistant_text:
        latex_matches = re.findall(latex_pattern, assistant_text)
        for match in latex_matches:
            try:
                if match[0].startswith(r"\\["):
                    # Handle \[...\] format
                    latex_expression = match[0][1:-1]  # Remove \[ and \]
                    latex_expression = r"\(" + latex_expression + r"\)"  # Convert to \(...\)
                    sanitized_match = latex_expression.strip()
                    st.latex(sanitized_match)
                else:
                    # Handle \(...\) format
                    sanitized_match = match[0].strip()
                    st.latex(sanitized_match)
            except Exception as e:
                logging.error(f"Error formatting latex: {str(e)}")
    return assistant_text

# =============================================================================
# Main Streamlit App
# =============================================================================
def main():
    """
    Note: If you encounter "inotify watch limit reached" error on Linux:
    1. Check current limits: `cat /proc/sys/fs/inotify/max_user_watches`
    2. Increase the limit temporarily: `sudo sysctl fs.inotify.max_user_watches=524288`
    3. Make it permanent: Add `fs.inotify.max_user_watches=524288` to /etc/sysctl.conf
    """
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

        max_iterations = st.slider("Max Iterations:", min_value=1, max_value=20, value=5, step=1)
        st.session_state["max_iterations"] = max_iterations

        if st.button("Clear Conversation"):
            st.session_state["messages"] = [
                {"role": "assistant", "content": "Conversation cleared. How can I help you now?"}
            ]

    for i, msg in enumerate(st.session_state["messages"]):
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    # After rendering all messages, create a single chat input
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
        if st.session_state["messages"] and st.session_state["messages"][-1]["role"] != "user":
            st.session_state["messages"].append({"role": "assistant", "content": ""})

        with st.spinner("Thinking..."):
            if prompt and prompt.strip():
                messages = st.session_state["messages"]
                status_placeholder.text("Generating response...")
                logging.info("Calling get_openai_response") # ADDED LOGGING
                response = get_openai_response(messages, model_choice)
                logging.info("Returned from get_openai_response, calling process_response") # ADDED LOGGING
                assistant_text = process_response(response, prompt, model_choice, status_placeholder)
                st.session_state["messages"][-1]["content"] = assistant_text
            else:
                assistant_text = ""

            if assistant_text:
                message_placeholder.markdown(assistant_text)
                status_placeholder.empty() # Clear the status message
            else:
                message_placeholder.empty()
                status_placeholder.empty()

        # Display reflections in an expander, outside the chat message
        if len(st.session_state["reflections"]) > 0:
            with st.expander("Reflections"):
                reflections_output = ""
                for i, r in enumerate(st.session_state["reflections"], start=1):
                    reflections_output += f"**Reflection {i}:**  \n- Missing: {r.get('missing','')}  \n- Superfluous: {r.get('superfluous','')}  \n---  \n"
                # Add reflections to the chat with markdown
                st.markdown(reflections_output)
                logging.info(f"Reflections Output: {reflections_output}") # ADDED LOGGING

if __name__ == "__main__":
    main()
