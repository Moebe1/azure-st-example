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
import json
from typing import List, Dict, Any, Optional

class BraveSearchResults:
    """Tool that queries the Brave Search API."""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.headers = {
            "X-Subscription-Token": api_key,
            "Accept": "application/json",
        }
        self.base_url = "https://api.search.brave.com/res/v1/web/search"

    def run(self, query: str) -> List[Dict[str, Any]]:
        """Run query through Brave Search and return results."""
        try:
            response = requests.get(
                self.base_url,
                headers=self.headers,
                params={"q": query}
            )
            response.raise_for_status()
            data = response.json()
            
            # Extract and format web results
            results = []
            if "web" in data and "results" in data["web"]:
                for result in data["web"]["results"]:
                    results.append({
                        "title": result.get("title", ""),
                        "url": result.get("url", ""),
                        "content": result.get("description", ""),
                        "score": 1.0  # Brave doesn't provide a score, using default
                    })
            return results
        except Exception as e:
            logging.error(f"Brave Search API error: {str(e)}")
            return None

# Define iteration limits
answer_iterations_limit = 3
reflection_iterations_limit = 2
iteration_type = "answer"  # Start with answer generation

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
BRAVE_SEARCH_API_KEY = st.secrets["BRAVE_SEARCH_API_KEY"]

# List of deployments you have in Azure OpenAI
AVAILABLE_MODELS = [
    "o1-mini",
    "gpt-4o",
    "gpt-4o-mini"
]

# Create an Azure OpenAI client
client = AzureOpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_version=AZURE_OPENAI_API_VERSION
)

def get_search_tool():
    """Returns the appropriate search tool based on user selection."""
    search_provider = st.session_state.get("search_provider", "brave")
    
    if search_provider == "brave":
        return BraveSearchResults(api_key=BRAVE_SEARCH_API_KEY)
    elif search_provider == "tavily":
        return tavily_search
    else:
        st.error("Invalid search provider selected. Defaulting to Tavily.")
        return tavily_search

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
    """Evaluates a mathematical expression and returns the result."""
    try:
        result = numexpr.evaluate(expression)
        return str(result)
    except Exception as e:
        return f"Error: {str(e)}"

def summarize_document(url: str) -> str:
    """Summarizes the content of a document at a given URL."""
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, "html.parser")
        text = ' '.join(soup.stripped_strings)
        messages = [{"role": "user", "content": f"Summarize the following text:\n{text}"}]
        summary_response = client.chat.completions.create(
            model="gpt-4o",
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
def get_cached_openai_response(messages_str, model_name, tools_config):
    """Cached wrapper for OpenAI API calls."""
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=eval(messages_str),
            stream=False,
            tools=tools_config,
            tool_choice="auto"
        )
        return response
    except OpenAIError as e:
        st.error(f"OpenAI API Error: {str(e)}")
        return None

def get_openai_response(messages, model_name, use_revise_answer=False):
    """
    Gets a response from the OpenAI API with caching.
    Prevents searches during revision phase.
    """
    # Initialize response cache if not exists
    if "response_cache" not in st.session_state:
        st.session_state.response_cache = {}
        
    # Define base prompt
    prompt = f"""You are a helpful AI assistant. You have access to the following tools:
    - tavily_search_results_json: Web search using either Brave Search (default) or Tavily with automatic fallback. Only use this for questions requiring current or factual information.
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
    
    # Add revision-specific instructions if in revision mode
    if use_revise_answer:
        prompt += "\nYou are in revision mode. Focus on improving the existing answer without making new searches."
    
    messages = [{"role": "system", "content": prompt}] + messages

    # Define base tools (always available)
    tools = [
        {
            "type": "function",
            "function": {
                "name": "AnswerQuestion" if not use_revise_answer else "ReviseAnswer",
                "description": "Answer the question. Provide an answer and reflection.",
                "parameters": (ReviseAnswer if use_revise_answer else AnswerQuestion).model_json_schema(),
            }
        }
    ]
    
    # Only include search tool if not in revision phase
    if not use_revise_answer:
        tools.append({
            "type": "function",
            "function": {
                "name": "tavily_search_results_json",
                "description": "Web search using Brave Search (default) or Tavily with automatic fallback. Use for questions requiring current or factual information.",
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
        })

    # Convert messages to string for caching
    messages_str = str(messages)
    cache_key = f"{messages_str}_{model_name}_{use_revise_answer}"

    # Check if we have a cached response
    if cache_key in st.session_state.response_cache:
        logging.info("Using cached OpenAI response")
        return st.session_state.response_cache[cache_key]

    # If not in cache, get new response
    try:
        response = get_cached_openai_response(messages_str, model_name, tools)
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
    """Validates the response content against task-specific criteria."""
    if not response_content:
        return False, "Response is empty."
    if len(response_content) > 1000:
        return False, "Response exceeds the maximum allowed length."

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
    """Determines if a question requires web search based on its content."""
    no_search_patterns = [
        r"^how (do|does|can|would|should)",  # How-to questions
        r"^what (is|are) the difference",     # Definition/comparison questions
        r"^explain",                          # Explanatory questions
        r"^describe",                         # Descriptive questions
        r"^calculate",                        # Mathematical questions
        r"\d[\+\-\*\/\=]",                   # Mathematical expressions
    ]
    
    search_patterns = [
        r"latest|recent|current|new|update",  # Time-sensitive information
        r"(in|for|during) \d{4}",            # Specific year references
        r"statistics|data|numbers",           # Data-driven questions
        r"price|cost|market",                # Market-related information
        r"news|event",                       # Current events
    ]
    
    for pattern in no_search_patterns:
        if re.search(pattern, question.lower()):
            logging.info(f"Question matches no-search pattern: {pattern}")
            return False
            
    for pattern in search_patterns:
        if re.search(pattern, question.lower()):
            logging.info(f"Question matches search pattern: {pattern}")
            return True
            
    return True

def process_response(response, user_question, model_choice, status_placeholder):
    """Process the response and handle search queries efficiently."""
    logging.info("Entering process_response function")
    assistant_text = ""
    iteration = 0
    use_revise_answer = False
    max_iterations = st.session_state.get("max_iterations", 5)

    # Check if search is needed
    if not needs_search(user_question):
        st.info("⚠️ This query did not require a search. If you need real-time data, "
                "consider switching to Tavily in the sidebar.")
    
    # Initialize search tracking in session state
    if "search_cache" not in st.session_state:
        st.session_state.search_cache = {}
        st.session_state.search_cache_timestamps = {}
        st.session_state.search_requests_made = set()  # Track all searches made in session
    
    # Clear expired cache entries (older than 1 hour)
    current_time = time.time()
    expired_queries = [
        query for query, timestamp in st.session_state.search_cache_timestamps.items()
        if current_time - timestamp > 3600
    ]
    for query in expired_queries:
        del st.session_state.search_cache[query]
        del st.session_state.search_cache_timestamps[query]
        logging.info(f"Cleared expired cache entry for query: {query}")

    while iteration < max_iterations:
        if response and response.choices and response.choices[0].message:
            message = response.choices[0].message
            tool_calls = message.tool_calls or []
            logging.info(f"LLM Response Content (Iteration {iteration + 1}): {message.content}")
            
            search_queries = []
            if tool_calls:
                for tool_call in tool_calls:
                    function_name = tool_call.function.name
                    function_args = tool_call.function.arguments
                    logging.info(f"Function Name: {function_name}, Arguments: {function_args}")
                    status_placeholder.text(f"Using tool: {function_name}")

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
                        
                    elif function_name == "tavily_search_results_json" and not use_revise_answer:
                        try:
                            query = eval(function_args)['query']
                            search_queries.append(query)
                        except Exception as e:
                            assistant_text += f"\n\nError processing tool call: {function_name} - {str(e)}"
                            logging.error(f"Error processing tool call: {function_name} - {str(e)}")
                    else:
                        assistant_text += f"\n\nUnknown function: {function_name}"

                # Process search queries efficiently
                if search_queries and needs_search(user_question):
                    combined_content = ""
                    # Preserve order while getting unique queries, limited to first 3
                    seen = set()
                    unique_queries = [x for x in search_queries if not (x in seen or seen.add(x))][:3]
                    
                    for query in unique_queries:
                        # Skip if query was already made in this session
                        if query in st.session_state.search_requests_made:
                            if query in st.session_state.search_cache:
                                search_results = st.session_state.search_cache[query]
                                if search_results and isinstance(search_results, list):
                                    combined_content += "\n".join([result.get("content", "") for result in search_results])
                                logging.info(f"Using cached results for query: {query}")
                            continue
                        
                        try:
                            if query in st.session_state.search_cache:
                                search_results = st.session_state.search_cache[query]
                                logging.info(f"Cache hit for query: {query}")
                            else:
                                logging.info(f"Cache miss - making new request for query: {query}")
                                st.session_state.search_requests_made.add(query)
                                search_tool = get_search_tool()  # Get current search tool
                                search_results = search_tool.run(query)

                                if search_results and isinstance(search_results, list):
                                    st.session_state.search_cache[query] = search_results
                                    st.session_state.search_cache_timestamps[query] = time.time()
                                    st.session_state.search_requests_made.add(query)
                                else:
                                    raise ValueError("No valid search results returned.")
                            
                            if search_results and isinstance(search_results, list):
                                combined_content += "\n".join([result.get("content", "") for result in search_results])
                            else:
                                assistant_text += f"\n\nNo relevant information found for query: {query}"
                        except Exception as e:
                            logging.error(f"Error during search for query '{query}': {str(e)}")
                            
                            # Alert user and suggest switching providers
                            st.warning(f"Search provider '{st.session_state.search_provider}' failed. "
                                     "Try switching to Tavily in the sidebar settings.")
                            
                            assistant_text += f"\n\n⚠️ Search failed using {st.session_state.search_provider}. Try switching to Tavily."
                    
                    if combined_content:
                        messages = [
                            {"role": "user", "content": user_question},
                            {"role": "assistant", "content": f"Search results: {combined_content}"}
                        ]
                        response = get_openai_response(messages, model_choice, use_revise_answer=True)
            else:
                assistant_text = message.content or ""

            iteration += 1
            if iteration < max_iterations:
                if iteration <= answer_iterations_limit:
                    use_revise_answer = True
                else:
                    use_revise_answer = False

                messages = st.session_state["messages"] + [{"role": "assistant", "content": assistant_text}]
                response = get_openai_response(messages, model_choice, use_revise_answer=use_revise_answer)
        else:
            assistant_text = "Could not get a valid response from the model."
            break

    if iteration == max_iterations:
        logging.warning("Maximum iterations reached.")

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

    # Initialize session state variables
    if "messages" not in st.session_state:
        st.session_state["messages"] = [
            {"role": "assistant", "content": "Hello! How can I assist you today?"}
        ]
    
    if "search_provider" not in st.session_state:
        st.session_state["search_provider"] = "brave"  # Set default search provider
        
    st.session_state["reflections"] = []

    with st.sidebar:
        st.header("Configuration")
        model_choice = st.selectbox("Select the Azure deployment:", AVAILABLE_MODELS, index=0)
        
        # Search provider selection
        search_provider = st.selectbox(
            "Search Provider:",
            ["brave", "tavily"],
            index=0 if st.session_state.get("search_provider", "brave") == "brave" else 1,
            key="search_provider_select"
        )
        st.session_state.search_provider = search_provider
        
        max_iterations = st.slider("Max Iterations:", min_value=1, max_value=20, value=5, step=1)
        st.session_state["max_iterations"] = max_iterations

        if st.button("Clear Conversation"):
            st.session_state["messages"] = [
                {"role": "assistant", "content": "Conversation cleared. How can I help you now?"}
            ]

    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    prompt = st.chat_input("Type your message here...", key="main_chat_input")
    if prompt:
        st.session_state["messages"].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            status_placeholder = st.empty()
            assistant_text = ""

            if st.session_state["messages"][-1]["role"] != "user":
                st.session_state["messages"].append({"role": "assistant", "content": ""})

            with st.spinner("Thinking..."):
                if prompt.strip():
                    messages = st.session_state["messages"]
                    status_placeholder.text("Generating response...")
                    response = get_openai_response(messages, model_choice)
                    assistant_text = process_response(response, prompt, model_choice, status_placeholder)
                    st.session_state["messages"][-1]["content"] = assistant_text

                if assistant_text:
                    message_placeholder.markdown(assistant_text)
                    status_placeholder.empty()
                else:
                    message_placeholder.empty()
                    status_placeholder.empty()

            if st.session_state["reflections"]:
                with st.expander("Reflections"):
                    reflections_output = ""
                    for i, r in enumerate(st.session_state["reflections"], start=1):
                        reflections_output += f"**Reflection {i}:**\n- Missing: {r.get('missing','')}\n- Superfluous: {r.get('superfluous','')}\n---\n"
                    st.markdown(reflections_output)

if __name__ == "__main__":
    main()
