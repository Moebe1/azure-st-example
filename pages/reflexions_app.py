# import streamlit as st
# from openai import AzureOpenAI, OpenAIError
# import re
# from pydantic import BaseModel, Field
# import logging
# import numexpr
# from bs4 import BeautifulSoup
# import requests
# import time
# from langchain_community.tools.tavily_search import TavilySearchResults
# import json
# from typing import List, Dict, Any, Optional

# class BraveSearchResults:
#     """Tool that queries the Brave Search API with rate limiting (1 request per second)."""
    
#     def __init__(self, api_key: str):
#         self.api_key = api_key
#         self.headers = {
#             "X-Subscription-Token": api_key,
#             "Accept": "application/json",
#         }
#         self.base_url = "https://api.search.brave.com/res/v1/web/search"
#         self.last_request_time = 0  # Track the last request time (instance-specific)

#     def run(self, query: str) -> List[Dict[str, Any]]:
#         """Run query through Brave Search and return results with improved rate limiting."""
#         try:
#             backoff_time = 1.0  # Initial backoff time in seconds
#             max_retries = 5
#             retries = 0

#             while retries < max_retries:
#                 # Calculate time since last request
#                 current_time = time.time()
#                 time_since_last_request = current_time - self.last_request_time

#                 # If less than 1 second has passed, sleep for the remaining time
#                 if time_since_last_request < 0.833:
#                     time.sleep(0.833 - time_since_last_request)

#                 # Make the request
#                 response = requests.get(
#                     self.base_url,
#                     headers=self.headers,
#                     params={"q": query}
#                 )

#                 # Update last request time
#                 self.last_request_time = time.time()

#                 if response.status_code == 429:
#                     logging.warning("Brave Search API rate limit exceeded. Retrying...")
#                     time.sleep(backoff_time)
#                     backoff_time = min(backoff_time * 2, 10)  # Exponential backoff with a max cap of 10 seconds
#                     retries += 1
#                     continue

#                 response.raise_for_status()
#                 data = response.json()
#                 break  # Exit loop if request is successful
            
#             # Extract and format web results
#             results = []
#             if "web" in data and "results" in data["web"]:
#                 for result in data["web"]["results"]:
#                     results.append({
#                         "title": result.get("title", ""),
#                         "url": result.get("url", ""),
#                         "content": result.get("description", ""),
#                         "score": 1.0  # Brave doesn't provide a score, using default
#                     })
#             return results if results else []
#         except Exception as e:
#             logging.error(f"Brave Search API error: {str(e)}")
#             return []

# # Define iteration limits
# answer_iterations_limit = 3
# reflection_iterations_limit = 2
# iteration_type = "answer"  # Start with answer generation

# # =============================================================================
# # Suppress Streamlit Debug Messages
# # =============================================================================
# logging.getLogger("streamlit").setLevel(logging.ERROR)

# # =============================================================================
# # Configuration - Azure OpenAI
# # =============================================================================
# AZURE_OPENAI_API_KEY = st.secrets["AZURE_OPENAI_API_KEY"]
# AZURE_OPENAI_ENDPOINT = st.secrets["AZURE_OPENAI_ENDPOINT"]
# AZURE_OPENAI_API_VERSION = st.secrets["AZURE_OPENAI_API_VERSION"]
# TAVILY_API_KEY = st.secrets["TAVILY_API_KEY"]
# BRAVE_SEARCH_API_KEY = st.secrets["BRAVE_SEARCH_API_KEY"]

# # List of deployments you have in Azure OpenAI
# AVAILABLE_MODELS = [
#     "o1-mini",
#     "gpt-4o",
#     "gpt-4o-mini"
# ]

# # Create an Azure OpenAI client
# client = AzureOpenAI(
#     api_key=AZURE_OPENAI_API_KEY,
#     azure_endpoint=AZURE_OPENAI_ENDPOINT,
#     api_version=AZURE_OPENAI_API_VERSION
# )

# def get_search_tool():
#     """Returns the appropriate search tool based on user selection."""
#     search_provider = st.session_state.get("search_provider", "brave")

#     if search_provider == "brave":
#         logging.info("Using Brave Search")
#         return BraveSearchResults(api_key=BRAVE_SEARCH_API_KEY)
#     elif search_provider == "tavily":
#         logging.info("Using Tavily Search")
#         return tavily_search
#     else:
#         logging.error("Invalid search provider selected.")
#         return None

# # =============================================================================
# # Reflexions Agent Components
# # =============================================================================
# class Reflection(BaseModel):
#     missing: str = Field(description="Critique of what is missing.")
#     superfluous: str = Field(description="Critique of what is superfluous.")

# class AnswerQuestion(BaseModel):
#     answer: str = Field(description="~250 word detailed answer to the question.")
#     reflection: Reflection = Field(description="Your reflection on the initial answer.")

# class ReviseAnswer(AnswerQuestion):
#     """Revise your original answer to your question. Provide an answer and reflection."""
#     pass

# # =============================================================================
# # Tool Definitions
# # =============================================================================
# def calculate(expression: str) -> str:
#     """Evaluates a mathematical expression and returns the result."""
#     try:
#         result = numexpr.evaluate(expression)
#         return str(result)
#     except Exception as e:
#         return f"Error: {str(e)}"

# def summarize_document(url: str) -> str:
#     """Summarizes the content of a document at a given URL."""
#     try:
#         response = requests.get(url)
#         response.raise_for_status()
#         soup = BeautifulSoup(response.content, "html.parser")
#         text = ' '.join(soup.stripped_strings)
#         messages = [{"role": "user", "content": f"Summarize the following text:\n{text}"}]
#         summary_response = client.chat.completions.create(
#             model="gpt-4o",
#             messages=messages,
#             stream=False
#         )
#         if summary_response and summary_response.choices and summary_response.choices[0].message:
#             return summary_response.choices[0].message.content or ""
#         else:
#             return "Could not summarize the document."
#     except requests.exceptions.RequestException as e:
#         return f"Error fetching URL: {str(e)}"
#     except Exception as e:
#         return f"Error summarizing document: {str(e)}"

# tavily_search = TavilySearchResults(api_key=TAVILY_API_KEY)

# # =============================================================================
# # Reflexion Actor Logic
# # =============================================================================
# @st.cache_data(ttl=3600)  # Cache for 1 hour
# def get_cached_openai_response(messages_str, model_name, tools_config):
#     """Cached wrapper for OpenAI API calls."""
#     try:
#         response = client.chat.completions.create(
#             model=model_name,
#             messages=eval(messages_str),
#             stream=False,
#             tools=tools_config,
#             tool_choice= "required"
#         )
#         return response
#     except OpenAIError as e:
#         st.error(f"OpenAI API Error: {str(e)}")
#         return None

# def get_openai_response(messages, model_name, use_revise_answer=False):
#     cache_key = f"{str(messages)}_{model_name}_{use_revise_answer}"

#     if use_revise_answer:
#         # Invalidate cache for revised answers
#         st.session_state.response_cache.pop(cache_key, None)

#     if cache_key in st.session_state.response_cache:
#         logging.info("Using cached OpenAI response")
#         return st.session_state.response_cache[cache_key]

#     search_tool_name = "BraveSearchResults" if st.session_state.get("search_provider", "brave") == "brave" else "TavilySearchResults"
#     tools = [
#         {
#             "type": "function",
#             "function": {
#                 "name": "AnswerQuestion",
#                 "description": "Answer the question. Provide an answer and reflection.",
#                 "parameters": AnswerQuestion.model_json_schema(),
#             }
#         },
#         {
#             "type": "function",
#             "function": {
#                 "name": "ReviseAnswer",
#                 "description": "Refine and improve the response with additional details if needed.",
#                 "parameters": ReviseAnswer.model_json_schema(),
#             }
#         }
#     ]

#     # Conditionally add the search tool based on user selection
#     search_provider = st.session_state.get("search_provider", "brave")
#     print(f"Selected search provider: {search_provider}")
#     if search_provider == "tavily":
#         tools.append({
#             "type": "function",
#             "function": {
#                 "name": "TavilySearchResults",
#                 "description": "Retrieve web search results.",
#                 "parameters": {
#                     "type": "object",
#                     "properties": {
#                         "query": {
#                             "type": "string",
#                             "description": "The search query to use."
#                         }
#                     },
#                     "required": ["query"]
#                 }
#             }
#         })
#     else:
#         tools.append({
#             "type": "function",
#             "function": {
#                 "name": "BraveSearchResults",
#                 "description": "Retrieve web search results.",
#                 "parameters": {
#                     "type": "object",
#                     "properties": {
#                         "query": {
#                             "type": "string",
#                             "description": "The search query to use."
#                         }
#                     },
#                     "required": ["query"]
#                 }
#             }
#         })

#     print(f"Tools list: {tools}")
#     response = get_cached_openai_response(str(messages), model_name, tools)
#     if response:
#         st.session_state.response_cache[cache_key] = response
#         logging.info("Cached new OpenAI response")
#     return response

# def validate_response(response_content):
#     """Validates the response content against task-specific criteria."""
#     if not response_content:
#         return False, "Response is empty."
#     if len(response_content) > 1000:
#         return False, "Response exceeds the maximum allowed length."

#     try:
#         evaluation_prompt = f"Evaluate the following response for correctness and completeness:\n\n{response_content}"
#         evaluation_response = client.chat.completions.create(
#             model="gpt-4o-mini",
#             messages=[{"role": "user", "content": evaluation_prompt}],
#             stream=False
#         )
#         if evaluation_response and evaluation_response.choices and evaluation_response.choices[0].message:
#             feedback = evaluation_response.choices[0].message.content or "No feedback provided."
#             if "valid" in feedback.lower():
#                 return True, feedback
#             else:
#                 return False, feedback
#         else:
#             return False, "External evaluator did not return a valid response."
#     except OpenAIError as e:
#         return False, f"Error during external evaluation: {str(e)}"

# def needs_search(question: str) -> bool:
#     """Determines if a question requires web search based on its content."""
#     search_patterns = [
#         r"(latest|recent|current|new|update)",
#         r"(in|for|during) \d{4}",
#         r"statistics|data|numbers",
#         r"price|cost|market",
#         r"news|event",
#         r"breaking news",
#         r"confirmed reports",
#         r"weather in"
#     ]

#     if any(re.search(pattern, question.lower()) for pattern in search_patterns):
#         logging.info("Web search is needed for this query.")
#         return True

#     return True

# def process_response(response, user_question, model_choice, status_placeholder):
#     """Process the response and handle search queries efficiently."""
#     logging.info("Entering process_response function")
#     assistant_text = ""
#     iteration = 0
#     use_revise_answer = False
#     max_iterations = st.session_state.get("max_iterations", 10)

#     # Check if search is needed
#     if not needs_search(user_question):
#         st.info("⚠️ This query did not require a search. If you need real-time data, "
#                 "consider switching to Tavily in the sidebar.")
    
#     # Initialize search tracking in session state
#     if "search_cache" not in st.session_state:
#         st.session_state.search_cache = {}
#         st.session_state.search_cache_timestamps = {}
#         st.session_state.search_requests_made = set()  # Track all searches made in session
    
#     # Clear expired cache entries (older than 1 hour)
#     current_time = time.time()
#     expired_queries = [
#         query for query, timestamp in st.session_state.search_cache_timestamps.items()
#         if current_time - timestamp > 3600
#     ]
#     for query in expired_queries:
#         del st.session_state.search_cache[query]
#         del st.session_state.search_cache_timestamps[query]
#         logging.info(f"Cleared expired cache entry for query: {query}")

#     while iteration < max_iterations:
#         if assistant_text and "Unknown function" not in assistant_text:
#             break  # Exit early if a valid response is obtained
#         if iteration >= 3 and "No relevant search results found" in assistant_text:
#             st.error("No valid data was found. Try a different query.")
#             return assistant_text  # Stop refining

#         if response and response.choices and response.choices[0].message:
#             message = response.choices[0].message
#             tool_calls = message.tool_calls or []
#             logging.info(f"LLM Response Content (Iteration {iteration + 1}): {message.content}")
            
#             search_queries = []
#             if tool_calls:
#                 for tool_call in tool_calls:
#                     function_name = tool_call.function.name
#                     function_args = tool_call.function.arguments
#                     logging.info(f"Function Name: {function_name}, Arguments: {function_args}")
#                     print(f"Function Name: {function_name}")
#                     status_placeholder.text(f"Using tool: {function_name}")

#                     if function_name == "AnswerQuestion":
#                         answer_data = AnswerQuestion.model_validate_json(function_args)
#                         assistant_text = answer_data.answer
#                         reflection = answer_data.reflection.model_dump() if hasattr(answer_data, "reflection") else {"missing": "No reflection provided.", "superfluous": "No reflection provided."}
#                         st.session_state["reflections"].append(reflection)

#                     elif function_name == "ReviseAnswer":
#                         answer_data = ReviseAnswer.model_validate_json(function_args)
#                         assistant_text = answer_data.answer
#                         reflection = answer_data.reflection.model_dump()
#                         st.session_state["reflections"].append(reflection)
#                     search_tool_name = "BraveSearchResults" if st.session_state.get("search_provider") == "brave" \
#                   else "TavilySearchResults"
#                     if function_name != search_tool_name:
#                         st.error(f"The LLM called function {function_name}, but the selected search provider is {search_tool_name}. The LLM must use the selected search tool.")
#                         continue  # Skip the incorrect search function

#                     try:
#                         if function_args and isinstance(function_args, str):
#                             function_args = json.loads(function_args)  # Ensure it's parsed correctly
#                         query = function_args.get("query", "")
#                         search_queries.append(query)
#                     except json.JSONDecodeError:
#                         assistant_text += f"\n\nError: Failed to parse JSON arguments for {function_name}."
#                         logging.error(f"JSON parsing error in {function_name}: {function_args}")
#                         continue
#                     except Exception as e:
#                         assistant_text += f"\n\nError processing tool call: {function_name} - {str(e)}"
#                         logging.error(f"Error processing tool call: {function_name} - {str(e)}")
#                         pass

#                 # Process search queries efficiently
#                 if search_queries and needs_search(user_question):
#                     combined_content = ""
#                     # Preserve order while getting unique queries, limited to first 3
#                     seen = set()
#                     unique_queries = [x for x in search_queries if not (x in seen or seen.add(x))][:3]

#                     for query in unique_queries:
#                         # Skip if query was already made in this session
#                         #if query in st.session_state.search_requests_made:
#                         #    logging.info(f"Skipping duplicate search: {query}")
#                         #    continue  # Prevent duplicate searches

#                         try:
#                             if query in st.session_state.search_cache:
#                                 search_results = st.session_state.search_cache[query]
#                                 logging.info(f"Cache hit for query: {query}")
#                             else:
#                                 logging.info(f"Cache miss - making new request for query: {query}")
#                                 st.session_state.search_requests_made.add(query)
#                                 search_tool = get_search_tool()  # Get current search tool
#                                 if search_tool:
#                                     try:
#                                         search_results = search_tool.run(query)
#                                         logging.info(f"Search results for query '{query}': {search_results}")
#                                     except Exception as e:
#                                         logging.error(f"Error invoking search tool: {str(e)}")
#                                         search_results = []
#                                 else:
#                                     logging.error("No valid search tool found.")
#                                     search_results = []

#                                 if search_results and isinstance(search_results, list):
#                                     st.session_state.search_cache[query] = search_results
#                                     st.session_state.search_cache_timestamps[query] = time.time()
#                                     st.session_state.search_requests_made.add(query)
#                                 else:
#                                     raise ValueError("No valid search results returned.")

#                             if search_results and isinstance(search_results, list):
#                                 combined_content += "\n".join([result.get("content", "") for result in search_results])
#                             else:
#                                 assistant_text += f"\n\nNo relevant information found for query: {query}"
#                         except Exception as e:
#                             logging.error(f"Error during search for query '{query}': {str(e)}")

#                             # Alert user and suggest switching providers
#                             st.warning(f"Search provider '{st.session_state.search_provider}' failed. "
#                                        "Try switching to Tavily in the sidebar settings.")

#                             assistant_text += f"\n\n⚠️ Search failed using {st.session_state.search_provider}. Try switching to Tavily."

#                     if combined_content:
#                         messages = [
#                             {"role": "user", "content": user_question},
#                             {"role": "assistant", "content": f"Search results: {combined_content}"}
#                         ]
#                         response = get_openai_response(messages, model_choice, use_revise_answer=True)
#             else:
#                 assistant_text = message.content or ""

#             iteration += 1
#             if iteration < max_iterations:
#                 if iteration <= answer_iterations_limit:
#                     use_revise_answer = True
#                 else:
#                     use_revise_answer = False

#                 messages = st.session_state["messages"] + [{"role": "assistant", "content": assistant_text}]
#                 response = get_openai_response(messages, model_choice, use_revise_answer=use_revise_answer)
#         else:
#             assistant_text = "Could not get a valid response from the model."
#             break

#     if iteration == max_iterations:
#         logging.warning("Maximum iterations reached.")

#     if not assistant_text:
#         assistant_text = "Reached maximum iterations without a final answer."
    
#     return assistant_text

# # =============================================================================
# # Main Streamlit App
# # =============================================================================
# def main():
#     # Clear the session state manually before running the app:
#     if "response_cache" in st.session_state:
#         del st.session_state["response_cache"]
#     if "search_cache" in st.session_state:
#         del st.session_state["search_cache"]
#     if "search_requests_made" in st.session_state:
#         del st.session_state["search_requests_made"]

#     st.set_page_config(page_title="Reflexions Multi-Tool Agent", page_icon="🤖")
#     st.title("Reflexions Multi-Tool Agent")

#     # Configure logging
#     logging.basicConfig(level=logging.INFO)

#     # Initialize session state variables
#     if "messages" not in st.session_state:
#         st.session_state["messages"] = [
#             {"role": "assistant", "content": "Hello! How can I assist you today?"}
#         ]
    
#     if "response_cache" not in st.session_state:
#         st.session_state["response_cache"] = {}
    
#     if "search_provider" not in st.session_state:
#         st.session_state["search_provider"] = "brave"  # Set default search provider
        
#     st.session_state["reflections"] = []

#     with st.sidebar:
#         st.header("Configuration")
#         model_choice = st.selectbox("Select the Azure deployment:", AVAILABLE_MODELS, index=0)
#         # Search provider selection
#         search_provider = st.selectbox(
#             "Search Provider:",
#             ["brave", "tavily"],
#             index=1,  # Default to Tavily Search
#             key="search_provider_select"
#         )
        
        
#         max_iterations = st.slider("Max Iterations:", min_value=1, max_value=20, value=5, step=1)
#         st.session_state["max_iterations"] = max_iterations
    
#         if st.button("Clear Conversation"):
#             st.session_state["messages"] = [
#                 {"role": "assistant", "content": "Conversation cleared. How can I help you now?"}
#             ]

#     for msg in st.session_state["messages"]:
#         with st.chat_message(msg["role"]):
#             st.write(msg["content"])

#     prompt = st.chat_input("Type your message here...", key="main_chat_input")
#     if prompt:
#         st.session_state["messages"].append({"role": "user", "content": prompt})
#         with st.chat_message("user"):
#             st.write(prompt)

#         with st.chat_message("assistant"):
#             message_placeholder = st.empty()
#             status_placeholder = st.empty()
#             assistant_text = ""

#             if st.session_state["messages"][-1]["role"] != "user":
#                 st.session_state["messages"].append({"role": "assistant", "content": ""})

#             with st.spinner("Thinking..."):
#                 if prompt.strip():
#                     messages = st.session_state["messages"]
#                     status_placeholder.text("Generating response...")
#                     response = get_openai_response(messages, model_choice)
#                     assistant_text = process_response(response, prompt, model_choice, status_placeholder)
#                     st.session_state["messages"][-1]["content"] = assistant_text

#                 if assistant_text:
#                     message_placeholder.markdown(assistant_text)
#                     status_placeholder.empty()
#                 else:
#                     message_placeholder.empty()
#                     status_placeholder.empty()

#             if st.session_state["reflections"]:
#                 with st.expander("Reflections"):
#                     reflections_output = ""
#                     for i, r in enumerate(st.session_state["reflections"], start=1):
#                         reflections_output += f"**Reflection {i}:**\n- Missing: {r.get('missing','')}\n- Superfluous: {r.get('superfluous','')}\n---\n"
#                     st.markdown(reflections_output)

# if __name__ == "__main__":
#     main()

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
        #logging.info(f"Raw OpenAI API Response: {response.model_dump_json()}") # FIXED LOGGING: Use model_dump_json()
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
    max_iterations = st.session_state.get("max_iterations", 5) # Default to 5 if not set

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