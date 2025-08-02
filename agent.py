# hide-cell
from tabnanny import check
from langchain_core.messages import convert_to_messages
from pandas._libs.window import aggregations


def pretty_print_message(message, indent=False):
    pretty_message = message.pretty_repr(html=True)
    if not indent:
        print(pretty_message)
        return

    indented = "\n".join("\t" + c for c in pretty_message.split("\n"))
    print(indented)


def pretty_print_messages(update, last_message=False):
    is_subgraph = False
    if isinstance(update, tuple):
        ns, update = update
        # skip parent graph updates in the printouts
        if len(ns) == 0:
            return

        graph_id = ns[-1].split(":")[0]
        print(f"Update from subgraph {graph_id}:")
        print("\n")
        is_subgraph = True

    for node_name, node_update in update.items():
        update_label = f"Update from node {node_name}:"
        if is_subgraph:
            update_label = "\t" + update_label

        print(update_label)
        print("\n")

        messages = convert_to_messages(node_update["messages"])
        if last_message:
            messages = messages[-1:]

        for m in messages:
            pretty_print_message(m, indent=is_subgraph)
        print("\n")

# let create a data_analysis agents
import os
import sys
import contextlib
import pandas as pd
import numpy as np
from io import StringIO
from dotenv import load_dotenv
from langgraph.prebuilt import create_react_agent
from langchain.chat_models import init_chat_model
from langgraph_supervisor import create_supervisor
from langgraph.checkpoint.memory import InMemorySaver
from langchain_experimental.utilities import PythonREPL
from langchain_core.tools import tool
from typing import Annotated
from typing import Literal

from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.prebuilt import create_react_agent
from langgraph.graph import MessagesState, END
from langgraph.types import Command

load_dotenv()

checkpointer = InMemorySaver()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# model = init_chat_model(
#     model='qwen3:14b',
#     model_provider="ollama",
#     temperature=0.01,
#     num_ctx=10000
# )

model = init_chat_model(
    model='llama3.1:latest',
    model_provider="ollama",
    temperature=0.01,
    num_ctx=10000
)

# model = init_chat_model(
#     model='gpt-4o-mini',
#     temperature=0.1,
#     api_key=OPENAI_API_KEY   
# )

from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.tools import tool

import os 

# Set Hugging Face API key directly in your environment
os.environ["HF_API_KEY"] = os.getenv("HF_API_KEY")


# llm = HuggingFaceEndpoint(
#     model="Llama-xLAM-2-8b-fc-r",
#     # model="Qwen/Qwen3-14B",
#     task="text-generation",
#     provider="auto",
#     temperature=0.1
# )

# llm = HuggingFaceEndpoint(
#     model="Qwen/Qwen3-32B",
#     # model="Qwen/Qwen3-14B",
#     task="text-generation",
#     provider="nebius",
#     temperature=0.1
# )
# model = ChatHuggingFace(llm=llm, verbose=True)

@tool
def elastic_db_api(payload: dict):
    """
    This API allows dynamic, programmatic querying and aggregation of data stored in an Elasticsearch database. It is designed to flexibly handle complex search, filter, and aggregation requirements using a structured JSON payload.
    **Args: 
      payload: This is a dictionary and has the keys -- `filters`, `fields`, `logical_opeartor`, etc.**
    
    """
    print("Calling the elastic api tool...")
    import requests

    # You may want to set this to your actual FastAPI server URL
    ELASTIC_API_URL = os.getenv("TOOL_API_URL") + "query"

    try:
        print("*"*80)
        response = requests.post(ELASTIC_API_URL, json=payload)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return {"error": str(e)}

repl = PythonREPL()


@tool
def python_repl_tool(code: Annotated[str, "The python code to execute to generate your chart."],):
    """Use this to execute python code. If you want to see the output of a value,
    you should print it out with `print(...)`. This is visible to the user."""
    try:
        result = repl.run(code)
    except Exception as e:
        return f"Failed to execute. Error: {str(e)}"
    # Check if result contains an error message
    result_str = str(result)
    if "Traceback" in result_str or "Error" in result_str or "Exception" in result_str or "TypeError" in result_str:
        print(result_str)
        return f"Execution failed:\n```python\n{code}\n```\nStdout: {result_str}"
    print(result)
    return (
        f"Successfully executed:\n```python\n{code}\n```\n\n**If you have completed all tasks, make sure to respond with prefix `FINAL ANSWER`.**"
    )


prompt = (
          "You are an **ElasticDB Agent** responsible for dynamically generating Elasticsearch query payloads based on natural language questions. You must construct valid Python dictionary objects that can be directly passed to the elastic_db_api function..\n\n"
          "\n **You are working with a Chart Generator Agent colleague which is capable of generating charts You do not have access to it directly.**"" If you or any of the other assistants have the final answer or deliverable,"
        " prefix your response with **FINAL ANSWER** so the team knows to stop. But when the user explicitly asks in the question itself to give a visualization or chart or representation, MAKE SURE TO PASS TO `Chart Generator Agent` colleague.\n\n"

        "\n## Execution Steps (Strictly follow this):\n"
        "1. **Understand the Question**: Analyze the user's question to identify the main requirements, such as fields, filters, and aggregation needs (e.g., date ranges, geographical filters, etc.). Extract any specific data types requested (e.g., emission data, day-wise counts, etc.). Return all fields if user asks all records.\n"
        "2. **Chart Requirement**: Analyse and Check in the user question if any chart or plot or any kind of visualization is required -- Something like 'Give me a graph or plot' or 'Give me a visualization'. If there is **NO** requirement, then you need to return with prefix 'FINAL ANSWER` in your response after fetching the data from tool. If there is requirement, then DO **NOT** mention prefix 'FINAL ANSWER'.\n"
        "3. **Break Down Complex Queries**: If the query is complex or contains multiple parts, break it down into **simpler sub-queries**. For example, if a query asks for data from multiple fields with different filters or multiple aggregations, separate the task into individual queries for each aspect (e.g., first filter for date range, then filter by geographical distance, then apply aggregation).\n"
        "4. **Determine the Filters**: Construct the filter conditions based on the user’s request. Ensure that `filters` is always a list of dictionaries (not a string). Apply the correct filter operators (e.g., `eq`, `range`, `geo_distance`) based on the question, using correct field names.\n"
        "5. **Logical Operators**: If multiple conditions are specified, combine them logically using `AND` or `OR`. Ensure that logical operators are applied appropriately across the filters.\n"
        "6. **Handle Aggregations**: If the user requests aggregation, determine which type of aggregation is needed (e.g., `date_histogram`, `terms`, etc.). Apply nested or multi-level aggregations if required. Ensure that the aggregation structure is correctly nested and adheres to Elasticsearch's expected format.\n"
        "7. **Fields and Size**: Determine which fields should be included in the response. If the question asks for aggregation data, set `size=0`. If the question asks for data, set `size=1000`. Always ensure that the appropriate size and scroll parameters are used for large queries.\n"
        "8. **Construct the Payload**: Using the information extracted, create the JSON payload for the Elasticsearch query. Ensure that the payload is valid JSON/dictionary with correct field names, operators, filters, aggregations, and other parameters. Go through the INSTRUCTION section as well.\n"
        "9. **Make the Tool Call**: Once the payload is constructed, call the `elastic_db_api` toll with the ONLY argument payload.\n"
        "10. **Handling Multiple Tool Calls**: If the user's question cannot be answered with a single query, **split the query into multiple sub-queries**. Each sub-query should handle a specific aspect of the overall task. After constructing each sub-query, call the `elastic_db_api` function with the appropriate payload for each.\n"
        "11. **Return the Response**: Once the queries are executed, return the response from the API or handle any errors (e.g., timeout, invalid query) accordingly. If the query was successful, return the result in a structured format.\n"
        "12. **Chart generation**: If the user's question explicitly asks about a visualization or chart or representation of various trend, pattern, day-wise, week-wise or month-wise or distribution across a date range, time period, or over time, then the query and its results should be prepared for chart generation and the output data should be transferred to a `chart generator agent` component for visualization. In this case, do **NOT** mention prefix FINAL ANSWER in you answer.\n"
        "13. **Clarification (If Needed)**: If the query is unclear or the provided details are insufficient, ask the user for clarification before proceeding to avoid generating an incorrect query.\n"
        "14. **Retrying (If tool call failed)**: If the tool call returns and error or failure, regenerate the payload to fix the error and retry calling the tool `elastic_db_api` with the newly constructed payload.\n"

        "Note: In cases of graph asked by user, just DO NOT mention prefix 'FINAL RESPONSE'. THAT'S IT."
        "Execution should be methodical and follow the above steps in sequence to ensure that the query is generated correctly and efficiently. **Always break down complex queries into simpler questions** and **issue multiple calls** if necessary.\n\n"

        "# **CRITICAL**: Make sure to construct the payload as just ONE argument, NOT as seperate keys.\n\n"
        
        "##IMPORTANT INSTRUCTIONS:\n"
        "- You assist ONLY with data retrieval, filtering, and aggregation tasks using the Elasticsearch database via the provided API. You will receive a query from the user describing the information or analysis required.\n"
        "- The Chart generator Agent is ALWAYS available and will assist in generating chart or plots or visualization. Follow the instructions to correctly use the chart agent. And you are capable of generation of charts.\n"
        "- Your job is to construct a JSON payload for the /query endpoint of the Elasticsearch API, using the allowed fields, operators, and aggregation types. Do NOT reuse the same values from the examples given. Use the examples to dynamically construct a new payload.\n"
        "- You will be given a question and you need to construct a JSON payload analyzing the user's question for the /query endpoint of the Elasticsearch API, using the allowed fields, operators, and aggregation types. Do NOT take the same values from the examples given.\n"
        "- Make sure to correctly call the tool 'elastic_db_api' with the payload you have constructed.\n"
        "- The tool has ONLY ONE input argument, that is payload.\n"
        "- You may combine multiple filters using logical operators (AND, OR) and nest them as needed. You may specify which fields to return in the response.\n"
        "- Before returning the response, make sure to check if there is a requirement for chart or visualization. If there is requirement, then DO NOT MENTION 'FINAL ANSWER' in your response.\n"
        "- For large result sets, use the scroll feature by setting the size and scroll parameters 'scroll=1000'.\n"
        "- Make sure to correctly understand and then generate the payload before hitting the API.\n"
        "- Use **size=0** ONLY for aggregations. For non-aggregation queries, use size=1000. Always use scroll='2m'.\n"
        "- If the user requests an aggregation or summary, construct the appropriate aggregation structure in the payload. Construct the JSON payload as per the question asked.\n"
        "- **NEVER** convert lists or dictionaries to JSON strings within the payload. Your payload MUST be a valid Python dictionary. It is always a single argument input for a tool call to elastic_db_api.\n"
        "- **IMPORTANT**: Make sure to mention prefix 'FINAL ANSWER' in your response unless it is explicitly asked by the user to plot the visualization or chart of the data.\n"
        "- **IMPORTANT**: In case the given user question cannot be solved by just one single tool call, you **MUST** break down the **question into simpler, more manageable parts** and issue multiple tool calls with appropriate payloads.\n"
        "- **IMPORTANT**: **If the user did NOT mention for any visualization or graph, then return your answer with prefix 'FINAL ANSWER' after fetching the data.**\n"
        "- **IMPORTANT**: **If the user question explicitly mention for visualization, then DO NOT return your response with prefix as FINAL ANSWER. If there is no ask about chart, then return your response with prefix 'FINAL ANSWER'**\n"
        "- Do NOT mention the logical operators in '{}' separately.\n"
        "- For the last 7 days or 30 days kind of questions, you can use `now/d` from elastic.\n"
        "- If the request is unclear, ambiguous, or cannot be fulfilled with the available fields and operations, respond with a clear message indicating the issue.\n"


        "## **CRITICAL**: **Make sure NOT to return the prefix 'FINAL RESPONSE' if the user question asks for a graph, plot or a visualization.\n\n"

        """\n## Payload description and structure
        - Supports nested and compound filters using logical operators (`AND`, `OR`).
        - Allows filtering on specific fields. Make sure to pass the correct fields
        - Supported filter operations:
          - `eq` (equals)
          - `ne` (not equals)
          - `range` (between values, e.g., date or number ranges)
          - `geo_distance` (geospatial queries within a distance from a point)

        - Supports a wide range of Elasticsearch aggregation types, including:
          - **Bucket Aggregations:** `terms`, `date_histogram`, `histogram`, `range`, `filters`, `composite`
          - **Metric Aggregations:** `avg`, `sum`, `min`, `max`, `cardinality`, `value_count`, `stats`, `extended_stats`
          - **Pipeline Aggregations:** `bucket_selector`, `bucket_sort`, `avg_bucket`, `sum_bucket`, `min_bucket`, `max_bucket`, `stats_bucket`, `extended_stats_bucket`
        - Aggregations can be nested (sub-aggregations) for multi-level grouping and analysis.

        - Allows specifying which fields to return in the response.
        """)

config = {"configurable": {"thread_id": "505", "recursion_limit": 15}}

elasticdb_agent = create_react_agent(
    model=model,
    tools=[elastic_db_api],
    checkpointer=checkpointer,
    prompt=prompt,
    name="elasticdb_agent",
)

import re
def remove_think_blocks(text):
    """
    Removes all content between <think> and </think> tags, including the tags themselves.
    """
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)

def get_next_node(last_message: BaseMessage, goto: str):
    if "FINAL ANSWER" in remove_think_blocks(last_message.content):
        # Any agent decided the work is done
        return END
    return goto

def elastic_node(
    state: MessagesState,
) -> Command[Literal["chart_generator", END]]:
    print("In elastic node.....")
    result = elasticdb_agent.invoke(state)
    goto = get_next_node(result["messages"][-1], "chart_generator")
    # wrap in a human message, as not all providers allow
    # AI message at the last position of the input messages list
    result["messages"][-1] = HumanMessage(
        content=result["messages"][-1].content, name="elastic"
    )
    return Command(
        update={
            # share internal message history of research agent with other agents
            "messages": result["messages"],
        },
        goto=goto,
    )


chart_agent = create_react_agent(
    model,
    [python_repl_tool],
    checkpointer=checkpointer,
    prompt=(
        "You can only generate charts and visualizations. You are working with a Elastic agent colleague.\n"
        "If you or any of the other assistants have the final answer or deliverable,"
        " Prefix your response with FINAL ANSWER so the team knows to stop. \n\n"
        "Carefully examine the input data and the user's original request (if available) to determine the appropriate chart type (e.g., bar, pie, line) and necessary configurations.\n"
        "Make sure to import all the required and necessary libraries such as `plotly`, `px` and so on.\n"
        "During code generation, start with correctly defining the data into variables and then using this data generate the code to get the graph or chart.\n"
        "**Compulsarily define the data in a variable.**\n"
        "IMPORTANT: Use plotly for plotting the data. And importantly use ONLY `fig` variable to store the final figure. NO NEED FOR `fig.show()` OR `fig`.\n"
        "IMPORTANT: Recheck with the logic to define all the variables and parse the data correctly. The code should be correct and give the plot without any errors.\n"
        "IMPORTANT: Use the data provided by `Elastic Agent` and construct a python that will correctly plot the data.\n"
        "IMPORTANT: Make sure to call the tool `python_repl_tool` with you to run and execute the python code to visualise the chart. If the code returns an error, then generate the code again to fix the error and rerun the tool.\n\n"
        "IMPORTANT: Do NOT use sample data. Take the data from your colleague `elastic agent`. And give the plot that is insightful -- like bar graph or anything that is insightful. Use colors for the charts to make it more informative and insightful.\n\n"
        "**Make sure to use your intelligence to give the correct plot of the data whichever will be more insightful to view. Keep it simple but informative.**\n"
        "**Make sure to return the answer for the user question as well in your FINAL ANSWER.\n"
    ),
)


def chart_node(state: MessagesState) -> Command[Literal["elastic", END]]:
    print("Now in chart node....")
    result = chart_agent.invoke(state)
    goto = get_next_node(result["messages"][-1], "elastic")
    # wrap in a human message, as not all providers allow
    # AI message at the last position of the input messages list
    result["messages"][-1] = HumanMessage(
        content=result["messages"][-1].content, name="chart_generator"
    )
    return Command(
        update={
            # share internal message history of chart agent with other agents
            "messages": result["messages"],
        },
        goto=goto,
    )


  
from langgraph.graph import StateGraph, START

workflow = StateGraph(MessagesState)
workflow.add_node("elastic", elastic_node)
workflow.add_node("chart_generator", chart_node)

workflow.add_edge(START, "elastic")
graph = workflow.compile(checkpointer=checkpointer)

def generate_response(user_question):
  response = []
  for chunk in graph.stream(
      {"messages": 
          [
              {
                  "role": "user", 
                  "content":user_question
              }
          ]
      },
      config,
  ):
      pretty_print_messages(chunk)
      response.append(chunk)
  return response
