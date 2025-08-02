import logging
import json
# Configure logging
logging.basicConfig(level=logging.WARNING)
debug_logger = logging.getLogger("elastic_agent")
debug_logger.setLevel(logging.DEBUG)

# Add a file handler to save debug logs
file_handler = logging.FileHandler("streamlit_debug.log")
file_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
debug_logger.addHandler(file_handler)

# Silence all loggers
for logger_name in logging.root.manager.loggerDict:
    logging.getLogger(logger_name).setLevel(logging.WARNING)
import warnings
import streamlit as st
from agent import generate_response
import pandas as pd
import os
from datetime import datetime
import re
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.colors
# from langchain_core.messages.tool import ToolMessage
from langchain_core.messages import ToolMessage, AIMessage, HumanMessage

# Suppress specific warnings
warnings.filterwarnings("ignore", message="resource module not available on Windows")
warnings.filterwarnings("ignore", message="Examining the path of torch.classes raised")
warnings.filterwarnings("ignore", message="Tried to instantiate class '__path__._path'")
warnings.filterwarnings("ignore", message=".*missing ScriptRunContext.*")

# Set page configuration first
st.set_page_config(page_title="Elastic Query Chat", layout="wide")

def remove_think_blocks(text):
    """
    Removes all content between <think> and </think> tags, including the tags themselves.
    """
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)

def extract_code_from_message(message_content):
    """
    Extracts the first Python code block from a markdown string.
    Returns the code as a string, or None if not found.
    """
    match = re.search(r"```(?:python)?\n([\s\S]+?)```", message_content)
    if match:
        return match.group(1)
    return None

def remove_code_blocks(text):
    """
    Removes all code blocks (```...```) from the text.
    """
    return re.sub(r"```(?:python)?\n[\s\S]+?```", "", text)


if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # List of dicts

def save_response_to_excel(response, user_question, final_answer, base_filename="elastic_response"):
    """
    Saves the response dict to an Excel file with two sheets:
    - Sheet 1: Query (question and final answer)
    - Sheet 2: Data (documents and aggregations, if any; both are concatenated vertically, with a separator row)
    Returns the file path.
    """
    import xlsxwriter
    # Create a unique filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{base_filename}_{timestamp}.xlsx"
    filepath = os.path.join("outputs", filename)
    os.makedirs("outputs", exist_ok=True)
    response = json.loads(response['tools']['messages'][0].content)

    with pd.ExcelWriter(filepath, engine="xlsxwriter") as writer:
        workbook = writer.book
        # --- Sheet 1: Query ---
        ws_query = workbook.add_worksheet("Query")
        bold_format = workbook.add_format({'bold': True})
        wrap_format = workbook.add_format({'text_wrap': True, 'valign': 'top'})
        large_wrap_format = workbook.add_format({'text_wrap': True, 'valign': 'top', 'font_size': 12})
        # Write headers
        ws_query.write(1, 0, "Query", bold_format)
        ws_query.write(1, 1, user_question, large_wrap_format)
        ws_query.write(2, 0, "AI Response", bold_format)
        ws_query.write(2, 1, final_answer, large_wrap_format)
        ws_query.set_column(0, 0, 15)   # Label column
        ws_query.set_column(1, 1, 100)  # Content column
        ws_query.set_row(1, 80)
        ws_query.set_row(2, 120)

        # --- Sheet 2: Data ---
        docs = response.get("documents")
        aggs = response.get("aggregations")
        dataframes = []

        # Documents section
        if docs:
            if isinstance(docs, list):
                if docs and isinstance(docs[0], dict):
                    df_docs = pd.DataFrame(docs)
                else:
                    df_docs = pd.DataFrame({"documents": docs})
            elif isinstance(docs, dict):
                df_docs = pd.DataFrame([docs])
            else:
                df_docs = pd.DataFrame({"documents": [docs]})
            dataframes.append(df_docs)

        # Aggregations section
        if aggs:
            # Insert a separator row if there are docs and aggs
            if dataframes:
                sep_df = pd.DataFrame({list(dataframes[0].columns)[0]: ["--- Aggregations ---"]})
                dataframes.append(sep_df)
            # Try to flatten aggregations for display
            if isinstance(aggs, dict):
                for agg_key, agg_val in aggs.items():
                    # Insert a sub-separator for each aggregation
                    sub_sep_df = pd.DataFrame({agg_key: [f"--- Aggregation: {agg_key} ---"]})
                    dataframes.append(sub_sep_df)
                    agg_val = agg_val['buckets']
                    if isinstance(agg_val, list):
                        df_agg = pd.DataFrame(agg_val)
                    elif isinstance(agg_val, dict):
                        df_agg = pd.DataFrame([agg_val])
                    else:
                        df_agg = pd.DataFrame({agg_key: [agg_val]})
                    dataframes.append(df_agg)
            elif isinstance(aggs, list):
                sep_df = pd.DataFrame({"aggregation": ["--- Aggregations ---"]})
                dataframes.append(sep_df)
                dataframes.append(pd.DataFrame(aggs))
            else:
                sep_df = pd.DataFrame({"aggregation": ["--- Aggregations ---"]})
                dataframes.append(sep_df)
                dataframes.append(pd.DataFrame({"aggregation": [aggs]}))

        # Combine all dataframes with a blank row in between
        if dataframes:
            combined = pd.concat(dataframes, ignore_index=True)
            combined.to_excel(writer, sheet_name="Data", index=False)
            ws_data = writer.sheets["Data"]
            # Format header: grey background, bold
            header_format = workbook.add_format({'bold': True, 'bg_color': '#D9D9D9'})
            for col_num, value in enumerate(combined.columns.values):
                ws_data.write(0, col_num, value, header_format)
            # Autofit columns
            for i, col in enumerate(combined.columns):
                max_len = max(
                    [len(str(col))] + [len(str(val)) for val in combined[col]]
                )
                ws_data.set_column(i, i, max_len + 2)
            # Freeze top row
            ws_data.freeze_panes(1, 0)
    return filepath

import re

def remove_markdown(text):
    """
    Removes common markdown formatting from the text, preserving newlines and structure.
    """
    text = re.sub(r"```(?:[\w]*)\n([\s\S]+?)```", r"\1", text)
    text = re.sub(r"`([^`]*)`", r"\1", text)
    text = re.sub(r"\*\*\*([^\*]+)\*\*\*", r"\1", text)  
    text = re.sub(r"\*\*([^\*]+)\*\*", r"\1", text)      
    text = re.sub(r"\*([^\*]+)\*", r"\1", text)          
    text = re.sub(r"__([^_]+)__", r"\1", text)           
    text = re.sub(r"_([^_]+)_", r"\1", text)          
    text = re.sub(r"^#+\s*(.*)", r"\1\n", text, flags=re.MULTILINE)
    text = re.sub(r"^>\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"^-{3,}$", "", text, flags=re.MULTILINE)
    text = re.sub(r"!\[[^\]]*\]\([^)]+\)", "", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"!\[[^\]]*\]\([^)]+\)", "", text)
    return text

def append_chat(sender, message, excel_path=None, chart_code=None):
    """Add a message to the chat history. Optionally attach an Excel file path and chart code."""
    st.session_state.chat_history.append({
        "sender": sender,
        "message": message,
        "excel_path": excel_path,
        "chart_code": chart_code
    })

def render_chat():
    """Display all chat history, with download buttons for assistant messages with Excel files and charts if present."""
    for idx, entry in enumerate(st.session_state.chat_history):
        sender = entry["sender"]
        message = entry["message"]
        excel_path = entry.get("excel_path")
        chart_code = entry.get("chart_code")
        st.chat_message(sender).markdown(message, unsafe_allow_html=True)
        if sender == "assistant" and excel_path and os.path.exists(excel_path):
            with open(excel_path, "rb") as f:
                st.download_button(
                    label="Download results as Excel",
                    data=f,
                    file_name=os.path.basename(excel_path),
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key=f"download_{idx}"
                )
        # Render chart if chart_code is present
        if sender == "assistant" and chart_code:
            # st.info("Chart generated from the assistant's response:")
            local_vars = {}
            try:
                exec(chart_code, {
                    "pd": pd,
                    "px": px,
                    "go": go,
                    "make_subplots": make_subplots,
                    "plotly": plotly
                }, local_vars)
                fig = local_vars.get('fig') 

                if fig:
                    st.plotly_chart(fig, use_container_width=False)  # Display the Plotly chart
                else:
                    st.error("Failed to generate the plot.")
            except Exception as e:
                st.error(f"Error executing chart code: {e}")

st.title("Elastic Query Chat")

render_chat()

user_input = st.chat_input("Enter your question:")
if user_input:
    append_chat("user", user_input)
    st.chat_message("user").markdown(user_input)
    with st.spinner("Processing..."):
        try:
            debug_logger.info(f"Processing new query: {user_input}")
            all_response = generate_response(user_input)
            
            if 'elastic' in all_response[-1]:
                last_msgs = all_response[-1]['elastic']['messages']
            elif 'chart_generator' in all_response[-1]:
                last_msgs = all_response[-1]['chart_generator']['messages']
            else:
                last_msgs = []

            if last_msgs:
                agent_message = last_msgs[-1].content
            else:
                agent_message = ""
            response = remove_think_blocks(agent_message).strip()
            response_text = remove_markdown(remove_code_blocks(response)).strip().replace("FINAL ANSWER", "").strip(":")

            # --- Find latest valid ToolMessage for Excel export ---
            excel_path = None
            elastic_msgs = all_response[0]['elastic']['messages'] if 'elastic' in all_response[0] else []
            for msg in reversed(elastic_msgs):
                if isinstance(msg, ToolMessage) and isinstance(msg.content, str):
                    try:
                        content_dict = json.loads(msg.content)
                    except Exception:
                        continue
                    if (
                        isinstance(content_dict, dict) and
                        all(k in content_dict for k in ['total', 'documents', 'aggregations'])
                    ):
                        try:
                            excel_path = save_response_to_excel(
                                {'tools': {'messages': [msg]}}, user_input, response_text)
                        except Exception:
                            excel_path = None
                        break

            chart_code = None
            chart_msgs = all_response[1]['chart_generator']['messages'] if len(all_response) > 1 and 'chart_generator' in all_response[1] else []
            for msg in reversed(chart_msgs):
                if isinstance(msg, ToolMessage) and isinstance(msg.content, str):
                    code = extract_code_from_message(msg.content)
                    if code:
                        chart_code = code
                        break

            response_text += f"\n\n💾 *The complete results can be downloaded.*" if excel_path else ""
        except Exception as e:
            debug_logger.error(f"Error in generate_response: {str(e)}", exc_info=True)
            response_text = f"Could not process request. Please try after some time...."
            chart_code = None
            excel_path = None

        append_chat("assistant", response_text, excel_path=excel_path, chart_code=chart_code)
        st.chat_message("assistant").markdown(response_text, unsafe_allow_html=True)
        if chart_code:
            st.info("Chart generated from the assistant's response:")
            local_vars = {}
            try:
                exec(chart_code, {"pd": pd, "px": px}, local_vars)
                fig = local_vars.get('fig')

                if fig:
                    st.plotly_chart(fig, use_container_width=False)
                else:
                    st.error("Failed to generate the plot.")

            except Exception as e:
                st.error(f"Error executing chart code: {e}")
        st.rerun()
