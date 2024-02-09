import streamlit as st
import pandas as pd
import os, json

# Load path.py from parent directory
from pathlib import Path
import sys
sys.path.append(str(Path('.').absolute().parent))
from path import *

from graph import generate_graphs
from streamlit_echarts import st_echarts

### Helpers
def load_df_from_folder(folder):
    files = os.listdir(folder) 
    df = pd.DataFrame()
    if len(files) == 0:
        return df
    for file in files:
        df = pd.concat([df, pd.read_json(f'{folder}/{file}')], ignore_index=True)
    return df

def get_tabs():
        # Insert containers separated into tabs:
    css = '''
<style>
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
    font-size:2rem;
    }
</style>
'''
    st.markdown(css, unsafe_allow_html=True)
    return st.tabs(["Results", "Tasks"], )

# Sidebar Functions
def SIDEBAR_select_evaluation_set():
    st.sidebar.title("Evaluation Set")

    evaluation_sets = os.listdir(EVALUATION_SET_DIR)
    sorted_list = sorted(evaluation_sets, key=lambda x: os.path.getctime(f"{EVALUATION_SET_DIR}/{x}"), reverse=True)
    evaluation_set_name = st.sidebar.selectbox("Select evaluation to load:", sorted_list)

    # Load evaluation set as dict
    evaluation_set = json.load(open(f"{EVALUATION_SET_DIR}/{evaluation_set_name}"))

    evaluation_set_name = evaluation_set_name.split(".")[0]
    if not os.path.exists(f"{RESULTS_DIR}/{evaluation_set_name}"):
        os.mkdir(f"{RESULTS_DIR}/{evaluation_set_name}")
    results_folder = f"{RESULTS_DIR}/{evaluation_set_name}"
    df = load_df_from_folder(results_folder)
    return df, evaluation_set

# Main Functions
def MAIN_create_filter(df, method):

    filter_options = {
        "model": list(df["model"].unique()),
        "task_id": list(df["task_id"].unique()),
        "ctx_n": list(df["ctx_n"].unique()),
        "temperature": list(df["temperature"].unique()),
        "top_k": list(df["top_k"].unique()),
        "top_p": list(df["top_p"].unique()),
        "num_predict": list(df["num_predict"].unique()),
        "passed": list(df["passed"].unique()),
        "few_shot": list(df["few_shot"].unique()) if "few_shot" in df.columns else [""],
        "rel_boards": list(df["rel_boards"].unique()) if "rel_boards" in df.columns else [""],
        "num_shots": list(set([ len(x) for x in df["shots"] ])),
    }
    filter_options = {k: sorted(v) for k, v in filter_options.items()}
    filter = {}

    # Remove NaNs from filter
    for key, value in filter_options.items():
        if isinstance(value[0], float):
            filter_options[key] = [x for x in value if not pd.isna(x)]
    
    # Create a grid of 2x4 for the options
    cols = st.columns(4)
    for i, (key, value) in enumerate(filter_options.items()):
        col_index = i % 4
        with cols[col_index]:
            selected_values = st.multiselect(key, value, placeholder=f"All")
            filter[key] = selected_values

    filter = {k: v for k, v in filter.items()}

    # Filter out non-target methods (that aren't random)
    df = df[df["method"].isin([method, "random"])]

    for key, value in filter.items():
        if value:
            if key=="num_shots":
                value = [int(x) for x in value]
                df["num_shots"] = [len(x) for x in df["shots"]]

            df = df[df[key].isin(value)]

    return df

def MAIN_description():
    st.title(":red[Evaluation Set Visualisation]")
    st.write("This page allows you to edit and view the results of evaluation sets.")
    st.write("Select an evaluation set from the sidebar to get started.")

def MAIN_task_tab(df):
    st.title("Tasks")
    st.write("The table below shows the tasks in the evaluation set.")
    st.write("Click on a row to view the full task description.")
    st.write("Use the filters on the left to filter the tasks.")

def MAIN_results_tab(df, method):
    st.title(f"Visualisation of Method: {method}")
    st.write("Filter the evaluation results by selecting the options below.")

    df = MAIN_create_filter(df, method)
    options_keys = generate_graphs(df)

    # Display graphs in a 2 column grid
    cols = st.columns(2)
    for i, (options, key) in enumerate(options_keys):
        col_index = i % 2
        with cols[col_index]:
            st_echarts(options, key=key)


if __name__ == "__main__":
    # Site options
    st.set_page_config(
        page_title="Evaluation Sets",
        page_icon="📚",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Sidebar button
    df, evaluation_set = SIDEBAR_select_evaluation_set()
    st.sidebar.text_area("Evaluation set description:", value=evaluation_set["details"]["description"], height=150)
    if st.sidebar.button("Refresh", use_container_width=True):
        st.rerun()

    if df.empty:
        st.text("No results found.")
        st.stop()

    # Sidebar method choice
    st.sidebar.title("Method")
    method_options = list(df["method"].unique())
    method_options = [x for x in method_options if x != "random"]
    method_options = sorted(method_options, reverse=True)
    method = st.sidebar.selectbox("Select method to visualise.", method_options)

    MAIN_description()

    results_tab, tasks_tab = get_tabs()

    with tasks_tab:
        MAIN_task_tab(df)

    with results_tab:
        MAIN_results_tab(df, method)
