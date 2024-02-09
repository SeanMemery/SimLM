import pandas as pd
import copy

# Set matplot colours 
colours = ["#0072b2", "#d55e00", "#cc79a7"]

# Marker colours dark green and light red
marker_colours = ["#009e73", "#e69f00"]

ROUND_N = 3

GLOBAL_OPTIONS = {
    "tooltip": {"trigger": "axis", "axisPointer": {"type": "shadow"}}, 
    "xAxis": {
        "type": "category",
        "nameLocation": "middle",
        "nameTextStyle": {
            "fontSize": 16,
        },
    },
    "yAxis": {
        "type": "value",
        "min": 0,
        #"max": 0.015,
    },
    "series": [{"type": "bar"}],
    "toolbox": {
        "show": True,
        "feature": {
            "dataView": { "show": True, "readOnly": False },
            "magicType": { "show": True, "type": ['line', 'bar'] },
            "restore": { "show": True },
            "saveAsImage": { "show": True}
        }
    },
}

def trial_count_per_model(df):
    """
    ### Trial count of each model
    """

    # Get trial count of each model
    trial_count_per_model = df.groupby("model").size().reset_index(name="count")

    # Plot
    options = copy.deepcopy(GLOBAL_OPTIONS)
    options["title"] = {"text": "Trial count of each model"}
    options["xAxis"]["data"] = trial_count_per_model["model"].tolist()
    options["series"][0]["data"] = trial_count_per_model["count"].tolist()

    # Set max y-axis value, rounded to nearest 100
    options["yAxis"]["max"] = max(options["series"][0]["data"]) + 1000 - (max(options["series"][0]["data"]) % 100)

    #Remove marklines
    options["series"][0]["markLine"] = {}

    return (options, "trial_count_per_model")

def pass_rate_ctx_n(df):
    """
    ### Pass rate of each context size
    """

    # Get pass rate of each context size
    pass_rate_ctx_n = df.groupby("ctx_n")["passed"].mean().reset_index(name="pass_rate")
    pass_rate_ctx_n["pass_rate"] = pass_rate_ctx_n["pass_rate"].apply(lambda x: round(x, ROUND_N))

    # Plot
    options = copy.deepcopy(GLOBAL_OPTIONS)
    options["title"] = {"text": "Pass rate of each context size"}
    options["xAxis"]["name"] = "Context size"
    options["xAxis"]["data"] = pass_rate_ctx_n["ctx_n"].tolist()
    options["series"][0]["data"] = pass_rate_ctx_n["pass_rate"].tolist()

    return (options, "pass_rate_ctx_n")

def pass_rate_temperature(df):
    """
    ### Pass rate of each temperature
    """

    # Get pass rate of each temperature
    pass_rate_temperature = df.groupby("temperature")["passed"].mean().reset_index(name="pass_rate")
    pass_rate_temperature["pass_rate"] = pass_rate_temperature["pass_rate"].apply(lambda x: round(x, ROUND_N))

    # Plot
    options = copy.deepcopy(GLOBAL_OPTIONS)
    options["title"] = {"text": "Pass rate of each temperature"}
    options["xAxis"]["name"] = "Temperature"
    options["xAxis"]["data"] = pass_rate_temperature["temperature"].tolist()
    options["series"][0]["data"] = pass_rate_temperature["pass_rate"].tolist()

    return (options, "pass_rate_temperature")

def pass_rate_num_predict(df):
    """
    ### Pass rate of each number of predictions to make during inference
    """

    # Get pass rate of each number of predictions to make during inference
    pass_rate_num_predict = df.groupby("num_predict")["passed"].mean().reset_index(name="pass_rate")
    pass_rate_num_predict["pass_rate"] = pass_rate_num_predict["pass_rate"].apply(lambda x: round(x, ROUND_N))

    # Plot
    options = copy.deepcopy(GLOBAL_OPTIONS)
    options["title"] = {"text": "Pass rate of each number of predictions"}
    options["xAxis"]["name"] = "Number of predictions"
    options["xAxis"]["data"] = pass_rate_num_predict["num_predict"].tolist()
    options["series"][0]["data"] = pass_rate_num_predict["pass_rate"].tolist()

    return (options, "pass_rate_num_predict")

def pass_rate_per_model(df):
    """
    ### Pass rate of each model (including average across all models and model=random performance)
    """

    # Get pass rate of each model
    pass_rate_model = df.groupby("model")["passed"].mean().reset_index(name="pass_rate")
    pass_rate_model["pass_rate"] = pass_rate_model["pass_rate"].apply(lambda x: round(x, ROUND_N))

    # Plot
    options = copy.deepcopy(GLOBAL_OPTIONS)
    options["title"] = {"text": "Pass rate of each model"}
    options["xAxis"]["data"] = pass_rate_model["model"].tolist()
    options["yAxis"]["max"] = max(pass_rate_model["pass_rate"].tolist())*2.5
    options["series"][0]["data"] = pass_rate_model["pass_rate"].tolist()

    return (options, "pass_rate_per_model")

def pass_rate_task_id(df):
    """
    ### Pass rate of each task ID
    """

    # Get pass rate of each task ID
    pass_rate_task_id = df.groupby("task_id")["passed"].mean().reset_index(name="pass_rate")
    pass_rate_task_id["pass_rate"] = pass_rate_task_id["pass_rate"].apply(lambda x: round(x, ROUND_N))

    # Plot
    options = copy.deepcopy(GLOBAL_OPTIONS)
    options["title"] = {"text": "Pass rate of each task ID"}
    options["xAxis"]["name"] = "Task ID"
    options["xAxis"]["data"] = pass_rate_task_id["task_id"].tolist()
    options["series"][0]["data"] = pass_rate_task_id["pass_rate"].tolist()
    
    return (options, "pass_rate_task_id")

def outcome_per_model(df):
    """
    The outcome rate per model, possible outcomes are:
    - "pass": If passed
    - "fail": If failed
    - "incomplete": If failed to complete 
    Contained in the "outcome" column
    """

    # Plot 3 bars per model: success rate , failure rate, incomplete rate per model
    outcome_per_model = df.groupby(["model", "outcome"]).size().reset_index(name="count")
    outcome_per_model["count"] = outcome_per_model.groupby("model")["count"].transform(lambda x: round(x / x.sum(), ROUND_N))
    
    # If any outcome is missing for a model, add it with count 0
    for model in df["model"].unique():
        for outcome in ["pass", "fail", "incomplete"]:
            if outcome not in outcome_per_model[outcome_per_model["model"] == model]["outcome"].tolist():
                outcome_per_model = pd.concat([outcome_per_model, pd.DataFrame({"model": model, "outcome": outcome, "count": 0}, index=[0])])

    # Pivot table
    outcome_per_model = outcome_per_model.pivot(index="model", columns="outcome", values="count").reset_index()
    
    # Plot
    options = copy.deepcopy(GLOBAL_OPTIONS)
    options["title"] = {"text": "Outcome rate per model"}
    options["xAxis"]["data"] = outcome_per_model["model"].tolist()
    options["series"] = [
        {"type": "bar", "name": "Success", "stack": "outcome", "data": outcome_per_model["pass"].tolist(), "itemStyle": {"color": colours[0]}},
        {"type": "bar", "name": "Failure", "stack": "outcome", "data": outcome_per_model["fail"].tolist(), "itemStyle": {"color": colours[1]}},
        {"type": "bar", "name": "Incomplete", "stack": "outcome", "data": outcome_per_model["incomplete"].tolist(), "itemStyle": {"color": colours[2]}},
    ]

    # Set max y-axis value to 1
    options["yAxis"]["max"] = 1

    # Remove marklines
    options["series"][0]["markLine"] = {}

    return (options, "outcome_per_model")

METHODS = [
    trial_count_per_model,
    pass_rate_task_id,
    pass_rate_per_model,
    outcome_per_model,
    # pass_rate_num_predict,
    # pass_rate_temperature,
    # pass_rate_ctx_n,
]

def generate_graphs(df) -> list[(dict, str)]:
    """
    Generate a set collection of graphs from a filtered Dataframe of results. Keys:
    - "model": Model name
    - "ctx_n": Context size 
    - "task_id": Task ID in evaluation set
    - "temperature": Temperature
    - "top_k": Top K
    - "top_p": Top P
    - "num_predict": Number of predictions to make during inference
    - "log": Whether to log or not
    - "evaluation_set": Evaluation set name
    - "passed": Whether the trial passed or not
    """

    # Remove empty trials, when len(shots) == 0
    df = df[df["shots"].apply(lambda x: len(x) > 0)]

    # Convert passed to numeric: 1 if df["passed"] == "True", 0 otherwise
    df = df.astype(str)  # Convert all columns to string
    df["passed"] = df["passed"].apply(lambda x: 1 if x == "True" else 0)    

    # Get pass rate of random performance when model is random
    random_pass_rate = df[df["method"] == "random"]["passed"].mean() if "random" in df["method"].tolist() else None

    # Drop random model from df
    df = df[df["method"] != "random"]

    # Check size 
    if len(df) == 0:
        return []

    # Add markLines to GLOBAL_OPTIONS
    GLOBAL_OPTIONS["series"][0]["markLine"] = {
        "precision": ROUND_N,
    }
    GLOBAL_OPTIONS["series"][0]["markLine"]["data"] = []

    if random_pass_rate:
        random_pass_rate = round(random_pass_rate, ROUND_N)
        GLOBAL_OPTIONS["series"][0]["markLine"]["data"].append({
                "name": "Random",
                "yAxis": random_pass_rate, 
                "label": {"show": True, "formatter": "Random: {:.3f}".format(random_pass_rate), "position": "insideEndTop"}, 
                "lineStyle": {"color": marker_colours[0]},
            }
        )

    average_pass_rate = df["passed"].mean()
    average_pass_rate = round(average_pass_rate, ROUND_N)
    GLOBAL_OPTIONS["series"][0]["markLine"]["data"].append({
                "name": "Average",
                "yAxis": average_pass_rate, 
                "label": {"show": True, "formatter": "Average: {:.3f}".format(average_pass_rate), "position": "insideEndTop"}, 
                "lineStyle": {"color": marker_colours[1]},
        }
    )

    return [method(df) for method in METHODS]