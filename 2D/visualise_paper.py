import argparse, glob, os, sys, json
import matplotlib.pyplot as plt
from matplotlib import rcParams
import pandas as pd
from scipy import stats
import numpy as np

LATEX_FIG_SCALE = 0.48
colors = ["#0072b2", "#d55e00", "#cc79a7"]

def plot_main_results(df):
    X_TICKS = ["E1", "E2"]
    rcParams['font.size'] = int(16)

    fig, axs = plt.subplots(1, 2, figsize=(12, 8))
    examples = 3
    pos = np.arange(len(X_TICKS))
    bar_width = 0.2
    space_between_groups = 0.01

    MODELS_TO_AVERAGE = ["Llama-2-70b-chat-hf"] #,  #, "chat-bison", "gpt-3.5-turbo"
    experiments = [(1,8), (2,9)]
    EXAMPLE_TO_PLOT = [0,1,2]

    results_dict = {
        "1": {
            EX: {
                "baseline": (df[(df["model"].isin(MODELS_TO_AVERAGE)) & (df["experiment"] == 8) & (df["examples"] == EX)]["error"].mean(), df[(df["model"].isin(MODELS_TO_AVERAGE)) & (df["experiment"] == 8) & (df["examples"] == EX)]["error"].std()),
                "non-baseline": (df[(df["model"].isin(MODELS_TO_AVERAGE)) & (df["experiment"] == 1) & (df["examples"] == EX)]["error"].mean(), df[(df["model"].isin(MODELS_TO_AVERAGE)) & (df["experiment"] == 1) & (df["examples"] == EX)]["error"].std()),
            } for EX in EXAMPLE_TO_PLOT
        },
        "2": {
            EX: {
                "baseline": (df[(df["model"].isin(MODELS_TO_AVERAGE)) & (df["experiment"] == 9) & (df["examples"] == EX)]["error"].mean(), df[(df["model"].isin(MODELS_TO_AVERAGE)) & (df["experiment"] == 9) & (df["examples"] == EX)]["error"].std()),
                "non-baseline": (df[(df["model"].isin(MODELS_TO_AVERAGE)) & (df["experiment"] == 2) & (df["examples"] == EX)]["error"].mean(), df[(df["model"].isin(MODELS_TO_AVERAGE)) & (df["experiment"] == 2) & (df["examples"] == EX)]["error"].std()),
            } for EX in EXAMPLE_TO_PLOT
        },
    }

    ### Ax 1: baseline
    # Two sets of bars, one for each experiment, with each bar being a model's baseline performance

    # Plot 6 bars, 3 for each experiment (one for each example) with std dev as error bars
    for i in range(examples):
        axs[0].bar(pos + i * (bar_width + space_between_groups), 
                [results_dict[str(experiments[0][0])][i]["baseline"][0], results_dict[str(experiments[1][0])][i]["baseline"][0]], 
                bar_width, 
                #yerr= [results_dict[str(experiments[0][0])][i]["baseline"][1], results_dict[str(experiments[1][0])][i]["baseline"][1]],
                label=f'{i}-Shot\n  CoT',
                color="none",
                edgecolor=colors[i],
                alpha=1-0.1*i,
                hatch="//")
        
    axs[0].set_ylim(0, 50)
    axs[0].set_yticks(np.arange(0, 60, 10))
    axs[0].set_ylabel("Error (|bounce-target|)")
    axs[0].set_xlabel("No Physics")

    middle_of_group = bar_width + (bar_width + space_between_groups) * (examples-3) / 2
    axs[0].set_xticks(pos + middle_of_group, X_TICKS, fontsize=16)

    # ### Ax 2: non-baseline
    # Two sets of bars, one for each experiment, with each bar being a model's non-baseline performance

    # Plot 6 bars, 3 for each experiment (one for each example)
    for i in range(examples):
        axs[1].bar(pos + i * (bar_width + space_between_groups), 
                [results_dict[str(experiments[0][0])][i]["non-baseline"][0], results_dict[str(experiments[1][0])][i]["non-baseline"][0]], 
                bar_width, 
                #yerr= [results_dict[str(experiments[0][0])][i]["non-baseline"][1], results_dict[str(experiments[1][0])][i]["non-baseline"][1]],
                label=f'{i}-Shot\nReSim',
                color=colors[i],
                alpha=1-0.1*i,)    
        
    axs[1].set_ylim(0, 50)
    axs[1].set_yticks(np.arange(0, 60, 10))
    axs[1].set_xlabel("Physics")

    middle_of_group = bar_width + (bar_width + space_between_groups) * (examples-3) / 2
    axs[1].set_xticks(pos + middle_of_group, X_TICKS, fontsize=16)

    # Add legend above both axes
    axs[0].legend(loc='upper center', bbox_to_anchor=(0.5, 1.16), ncol=3, fancybox=True, fontsize=16)
    axs[1].legend(loc='upper center', bbox_to_anchor=(0.5, 1.16), ncol=3, fancybox=True, fontsize=16)

    # Add grids
    axs[0].grid(axis="y")
    axs[1].grid(axis="y")

    fig.tight_layout(pad=.5)
    fig.savefig(f"figures/main_results.pdf")
def plot_difficulty_curve():
    pass
def plot_heatmaps():
    pass
def plot_difficulty_bar(df):
    rcParams['font.size'] = int(16)
    MODELS = ["chat-bison"]
    MODEL_TICKS = ["PaLM-2"]
    EXP = (10,11)

    for MODEL in MODELS:

        results = {
            "easy": {},
            "medium": {},
            "hard": {},
        }
        for D_NAME, DIFFICULTY in zip(["easy", "medium", "hard"], [[1,2,3],[4,5,6,7],[8,9,10]]):
            condition = (df["model"] == MODEL) & (df["difficulty"].isin(DIFFICULTY))
            results[D_NAME][0] =   df[condition & (df["examples"]==0) & (df["experiment"] == EXP[0])]["error"].mean() / df[condition & (df["examples"]==0) & (df["experiment"] == EXP[1])]["error"].mean()
            results[D_NAME][1] =   df[condition & (df["examples"]==1) & (df["experiment"] == EXP[0])]["error"].mean() / df[condition & (df["examples"]==1) & (df["experiment"] == EXP[1])]["error"].mean()
            results[D_NAME][2] =   df[condition & (df["examples"]==2) & (df["experiment"] == EXP[0])]["error"].mean() / df[condition & (df["examples"]==2) & (df["experiment"] == EXP[1])]["error"].mean()

        fig, ax = plt.subplots(1, 1, figsize=(12, 6))


        # Position of the bars on the x-axis
        pos = np.arange(3)

        # Width of a bar and space between groups
        bar_width = 0.2
        space_between_groups = 0.01
    
        # Creating the bars, one for each example
        for i, EXAMPLE in enumerate([0,1,2]):
            ax.bar(pos + i * bar_width + i * space_between_groups, [results["easy"][EXAMPLE], results["medium"][EXAMPLE], results["hard"][EXAMPLE]], bar_width, label=f"{EXAMPLE}-Shot", color=colors[i])

        # Labeling and showing the plot
        ax.set_xlabel("Difficulty")
        ax.set_ylabel("ReSim Error / CoT Error")
        
        ax.set_xticks(np.arange(0, 4, 1))
        ax.set_ylim(0.5,0.75)
        ax.set_yticks(np.arange(0.5, 0.755, 0.05))
        ax.legend(loc="best", fontsize=16)

        ax.grid(axis="y")

        # Adjust the position of the x-ticks to be in the middle of each group
        middle_of_group = bar_width / 2
        plt.xticks(pos + middle_of_group, ["Easy", "Medium", "Hard"])

        fig.tight_layout(pad=.5)

        fig.savefig(f"figures/difficulty_bar_{MODEL}.pdf")
def plot_main_results_ratio():
    pass

def plot_error_examples_target(df):
    EXPS = [1,2,8,9]
    MODEL = "chat-bison"
    EXAMPLES = [1,2]

    TO_PLOT = "error"
    # Plot correlation of error and examples_target
    with plt.style.context("seaborn-v0_8-colorblind"):
        fig, axs = plt.subplots(len(EXPS), len(EXAMPLES), figsize=(LATEX_FIG_SCALE*24, LATEX_FIG_SCALE*24))
        BINS = 100 #int(max(df[TO_PLOT]))
        XTICKS = 100 #int(max(df[TO_PLOT]))
        Y_LIM = 100

        # Plot for each experiment and example
        for i, EXP in enumerate(EXPS):
            for j, EX in enumerate(EXAMPLES):

                condition = (df["experiment"] == EXP) & (df["model"] == MODEL) & (df["examples"] == EX)

                ERRORS = df[condition][TO_PLOT]
                if j ==1:
                    EXAMPLE_TARGETS = df[condition]["example_target_1"]
                else:
                    EXAMPLE_TARGETS = df[condition]["example_target_1"] + df[condition]["example_target_2"]
                corr = ERRORS.corr(EXAMPLE_TARGETS)

                axs[i,j].scatter(EXAMPLE_TARGETS, ERRORS, alpha=0.25, label=f"Correlation: {corr:.2f}")

                axs[i,j].set_title(f"Model {MODEL}, Experiment {EXP}, Example {EX}")
                axs[i,j].set_xlabel("Example Target")
                axs[i,j].set_xlim(0, XTICKS)
                axs[i,j].set_ylabel("Error")
                axs[i,j].set_ylim(0, Y_LIM)
                axs[i,j].legend()
                axs[i,j].grid()

        plt.tight_layout()
        plt.show()

def plot_example_target_1(df):
    EXPS = [1,2,8,9]
    MODEL = "chat-bison"
    EXAMPLES = [0,1,2]
    TO_PLOT = "example_target_1"

    #Plot TO_PLOT distribution
    with plt.style.context("seaborn-v0_8-colorblind"):

        fig, axs = plt.subplots(len(EXPS), len(EXAMPLES), figsize=(LATEX_FIG_SCALE*24, LATEX_FIG_SCALE*24))
        BINS = int(max(df[TO_PLOT]))
        XTICKS = int(max(df[TO_PLOT]))
        Y_LIM = 0.15

        # Plot the score distribution for each experiment and example
        for i, EXP in enumerate(EXPS):
            for j, EX in enumerate(EXAMPLES):

                condition = (df["experiment"] == EXP) & (df["model"] == MODEL) & (df["examples"] == EX)

                axs[i,j].hist(df[condition][TO_PLOT], bins=BINS, density=True, label=f"Mean: {df[condition][TO_PLOT].mean():.2f}\nStd: {df[condition][TO_PLOT].std():.2f}\nMedian: {df[condition][TO_PLOT].median():.2f}")

                axs[i,j].set_title(f"Model {MODEL}, Experiment {EXP}, Example {EX}")

                axs[i,j].set_xlabel(TO_PLOT)
                axs[i,j].set_xlim(0, XTICKS)
                axs[i,j].set_ylabel("Density")
                axs[i,j].set_ylim(0, Y_LIM)
                axs[i,j].legend()
                axs[i,j].grid()

        plt.tight_layout()
        plt.show()

plot_dict = {
    "difficulty": plot_difficulty_curve,
    "heatmap": plot_heatmaps,
    "difficulty_bar": plot_difficulty_bar,
    "main": plot_main_results,
    "main_ratio": plot_main_results_ratio,
    "error_examples_target": plot_error_examples_target,
    "example_target_1": plot_example_target_1,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", action="store_true", help="Save the plot")
    parser.add_argument("--plot", type=str, default=None, help="Plot to visualize")
    args = parser.parse_args()

    # Set the font family and size to match your LaTeX document
    rcParams['font.family'] = 'serif'

    # Load all data as a dataframe
    MODELS = ["chat-bison", "gpt-3.5-turbo", "Llama-2-70b-chat-hf", "Llama-2-7b-chat-hf", "Llama-2-13b-chat-hf"]
    df = []
    for MODEL in MODELS:
            if not any(x in MODEL for x in ["bison", "gpt"]):
                MODEL += "/8_bit"
            results_folders = glob.glob(f"results/experiment_*/{MODEL}/*")
            for results_folder in results_folders:
                # Load each json in folder
                json_files = glob.glob(f"{results_folder}/*.json")
                for json_file in json_files:
                    with open(json_file, "r") as f:
                        data = json.load(f)
                    for TARGET in data.keys():
                        df += data[TARGET]
    df = pd.DataFrame(df)

    # Remove errors
    df = df[df["score"] != df["target"]]
    # Rename score to error
    df = df.rename(columns={"score": "error"})

    # condition = (df["model"] == "Llama-2-13b-chat-hf") & (df["examples"] == 1) & (df["experiment"] == 2)
    # arg_min = df[condition]["error"].argmax()
    # print("Arg Min Entry:", df[condition].iloc[arg_min])

    # Print average for each model, experiment, and example
    for MODEL in MODELS:
        for EXP in [1,8,2,9]:
            for EXAMPLES in [0,1,2]:
                condition = (df["model"] == MODEL) & (df["examples"] == EXAMPLES) & (df["experiment"] == EXP)
                results = df[condition]["error"]
                print(f"Model: {MODEL}, Experiment: {EXP}, Examples: {EXAMPLES}, Average: {results.mean():.2f}")

    # Perform automatic t-test on baseline and non-baseline for each model
    threshold = 0.05
    results = {}
    for MODEL in MODELS:
        for EXP_SET in [(1,8),(2,9)]:
            for EXAMPLES in [0,1,2]:
                condition = (df["model"] == MODEL) & (df["examples"] == EXAMPLES)
                baseline = df[condition & (df["experiment"]==EXP_SET[1])]["error"]
                non_baseline = df[condition & (df["experiment"]==EXP_SET[0])]["error"]
                t, p = stats.ttest_ind(baseline, non_baseline)
                if p < threshold:
                    results[f"{MODEL}, {EXP_SET}, {EXAMPLES}"] = f"count: {len(non_baseline)}/{len(baseline)}, p-value: {p:.2f}, t-value: {t:.2f}, SIGNIFICANT" 
                else:
                    results[f"{MODEL}, {EXP_SET}, {EXAMPLES}"] = f"count: {len(non_baseline)}/{len(baseline)}, p-value: {p:.2f}, t-value: {t:.2f}, NOT SIGNIFICANT" 

    print("T-TEST RESULTS")
    for key, value in results.items():
        print(key, value)

    # Plot the desired plot
    if args.plot is not None:
        plot_dict[args.plot](df)
    else:
        print("ERROR: No plot specified.")
    exit()

    print("----------------------------------")
    condition = (df["model"] == "gpt-3.5-turbo") & (df["examples"] == 0) & (df["experiment"].isin([1,2,8,9]) & (df["response"]).str.contains("1cm"))
    print(f"Valid Entries: {df[condition].shape[0]}/400")
    # plot_main_results(df[condition])


    EXPS = [1,2,8,9]
    MODEL = "gpt-3.5-turbo"
    EXAMPLES = [0]
    TO_PLOT = "error"

    # Plot TO_PLOT distribution
    with plt.style.context("seaborn-v0_8-colorblind"):

        fig, axs = plt.subplots(len(EXPS), len(EXAMPLES), figsize=(LATEX_FIG_SCALE*24, LATEX_FIG_SCALE*24))
        BINS = 100# int(max(df[TO_PLOT]))
        XTICKS = 100# int(max(df[TO_PLOT]))
        Y_LIM = 0.15

        # Plot the score distribution for each experiment and example
        for i, EXP in enumerate(EXPS):
            for j, EX in enumerate(EXAMPLES):

                condition = (df["experiment"] == EXP) & (df["model"] == MODEL) & (df["examples"] == EX)

                axs[i].hist(df[condition][TO_PLOT], bins=BINS, density=True, label=f"Mean: {df[condition][TO_PLOT].mean():.2f}\nStd: {df[condition][TO_PLOT].std():.2f}\nMedian: {df[condition][TO_PLOT].median():.2f}")

                axs[i].set_title(f"Model {MODEL}, Experiment {EXP}, Example {EX}")

                axs[i].set_xlabel(TO_PLOT)
                axs[i].set_xlim(0, XTICKS)
                axs[i].set_ylabel("Density")
                axs[i].set_ylim(0, Y_LIM)
                axs[i].legend()
                axs[i].grid()

        plt.tight_layout()
        plt.show()

    
        