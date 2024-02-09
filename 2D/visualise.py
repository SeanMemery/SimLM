import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob, os, argparse, json
from matplotlib import rcParams
from tqdm import tqdm
from sim import CustomSimulation
from experiments import sample_surface
import matplotlib.gridspec as gridspec

### TEMP
import warnings
warnings.filterwarnings("ignore")


def exp1(x):
    return 0
def exp2(x):
    return np.sin(x)

FUNCTIONS = {
    "Experiment 1": exp1,
    "Experiment 2": exp2,
    "Experiment 3\nDifficulty 1": sample_surface(1),
    "Experiment 3\nDifficulty 3": sample_surface(3),
    "Experiment 3\nDifficulty 6": sample_surface(6),
    "Experiment 3\nDifficulty 10": sample_surface(10),
}

colors = ["#0072b2", "#d55e00", "#cc79a7", "#e69f00"]

LATEX_FIG_SCALE = 0.24
PLOT_SIZE = 6
FONT_SIZE = 24

YLIM_UPPER = 135
YLIM_LOWER = -25

MAX_SCORE = 100

def plot_main_results():
    rcParams['font.size'] = int(16)
    MODELS_TO_AVERAGE = ["chat-bison"]#, "gpt-3.5-turbo", "Llama-2-70b-chat-hf"]
    X_TICKS = ["E1", "E2"]
    colors = ["#3498DB", "#E74C3C", "#34195E", "#2ECC71", "#F39C12", "#9B59B6", "#1ABC9C", "#34495E", "#E67E22", "#95A5A6"]
    experiments = [(12,12),(12,12)]
    avg_results_dicts = {
        M: {} for M in MODELS_TO_AVERAGE
    }

    ### Collect results
    for MODEL in MODELS_TO_AVERAGE:

        results_dict = avg_results_dicts[MODEL]

        for EXP in experiments:
            
            results_dict[str(EXP[0])] = {}
                
            MODEL = MODEL.split("/")[-1] 
            if not any(i in MODEL for i in ["bison", "gpt", "random"]):
                MODEL_DIR = f"{MODEL}/8_bit"
            else:
                MODEL_DIR = MODEL

            results_folder_nb = f"results/experiment_{EXP[0]}/{MODEL_DIR}/"
            results_folder_b = f"results/experiment_{EXP[1]}/{MODEL_DIR}/"
            folders_nb = glob.glob(results_folder_nb + "*")
            folders_b = glob.glob(results_folder_b + "*")

            ### Get all jsons in the folders
            total_jsons = {"non-baseline": {}, "baseline": {}}
            for k in total_jsons.keys():
                folders = folders_nb if k == "non-baseline" else folders_b
                for f in folders:
                    jsons = glob.glob(f"{f}/*.json")     
                    jsons.sort()
                    for j in jsons:
                        try:
                            EXAMPLE = int(j.split("/")[-1].split("_")[1].split(".")[0])
                        except:
                            continue
                        total_jsons[k][EXAMPLE] = total_jsons[k].get(EXAMPLE, []) + [j]

            EXAMPLE_TO_PLOT = [0,1,2]
            for EXAMPLE in EXAMPLE_TO_PLOT:
                
                results_dict[str(EXP[0])][EXAMPLE] = {}

                baseline_mean = 0
                baseline_std = 0
                b_count = 0
                non_baseline_mean = 0
                non_baseline_std = 0
                nb_count = 0
                
                if not EXAMPLE in total_jsons["baseline"].keys() or not EXAMPLE in total_jsons["non-baseline"].keys():
                    results_dict[str(EXP[0])][EXAMPLE] = {"baseline": 0, "non-baseline": 0}
                    continue

                for j in total_jsons["baseline"][EXAMPLE]:
                    j = json.load(open(j, "r"))
                    for key, value in j.items():
                        for ENTRY in value:
                            if ENTRY["score"] == ENTRY["target"]:
                                continue
                            baseline_mean += ENTRY["score"]
                            b_count += 1
                baseline_mean = baseline_mean / b_count if b_count > 0 else 0

                for j in total_jsons["non-baseline"][EXAMPLE]:
                    j = json.load(open(j, "r"))
                    for key, value in j.items():
                        for ENTRY in value:
                            if ENTRY["score"] == ENTRY["target"]:
                                continue
                            non_baseline_mean += ENTRY["score"]
                            nb_count += 1
                non_baseline_mean = non_baseline_mean / nb_count if nb_count > 0 else 0

                for j in total_jsons["baseline"][EXAMPLE]:
                    j = json.load(open(j, "r"))
                    for key, value in j.items():
                        for ENTRY in value:
                            if ENTRY["score"] == ENTRY["target"]:
                                continue
                            baseline_std += (ENTRY["score"] - baseline_mean)**2
                baseline_std = np.sqrt(baseline_std / b_count) if b_count > 0 else 0

                for j in total_jsons["non-baseline"][EXAMPLE]:
                    j = json.load(open(j, "r"))
                    for key, value in j.items():
                        for ENTRY in value:
                            if ENTRY["score"] == ENTRY["target"]:
                                continue
                            non_baseline_std += (ENTRY["score"] - non_baseline_mean)**2
                non_baseline_std = np.sqrt(non_baseline_std / nb_count) if nb_count > 0 else 0

                results_dict[str(EXP[0])][EXAMPLE]["baseline"] = (baseline_mean, baseline_std)
                results_dict[str(EXP[0])][EXAMPLE]["non-baseline"] = (non_baseline_mean, non_baseline_std)

    ### Avg results
    results_dict = {}
    for MODEL in MODELS_TO_AVERAGE:
        for EXP in experiments:
            for EXAMPLE in EXAMPLE_TO_PLOT:
                results_dict[str(EXP[0])] = results_dict.get(str(EXP[0]), {})
                results_dict[str(EXP[0])][EXAMPLE] = results_dict[str(EXP[0])].get(EXAMPLE, {})
                results_dict[str(EXP[0])][EXAMPLE]["baseline"] = results_dict[str(EXP[0])][EXAMPLE].get("baseline", [])
                results_dict[str(EXP[0])][EXAMPLE]["non-baseline"] = results_dict[str(EXP[0])][EXAMPLE].get("non-baseline", [])

                results_dict[str(EXP[0])][EXAMPLE]["baseline"] = (np.mean([avg_results_dicts[MODEL][str(EXP[0])][EXAMPLE]["baseline"][0] for MODEL in MODELS_TO_AVERAGE]),
                                                                    np.mean([avg_results_dicts[MODEL][str(EXP[0])][EXAMPLE]["baseline"][1] for MODEL in MODELS_TO_AVERAGE]))
                results_dict[str(EXP[0])][EXAMPLE]["non-baseline"] = (np.mean([avg_results_dicts[MODEL][str(EXP[0])][EXAMPLE]["non-baseline"][0] for MODEL in MODELS_TO_AVERAGE]),
                                                                    np.mean([avg_results_dicts[MODEL][str(EXP[0])][EXAMPLE]["non-baseline"][1] for MODEL in MODELS_TO_AVERAGE]))

    fig, axs = plt.subplots(1, 2, figsize=(12, 8))
    examples = 3
    pos = np.arange(len(X_TICKS))
    bar_width = 0.2
    space_between_groups = 0.01

    ### Ax 1: baseline
    # Two sets of bars, one for each experiment, with each bar being a model's baseline performance

    # Plot 6 bars, 3 for each experiment (one for each example) with std dev as error bars
    for i in range(examples):
        axs[0].bar(pos + i * (bar_width + space_between_groups), 
                [results_dict[str(experiments[0][0])][i]["baseline"][0], results_dict[str(experiments[1][0])][i]["baseline"][0]], 
                bar_width, 
                yerr= [results_dict[str(experiments[0][0])][i]["baseline"][1], results_dict[str(experiments[1][0])][i]["baseline"][1]],
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
                yerr= [results_dict[str(experiments[0][0])][i]["non-baseline"][1], results_dict[str(experiments[1][0])][i]["non-baseline"][1]],
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

def plot_main_results_ratio():
    """
    Plots the main results for the paper, for each model with a relative error of baseline vs non-baseline
    """

    rcParams['font.size'] = int(12/LATEX_FIG_SCALE)
    MODELS = ["chat-bison", "gpt-3.5-turbo"] #, "Llama-2-70b-chat-hf"
    MODEL_TICKS = ["PaLM-2", "GPT-3.5"] #, "Llama-2\n70b"
    colors = ["#3498DB", "#E74C3C", "#34195E", "#2ECC71", "#F39C12", "#9B59B6", "#1ABC9C", "#34495E", "#E67E22", "#95A5A6"]
    experiments = [(1,8), (2,9)]

    for EXP in experiments:

        results_dict = {}

        for M, MODEL in enumerate(MODELS):

            results_dict[MODEL] = {}
                
            MODEL = MODEL.split("/")[-1] 
            if not any(i in MODEL for i in ["bison", "gpt", "random"]):
                MODEL_DIR = f"{MODEL}/8_bit"
            else:
                MODEL_DIR = MODEL

            results_folder_nb = f"results/experiment_{EXP[0]}/{MODEL_DIR}/"
            results_folder_b = f"results/experiment_{EXP[1]}/{MODEL_DIR}/"
            folders_nb = glob.glob(results_folder_nb + "*")
            folders_b = glob.glob(results_folder_b + "*")

            ### Get all jsons in the folders
            total_jsons = {"non-baseline": {}, "baseline": {}}
            for k in total_jsons.keys():
                folders = folders_nb if k == "non-baseline" else folders_b
                for f in folders:
                    jsons = glob.glob(f"{f}/*.json")     
                    jsons.sort()
                    for j in jsons:
                        try:
                            EXAMPLE = int(j.split("/")[-1].split("_")[1].split(".")[0])
                        except:
                            continue
                        total_jsons[k][EXAMPLE] = total_jsons[k].get(EXAMPLE, []) + [j]

            EXAMPLE_TO_PLOT = [0,1,2]
            for EXAMPLE in EXAMPLE_TO_PLOT:

                baseline_mean = 0
                b_count = 0
                non_baseline_mean = 0
                nb_count = 0
                
                if not EXAMPLE in total_jsons["baseline"].keys() or not EXAMPLE in total_jsons["non-baseline"].keys():
                    results_dict[MODEL][EXAMPLE] = 0
                    continue

                for j in total_jsons["baseline"][EXAMPLE]:
                    j = json.load(open(j, "r"))
                    for key, value in j.items():
                        for ENTRY in value:
                            if ENTRY["score"] == ENTRY["target"]:
                                continue
                            baseline_mean += ENTRY["score"]
                            b_count += 1
                baseline_mean = baseline_mean / b_count if b_count > 0 else 0

                for j in total_jsons["non-baseline"][EXAMPLE]:
                    j = json.load(open(j, "r"))
                    for key, value in j.items():
                        for ENTRY in value:
                            if ENTRY["score"] == ENTRY["target"]:
                                continue
                            non_baseline_mean += ENTRY["score"]
                            nb_count += 1
                non_baseline_mean = non_baseline_mean / nb_count if nb_count > 0 else 0

                try:
                    relative_error = non_baseline_mean / baseline_mean
                except:
                    print(f"0 baseline_mean: example {EXAMPLE} for {MODEL} and experiment {EXP}")
                    continue

                results_dict[MODEL][EXAMPLE] = relative_error

        fig, ax = plt.subplots(1, 1, figsize=(12, 10))

        # Extract model names and their corresponding values
        models = list(results_dict.keys())
        values = [list(val.values()) for val in results_dict.values()]

        # Number of models and examples
        num_models = len(models)
        examples = len(values[0])

        # Position of the bars on the x-axis
        pos = np.arange(num_models)

        # Width of a bar and space between groups
        bar_width = 0.2
        space_between_groups = 0.01
    
        # Add dashed line at y=1
        ax.axhline(y=1, color='black', linestyle='--', linewidth=2, alpha=0.75)

        # Creating the bars
        for i in range(examples):
            plt.bar(pos + i * (bar_width + space_between_groups), 
                    [value[i] for value in values], 
                    bar_width, 
                    label=f'{i}-Shot',
                    color=colors[i],
                    alpha=1-0.1*i,)
            
        # Plot good and bad dashed line
        ax.axhline(y=0.5, color='green', linestyle='--', linewidth=2, alpha=0.5)
        ax.axhline(y=1.5, color='red', linestyle='--', linewidth=2, alpha=0.5)

        # Labeling and showing the plot
        ax.set_xlabel("Models")
        ax.set_ylabel("Relative Error")
        
        ax.set_xticks(np.arange(0, 4, 1))
        ax.set_ylim(0,2)
        ax.legend()

        # Adjust the position of the x-ticks to be in the middle of each group
        middle_of_group = bar_width + (bar_width + space_between_groups) * (examples - 3) / 2
        plt.xticks(pos + middle_of_group, MODEL_TICKS, rotation=45)

        fig.tight_layout(pad=.5)

        fig.savefig(f"figures/main_results_exp{EXP[0]}_exp{EXP[1]}.pdf")

def plot_difficulty_bar():
    """
    Plots the results of experiment 3, for each model with a relative error of baseline vs non-baseline
    """

    rcParams['font.size'] = int(16)
    MODELS = ["chat-bison"]
    MODEL_TICKS = ["PaLM-2"]
    colors = ["#3498DB", "#E74C3C", "#34195E", "#2ECC71", "#F39C12", "#9B59B6", "#1ABC9C", "#34495E", "#E67E22", "#95A5A6"]
    experiments = [(10,11)]

    for EXP in experiments:

        results_dict = {}

        for M, MODEL in enumerate(MODELS):

            results_dict[MODEL] = {}
                
            MODEL = MODEL.split("/")[-1] 
            if not any(i in MODEL for i in ["bison", "gpt", "random"]):
                MODEL_DIR = f"{MODEL}/8_bit"
            else:
                MODEL_DIR = MODEL

            print("Visualizing main:", MODEL)
            results_folder_nb = f"OLD/results-2023-12-03-(pre-fixed)/results-2023-12-01/experiment_{EXP[0]}/{MODEL_DIR}/"
            results_folder_b = f"OLD/results-2023-12-03-(pre-fixed)/results-2023-12-01/experiment_{EXP[1]}/{MODEL_DIR}/"
            folders_nb = glob.glob(results_folder_nb + "*")
            folders_b = glob.glob(results_folder_b + "*")

            ### Get all jsons in the folders
            total_jsons = {"non-baseline": {}, "baseline": {}}
            for k in total_jsons.keys():
                folders = folders_nb if k == "non-baseline" else folders_b
                for f in folders:
                    jsons = glob.glob(f"{f}/*.json")     
                    jsons.sort()
                    for j in jsons:
                        try:
                            EXAMPLE = int(j.split("/")[-1].split("_")[1].split(".")[0])
                        except:
                            continue
                        total_jsons[k][EXAMPLE] = total_jsons[k].get(EXAMPLE, []) + [j]

            EXAMPLE_TO_PLOT = [0,1,2]
            for EXAMPLE in EXAMPLE_TO_PLOT:

                baseline_mean = [0,0,0]
                baseline_std_dev = [0,0,0]
                b_count = [0,0,0]
                non_baseline_mean = [0,0,0]
                non_baseline_std_dev = [0,0,0]
                nb_count = [0,0,0]
                
                if not EXAMPLE in total_jsons["baseline"].keys() or not EXAMPLE in total_jsons["non-baseline"].keys():
                    results_dict[MODEL][EXAMPLE] = {"easy": 0, "medium": 0, "hard": 0}
                    continue

                for j in total_jsons["baseline"][EXAMPLE]:
                    j = json.load(open(j, "r"))
                    for key, value in j.items():
                        for ENTRY in value:
                            if ENTRY["score"] == ENTRY["target"]:
                                continue
                            if ENTRY["difficulty"] in [1,2,3]:
                                baseline_mean[0] += ENTRY["score"]
                                b_count[0] += 1
                            elif ENTRY["difficulty"] in [4,5,6,7]:
                                baseline_mean[1] += ENTRY["score"]
                                b_count[1] += 1
                            elif ENTRY["difficulty"] in [8,9,10]:
                                baseline_mean[2] += ENTRY["score"]
                                b_count[2] += 1
                            else:
                                continue
                baseline_mean[0] = baseline_mean[0] / b_count[0] if b_count[0] > 0 else 0
                baseline_mean[1] = baseline_mean[1] / b_count[1] if b_count[1] > 0 else 0
                baseline_mean[2] = baseline_mean[2] / b_count[2] if b_count[2] > 0 else 0

                for j in total_jsons["baseline"][EXAMPLE]:
                    j = json.load(open(j, "r"))
                    for key, value in j.items():
                        for ENTRY in value:
                            if ENTRY["score"] == ENTRY["target"]:
                                continue
                            if ENTRY["difficulty"] in [1,2,3]:
                                baseline_std_dev[0] += (ENTRY["score"] - baseline_mean[0])**2
                            elif ENTRY["difficulty"] in [4,5,6,7]:
                                baseline_std_dev[1] += (ENTRY["score"] - baseline_mean[1])**2
                            elif ENTRY["difficulty"] in [8,9,10]:
                                baseline_std_dev[2] += (ENTRY["score"] - baseline_mean[2])**2
                            else:
                                continue

                baseline_std_dev[0] = np.sqrt(baseline_std_dev[0] / b_count[0]) if b_count[0] > 0 else 0
                baseline_std_dev[1] = np.sqrt(baseline_std_dev[1] / b_count[1]) if b_count[1] > 0 else 0
                baseline_std_dev[2] = np.sqrt(baseline_std_dev[2] / b_count[2]) if b_count[2] > 0 else 0

                for j in total_jsons["non-baseline"][EXAMPLE]:
                    j = json.load(open(j, "r"))
                    for key, value in j.items():
                        for ENTRY in value:
                            if ENTRY["score"] == ENTRY["target"]:
                                continue
                            if ENTRY["difficulty"] in [1,2,3]:
                                non_baseline_mean[0] += ENTRY["score"]
                                nb_count[0] += 1
                            elif ENTRY["difficulty"] in [4,5,6,7]:
                                non_baseline_mean[1] += ENTRY["score"]
                                nb_count[1] += 1
                            elif ENTRY["difficulty"] in [8,9,10]:
                                non_baseline_mean[2] += ENTRY["score"]
                                nb_count[2] += 1
                            else:
                                continue      
                non_baseline_mean[0] = non_baseline_mean[0] / nb_count[0] if nb_count[0] > 0 else 0
                non_baseline_mean[1] = non_baseline_mean[1] / nb_count[1] if nb_count[1] > 0 else 0
                non_baseline_mean[2] = non_baseline_mean[2] / nb_count[2] if nb_count[2] > 0 else 0

                for j in total_jsons["baseline"][EXAMPLE]:
                    j = json.load(open(j, "r"))
                    for key, value in j.items():
                        for ENTRY in value:
                            if ENTRY["score"] == ENTRY["target"]:
                                continue
                            if ENTRY["difficulty"] in [1,2,3]:
                                non_baseline_std_dev[0] += (ENTRY["score"] - non_baseline_mean[0])**2
                            elif ENTRY["difficulty"] in [4,5,6,7]:
                                non_baseline_std_dev[1] += (ENTRY["score"] - non_baseline_mean[1])**2
                            elif ENTRY["difficulty"] in [8,9,10]:
                                non_baseline_std_dev[2] += (ENTRY["score"] - non_baseline_mean[2])**2
                            else:
                                continue

                non_baseline_std_dev[0] = np.sqrt(non_baseline_std_dev[0] / b_count[0]) if b_count[0] > 0 else 0
                non_baseline_std_dev[1] = np.sqrt(non_baseline_std_dev[1] / b_count[1]) if b_count[1] > 0 else 0
                non_baseline_std_dev[2] = np.sqrt(non_baseline_std_dev[2] / b_count[2]) if b_count[2] > 0 else 0

                print("-------")
                print(f"Std dev Baseline: {baseline_std_dev}, \nStd dev Non-Baseline {non_baseline_std_dev}")

                try:
                    results_dict[MODEL][EXAMPLE] = {"easy": non_baseline_mean[0] / baseline_mean[0], "medium": non_baseline_mean[1] / baseline_mean[1], "hard": non_baseline_mean[2] / baseline_mean[2]}
                except:
                    print(f"0 baseline_mean: example {EXAMPLE} for {MODEL} and experiment {EXP}")
                    continue

            fig, ax = plt.subplots(1, 1, figsize=(8, 8))

            # Extract values
            values = list(results_dict[MODEL].values())
            examples = len(values[0])

            # Position of the bars on the x-axis
            pos = np.arange(3)

            # Width of a bar and space between groups
            bar_width = 0.2
            space_between_groups = 0.01
        
            # Creating the bars
            for i, d in enumerate(["easy", "medium", "hard"]):
                plt.bar(pos + i * (bar_width + space_between_groups), 
                        values[i].values(), 
                        bar_width, 
                        label=f'{i}-Shot',
                        color=colors[i],
                        alpha=1-0.1*i,)

            # Labeling and showing the plot
            ax.set_xlabel("Difficulty")
            ax.set_ylabel("Relative Error")
            
            ax.set_xticks(np.arange(0, 4, 1))
            ax.set_ylim(0.5,0.875)
            ax.set_yticks(np.arange(0.5, 1, 0.125))
            ax.legend()

            ax.grid(axis="y")

            # Adjust the position of the x-ticks to be in the middle of each group
            middle_of_group = bar_width + (bar_width + space_between_groups) * (examples - 3) / 2
            plt.xticks(pos + middle_of_group, ["Easy", "Medium", "Hard"])

            fig.tight_layout(pad=.5)

            fig.savefig(f"figures/difficulty_bar_{MODEL}.pdf")

def plot_difficulty_curve():
    rcParams['font.size'] = int(16)
    MODELS = ["chat-bison"] #, "gpt-3.5-turbo", "Llama-2-7b-chat-hf", "Llama-2-13b-chat-hf", "Llama-2-70b-chat-hf"
    X_TICKS = ["PaLM-2"]

    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    for M, MODEL in enumerate(MODELS):
        
        MODEL = MODEL.split("/")[-1] 
        if not any(i in MODEL for i in ["bison", "gpt", "random"]):
            MODEL += f"/8_bit"

        print("Visualizing difficulty:", MODEL)
        results_folder_nb = f"results/experiment_10/{MODEL}/"
        results_folder_b = f"results/experiment_11/{MODEL}/"
        folders_nb = glob.glob(results_folder_nb + "*")
        folders_b = glob.glob(results_folder_b + "*")

        ### Get all jsons in the folders
        total_jsons = {"non-baseline": {}, "baseline": {}}
        for k in total_jsons.keys():
            folders = folders_nb if k == "non-baseline" else folders_b
            for f in folders:
                jsons = glob.glob(f"{f}/*.json")     
                jsons.sort()
                for j in jsons:
                    try:
                        EXAMPLE = int(j.split("/")[-1].split("_")[1].split(".")[0])
                    except:
                        continue
                    total_jsons[k][EXAMPLE] = total_jsons[k].get(EXAMPLE, []) + [j]

        NUM_EXAMPLES = len(total_jsons["non-baseline"].keys())
        EXAMPLE_TO_PLOT = [2]
        for EXAMPLE in EXAMPLE_TO_PLOT:
            if len(total_jsons["non-baseline"]) > 0:                
                jsons = total_jsons["non-baseline"][EXAMPLE]

                # Load the JSON data into a DataFrame
                json_full = {}
                for j in jsons:
                    j = json.load(open(j, "r"))
                    for key, value in j.items():
                        json_full[key] = json_full.get(key, []) + list(value)

                # Average the scores over each difficulty level 
                df_difficulty_nb = {}
                freq_counts_nb = {}
                for TARGET in json_full.keys():
                    for ENTRY in json_full[TARGET]:
                        ### IMPORTANT: SKIP IF FAILURE ###
                        if ENTRY["score"] == ENTRY["target"]:
                            continue
                        df_difficulty_nb[ENTRY["difficulty"]] = df_difficulty_nb.get(ENTRY["difficulty"], 0) + ENTRY["score"]
                        freq_counts_nb[ENTRY["difficulty"]] = freq_counts_nb.get(ENTRY["difficulty"], 0) + 1
                for k in df_difficulty_nb.keys():
                    df_difficulty_nb[k] = df_difficulty_nb[k] / freq_counts_nb[k] if freq_counts_nb[k] > 0 else 0
                df_difficulty_nb = pd.DataFrame(df_difficulty_nb.items(), columns=["difficulty", "score"])

                # Sort by difficulty
                df_difficulty_nb = df_difficulty_nb.sort_values(by=["difficulty"])

                # Get std dev
                mean_score = df_difficulty_nb["score"].mean()
                std_score = df_difficulty_nb['score'].std()
                
                # Plot data and lines connecting them
                ax.plot(df_difficulty_nb["difficulty"], df_difficulty_nb["score"], color=colors[3+M], alpha=1, label=f"{X_TICKS[0]} ReSim")
                #ax.scatter(df_difficulty_nb["difficulty"], df_difficulty_nb["score"], color=colors[-(M+1)], alpha=1)

                # Plot shaded region
                ax.fill_between(df_difficulty_nb["difficulty"], df_difficulty_nb["score"] + std_score, df_difficulty_nb["score"] - std_score, color=colors[3+M], alpha=0.2)

            if len(total_jsons["baseline"]) > 0:
                # Average the scores over each difficulty level
                jsons = total_jsons["baseline"][EXAMPLE]

                # Load the JSON data into a DataFrame
                json_full = {}
                for j in jsons:
                    j = json.load(open(j, "r"))
                    for key, value in j.items():
                        json_full[key] = json_full.get(key, []) + list(value)

                df_difficulty_b = {}
                freq_counts_b = {}
                for TARGET in json_full.keys():
                    for ENTRY in json_full[TARGET]:
                        ### IMPORTANT: SKIP IF FAILURE ###
                        if ENTRY["score"] == ENTRY["target"]:
                            continue
                        df_difficulty_b[ENTRY["difficulty"]] = df_difficulty_b.get(ENTRY["difficulty"], 0) + ENTRY["score"]
                        freq_counts_b[ENTRY["difficulty"]] = freq_counts_b.get(ENTRY["difficulty"], 0) + 1
                for k in df_difficulty_b.keys():
                    df_difficulty_b[k] = df_difficulty_b[k] / freq_counts_b[k] if freq_counts_b[k] > 0 else 0
                df_difficulty_b = pd.DataFrame(df_difficulty_b.items(), columns=["difficulty", "score"])

                df_difficulty_b = df_difficulty_b.sort_values(by=["difficulty"])

                # Get std dev
                mean_score = df_difficulty_nb["score"].mean()
                std_score = df_difficulty_nb['score'].std()

                ax.plot(df_difficulty_b["difficulty"], df_difficulty_b["score"], color=colors[3+M], alpha=0.5, label=f"{X_TICKS[0]} CoT", linestyle="--")
                #ax.scatter(df_difficulty_b["difficulty"], df_difficulty_b["score"], color=colors[-(M+1)], alpha=0.5, marker="x")

                # Plot shaded region
                ax.fill_between(df_difficulty_b["difficulty"], df_difficulty_b["score"] + std_score, df_difficulty_b["score"] - std_score, color=colors[3+M], alpha=0.1)

    # Labeling and showing the plot
    ax.set_xlabel("Difficulty")
    ax.set_ylabel("Error (|bounce-target|)")
    #ax.set_title(f"Difficulty vs Difference")
    ax.set_ylim(0, 50)
    ax.set_xticks(np.arange(1, 11, 1))
    ax.legend()
    plt.grid()

    #plt.show()

    # Save
    plt.tight_layout(pad=.5)
    fig.savefig(f"figures/difficulty_curve.pdf")
    plt.close()

def plot_surface(axs, i, f, f_n):
    sim = CustomSimulation(f)

    TARGET = 50
    H_LIM = (1, 50)
    V_LIM = (1, 50)
    N = 25
    RESULTS = []

    # # Add a single IC for each HxV combination
    # ICS = []
    # for h in np.linspace(H_LIM[0], H_LIM[1], N):
    #     for v in np.linspace(V_LIM[0], V_LIM[1], N):
    #         ICS.append({
    #             "height": h,
    #             "horizontal_velocity": v
    #         })

    ICS = [{
        "height": np.random.uniform(H_LIM[0], H_LIM[1]),
        "horizontal_velocity": np.random.uniform(V_LIM[0], V_LIM[1])
    } for _ in range(N*N)]

    try: 
        RESULTS = json.load(open(f"figures/heatmap_data/{f_n}.json", "r"))
    except:
        RESULTS = []
    
    for IC in tqdm(ICS, desc="Running Simulations"):
        bounces = sim.get_bounces(IC["height"],IC["horizontal_velocity"])
        RESULTS.append({
            "score": abs(bounces[-1]-TARGET), 
            "bounces": bounces, 
            "target": TARGET, 
            "IC": IC
        })
    #json.dump(RESULTS, open(f"figures/heatmap_data/{f_n}.json", "w"))

    df = pd.DataFrame(RESULTS)

    # Plot heatmap of scores
    v = df["IC"].apply(lambda x: x["horizontal_velocity"])
    h = df["IC"].apply(lambda x: x["height"])
    s = df["score"]

    # Set a max score of 100 
    s = s.apply(lambda x: min(x, 100))

    hb = axs[i].hexbin(v, h, C=s, gridsize=100, cmap=plt.colormaps["viridis"].reversed(), vmin=0, vmax=100)
    axs[i].set_title(f"{f_n}") if i in [0,1] else axs[i].set_title(f"{f_n}/10")
    axs[i].set_xlim(0, 50)
    axs[i].set_ylim(0, 50)
    axs[i].set_xlabel("Horizontal Velocity")
    axs[i].set_xticks(np.arange(0, 60, 10))
    if i == 0:
        axs[i].set_ylabel("Height")
        axs[i].set_yticks(np.arange(0, 60, 10))

    # Set aspect of each subplot to be equal, making them square
    axs[i].set_aspect('equal', adjustable='box')

    return hb

def plot_heatmaps():
    """
    Plots the heatmaps for surfaces of experiment 1 and 2, and for difficulty 1,3,6,10 of experiment 3
    """

    rcParams['font.size'] = int(16)

    fig = plt.figure(figsize=(int(14/(LATEX_FIG_SCALE)), int(3/(LATEX_FIG_SCALE))))
    gs = gridspec.GridSpec(1, len(FUNCTIONS)+1, width_ratios=[1]*len(FUNCTIONS) + [0.1], height_ratios=[1]) 
    axs = [plt.subplot(gs[i]) for i in range(len(FUNCTIONS))]

    for i, (f_n, f) in enumerate(FUNCTIONS.items()):
        hb = plot_surface(axs, i, f, f_n)

    cax = plt.subplot(gs[-1])
    cb = plt.colorbar(hb, cax=cax, orientation='vertical')
    cb.set_ticks([20, 40, 60, 80, 100])
    fig.tight_layout(pad=.5)

    fig.savefig(f"figures/heatmaps.pdf")

def plot_exp_no_variation(json_full, ax1, ax2, EXAMPLE):

    rcParams['font.size'] = int(FONT_SIZE / LATEX_FIG_SCALE)

    # Convert to DataFrame
    df = []
    for TARGET in json_full.keys():
        for ENTRY in json_full[TARGET]:
            df.append({"target": ENTRY["target"], "score": ENTRY["score"], "last_bounce": ENTRY["bounces"][-1], "attempts": ENTRY["attempts"]})
    df = pd.DataFrame(df)

    # Remove if score == target
    df = df[df["score"] != df["target"]]

    # If empty, skip
    if len(df) == 0:
        return

    # Calculate best fit lines for both dataframes
    z = np.polyfit(df['target'], df['score'], 1)
    p = np.poly1d(z)

    # Calculate standard deviations and mean for the difference column
    mean_score = df["score"].mean()
    std_score = df['score'].std()

    print(f"Mean: {mean_score:.2f}, Std: {std_score:.2f}")

    # Plot data and best fit lines with shaded regions
    x = np.linspace(df['target'].min(), df['target'].max(), 100)
    ax1.plot(x, p(x), color=colors[EXAMPLE], label=f"{EXAMPLE}-Shot", linewidth=int(3/(LATEX_FIG_SCALE)))
    ax1.fill_between(x, p(x) + std_score, p(x) - std_score, color=shade_colors[EXAMPLE], alpha=0.1)

    # Scatter plot
    ax1.scatter(df['target'], df['score'], color=colors[EXAMPLE], alpha=0.4, linewidth=2/LATEX_FIG_SCALE)

    # Labeling and showing the plot
    ax1.set_xlabel("Target X Position")
    ax1.xaxis.set_ticks(np.arange(0, 110, 50))
    ax1.set_yticks(np.arange(0, 110, 50))
    ax1.set_ylim(0, 100)
    ax1.set_ylabel("Error (|bounce-target|)")
    #ax1.set_title(f"Target vs Difference\nModel: {MODEL}")
    ax1.legend(loc = "upper left", fontsize=str(int(16/LATEX_FIG_SCALE)))

    # Plot the SCORE - PROB graph
    min_val, max_val = 0, 50  
    GRANULARITY = 5
    BINS = int(max_val / GRANULARITY)
    # Plot histogram for scores in the first subplot with defined range and increased bins
    ax2[0].hist(df["score"], bins=BINS, range=(min_val, max_val), color='lightblue', edgecolor='black', alpha=0.7, density=True)
    ax2[0].axvline(mean_score, color='red', linestyle='dashed', linewidth=1, label=f'Avg: {mean_score:.2f}')
    ax2[0].axvline(mean_score - std_score, color='green', linestyle='dashed', linewidth=1, label=f'Std Dev: {std_score:.2f}')
    ax2[0].axvline(mean_score + std_score, color='green', linestyle='dashed', linewidth=1)
    ax2[0].set_xlim(min_val, max_val)  # This limits the x-axis range
    Y_MAX = 1.0 / GRANULARITY
    ax2[0].set_ylim(0, Y_MAX)  # This limits the y-axis range
    ax2[0].yaxis.set_ticks(np.arange(0, Y_MAX + Y_MAX / GRANULARITY, Y_MAX / GRANULARITY))  
    ax2[0].set_title(f"Distribution of Scores\nModel: {MODEL}, Examples: {EXAMPLE}")
    ax2[0].set_xlabel('Score Value')
    ax2[0].set_ylabel('Probability Density')
    ax2[0].legend(loc = 'upper left')

    # Plot the TARGET - LAST BOUNCE graph
    ax2[1].scatter(df["target"], df["last_bounce"], alpha=0.4, edgecolors="w", linewidth=0.5)
    # Adding y=x line
    ax2[1].plot(df["target"], df["target"], color='gray', label='y=x')
    # Calculating and plotting best fit line
    z = np.polyfit(df["target"], df["last_bounce"], 1)
    p = np.poly1d(z)
    # Get accuracy
    predicted_y = p(df["target"])
    actual_y = df["target"]
    mae = np.mean(abs(predicted_y - actual_y))
    ax2[1].plot(df["target"], p(df["target"]), color='red', label=f'Best Fit, MAE: {mae:.4f}')
    ax2[1].set_title(f'Scatter plot of Targets vs. Last Bounce')
    ax2[1].set_xlabel('Target')
    ax2[1].set_ylabel('Last Bounce')
    ax2[1].set_xlim(0, 100)
    ax2[1].set_ylim(-25, 125)
    ax2[1].xaxis.set_ticks(np.arange(0, 110, 10))
    ax2[1].yaxis.set_ticks(np.arange(YLIM_LOWER, YLIM_UPPER ,10))
    ax2[1].legend(loc = 'upper left')

def plot_exp_variation(json_full, ax1, ax2, EXAMPLE, vary_param):

    # Average the scores and last bounces over each target
    df_target = []
    for TARGET in json_full.keys():
        avg_score = 0
        avg_last_bounce = 0
        num_entries = 0
        for ENTRY in json_full[TARGET]:
            ### IMPORTANT: SKIP IF FAILURE ###
            if ENTRY["score"] == ENTRY["target"]:
                continue
            num_entries += 1
            avg_score += ENTRY["score"] 
            avg_last_bounce += ENTRY["bounces"][-1] 
        avg_score = avg_score / num_entries if num_entries > 0 else 0   
        avg_last_bounce = avg_last_bounce / num_entries if num_entries > 0 else 0
        df_target.append({"target": ENTRY["target"], "score": avg_score, "last_bounce": avg_last_bounce})
    df_target = pd.DataFrame(df_target)

    # Average the scores over each vary_param
    df_vary_param = {}
    freq_counts = {}
    for TARGET in json_full.keys():
        for ENTRY in json_full[TARGET]:
            ### IMPORTANT: SKIP IF FAILURE ###
            if ENTRY["score"] == ENTRY["target"]:
                continue
            df_vary_param[ENTRY[vary_param]] = df_vary_param.get(ENTRY[vary_param], 0) + ENTRY["score"]
            freq_counts[ENTRY[vary_param]] = freq_counts.get(ENTRY[vary_param], 0) + 1
    for k in df_vary_param.keys():
        df_vary_param[k] = df_vary_param[k] / freq_counts[k] if freq_counts[k] > 0 else 0
    df_vary_param = pd.DataFrame(df_vary_param.items(), columns=[vary_param, "score"])

    ### Plot the TARGET - SCORE graph

    # Calculate best fit lines for both dataframes
    z = np.polyfit(df_target['target'], df_target['score'], 1)
    p = np.poly1d(z)

    # Calculate standard deviations for the difference column
    mean_score = df_target['score'].mean()
    std_dev = df_target['score'].std()

    # Plot data and best fit lines with shaded regions
    x = np.linspace(df_target['target'].min(), df_target['target'].max(), 100)
    ax1[0].plot(x, p(x), color=colors[EXAMPLE], label=f"{EXAMPLE}-Shot", linewidth=int(3/(LATEX_FIG_SCALE)))
    ax1[0].fill_between(x, p(x) + std_score, p(x) - std_score, color=shade_colors[EXAMPLE], alpha=0.1)

    # Scatter plot
    ax1[0].scatter(df_target['target'], df_target['score'], color=colors[EXAMPLE], alpha=0.5, linewidth=5)

    # Labeling and showing the plot
    ax1[0].set_xlabel("Target X Position")
    ax1[0].set_xlim(0, 100)
    ax1[0].xaxis.set_ticks(np.arange(0, 110, 10))
    ax1[0].set_ylabel("Error (|bounce-target|)")
    #ax1[0].set_title(f"Target vs Difference\nModel: {MODEL}")
    ax1[0].set_ylim(0, 100)
    ax1[0].legend()

    ### Plot the FREQ - SCORE graph
    # Calculate best fit lines for both dataframes
    z = np.polyfit(df_vary_param[vary_param], df_vary_param['score'], 1)
    p = np.poly1d(z)

    # Calculate standard deviations for the difference column
    mean_score = df_vary_param['score'].mean()
    std_dev = df_vary_param['score'].std()

    # Plot data and best fit lines with shaded regions
    x = np.linspace(df_vary_param[vary_param].min(), df_vary_param[vary_param].max(), 100)
    ax1[1].plot(x, p(x), color=colors[EXAMPLE], label=f"Ex: {EXAMPLE}, Std Dev: {std_dev:.2f},")
    ax1[1].fill_between(x, p(x) + std_dev, p(x) - std_dev, color=shade_colors[EXAMPLE], alpha=0.25)

    # Scatter plot
    ax1[1].scatter(df_vary_param[vary_param], df_vary_param['score'], color=colors[EXAMPLE], alpha=0.5, linewidth=5)

    # Labeling and showing the plot
    ax1[1].set_xlabel(f"{vary_param} of Surface Curve")
    ax1[1].set_ylabel("Error (|bounce-target|)")
    #ax1[1].set_title(f"{vary_param} vs Difference\nModel: {MODEL}")
    ax1[1].set_ylim(0, 100)
    ax1[1].legend()

    ### Plot the SCORE - PROB graph
    mean_score = df_target["score"].mean()
    std_score = df_target["score"].std()
    min_val, max_val = 0, 50  
    GRANULARITY = 5
    BINS = int(max_val / GRANULARITY)
    ax2[0].hist(df_target["score"], bins=BINS, range=(min_val, max_val), color='lightblue', edgecolor='black', alpha=0.7, density=True)
    ax2[0].axvline(mean_score, color='red', linestyle='dashed', linewidth=1, label=f'Avg: {mean_score:.2f}')
    ax2[0].axvline(mean_score - std_score, color='green', linestyle='dashed', linewidth=1, label=f'Std Dev: {std_score:.2f}')
    ax2[0].axvline(mean_score + std_score, color='green', linestyle='dashed', linewidth=1)
    ax2[0].set_xlim(min_val, max_val)  # This limits the x-axis range
    Y_MAX = 1.0 / GRANULARITY
    ax2[0].set_ylim(0, Y_MAX)  # This limits the y-axis range
    ax2[0].yaxis.set_ticks(np.arange(0, Y_MAX + Y_MAX / GRANULARITY, Y_MAX / GRANULARITY))  
    #ax2[0].set_title(f"Distribution of Scores\nModel: {MODEL}, Examples: {EXAMPLE}")
    ax2[0].set_xlabel('Error (|bounce-target|)')
    ax2[0].set_ylabel('Probability Density')
    ax2[0].legend()

    ### Plot the TARGET - LAST BOUNCE graph
    ax2[1].scatter(df_target["target"], df_target["last_bounce"], alpha=0.6, edgecolors="w", linewidth=0.5)
    # Adding y=x line
    ax2[1].plot(df_target["target"], df_target["target"], color='gray', label='y=x')
    # Calculating and plotting best fit line
    z = np.polyfit(df_target["target"], df_target["last_bounce"], 1)
    p = np.poly1d(z)
    # Get accuracy
    predicted_y = p(df_target["target"])
    actual_y = df_target["target"]
    mae = np.mean(abs(predicted_y - actual_y))
    ax2[1].plot(df_target["target"], p(df_target["target"]), color='red', label=f'Best Fit, MAE: {mae:.4f}')
    #ax2[1].set_title(f'Scatter plot of Targets vs. Last Bounce')
    ax2[1].set_xlabel('Target')
    ax2[1].set_ylabel('Last Bounce')
    ax2[1].set_xlim(0, 100)
    ax2[1].set_ylim(-25, 125)
    ax2[1].xaxis.set_ticks(np.arange(0, 110, 10))
    ax2[1].yaxis.set_ticks(np.arange(YLIM_LOWER, YLIM_UPPER, 10))
    ax2[1].legend()

plot_dict = {
    "difficulty": plot_difficulty_curve,
    "heatmaps": plot_heatmaps,
    "difficulty_bar": plot_difficulty_bar,
    "main": plot_main_results,
    "main_ratio": plot_main_results_ratio,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", nargs='+', type=str, default=None, help="Model to visualize")
    parser.add_argument("--exp", nargs='+', type=int, default=[1], help="Experiments to visualize")
    parser.add_argument("--save", action="store_true", help="Save the plot")
    parser.add_argument("--plot", type=str, default=None, help="Plot to visualize")
    args = parser.parse_args()

    # Set the font family and size to match your LaTeX document
    rcParams['font.family'] = 'serif'
    rcParams['font.size'] = int(FONT_SIZE / LATEX_FIG_SCALE)

    if args.plot is not None and args.plot in plot_dict:
        plot_dict[args.plot]()
        exit()

    SAVE = args.save
    MODELS = args.model
    BITS = 8

    for EXP in args.exp:

        for MODEL in MODELS:

            print("---------------------------")
            
            MODEL = MODEL.split("/")[-1] 
            if not any(i in MODEL for i in ["bison", "gpt", "random"]):
                MODEL += f"/{BITS}_bit" if BITS is not None else ""

            print("Visualizing model:", MODEL)

            # Get specific folder to visualize
            results_folder = f"results/experiment_{EXP}/{MODEL}/"

            folders = glob.glob(results_folder + "*")
            if len(folders) == 0:
                print(f"No results found for model: {MODEL} in folder {results_folder}*")
                continue

            # Define a nice color palette
            colors = ["#3498DB", "#E74C3C", "#34195E", "#2ECC71"]
            shade_colors = ["#3498db66", "#e74c3c66", "#34195E66", "#2ECC7166"]  # Lighter versions for shading
                
            ### Get all jsons in the folders
            total_jsons = {}
            for f in folders:
                jsons = glob.glob(f"{f}/*.json")     
                jsons.sort()
                for j in jsons:
                    try:
                        EXAMPLE = int(j.split("/")[-1].split("_")[1].split(".")[0])
                    except:
                        continue
                    total_jsons[EXAMPLE] = total_jsons.get(EXAMPLE, []) + [j]

            ### PLOT 1: TARGET - SCORE ###
            if EXP in [1,2,7,8,9,12]:
                fig1, ax1 = plt.subplots(1, 1, figsize=( int(PLOT_SIZE /(LATEX_FIG_SCALE)), int(PLOT_SIZE /(LATEX_FIG_SCALE))))
            else:
                fig1, ax1 = plt.subplots(1, 2, figsize=( int(PLOT_SIZE /(LATEX_FIG_SCALE)), int(PLOT_SIZE /(LATEX_FIG_SCALE))))

            ### SCORE - PROB ### TARGET - LAST BOUNCE ###
            NUM_EXAMPLES = len(total_jsons.keys())
            assert NUM_EXAMPLES > 0, "No results found"
            fig2, ax2 = plt.subplots(NUM_EXAMPLES+1 if NUM_EXAMPLES==1 else NUM_EXAMPLES, 2, figsize=( int(PLOT_SIZE * 1/(LATEX_FIG_SCALE)), int(PLOT_SIZE * 1/(LATEX_FIG_SCALE))))

            ### Plot for each number of examples
            for EXAMPLE in range(0,NUM_EXAMPLES):
                # Get the jsons for this number of examples in a single dict
                jsons = total_jsons[EXAMPLE]
                if len(jsons) == 0:
                    continue

                # Load the JSON data into a DataFrame
                json_full = {}
                for j in jsons:
                    j = json.load(open(j, "r"))
                    for key, value in j.items():
                        json_full[key] = json_full.get(key, []) + list(value)

                print(f"Experiment {EXP}, Model {MODEL}, Examples {EXAMPLE}")

                if EXP == 1:
                    plot_exp_no_variation(json_full, ax1, ax2[EXAMPLE], EXAMPLE)
                elif EXP == 2:
                    plot_exp_no_variation(json_full, ax1, ax2[EXAMPLE], EXAMPLE)
                elif EXP == 3:
                    plot_exp_variation(json_full, ax1, ax2[EXAMPLE], EXAMPLE, "frequency")
                elif EXP == 4:
                    plot_exp_variation(json_full, ax1, ax2[EXAMPLE], EXAMPLE, "amplitude")
                elif EXP == 5:
                    plot_exp_variation(json_full, ax1, ax2[EXAMPLE], EXAMPLE, "coeff_r")
                elif EXP == 6:
                    plot_exp_variation(json_full, ax1, ax2[EXAMPLE], EXAMPLE, "max_attempts")
                elif EXP == 7:
                    plot_exp_no_variation(json_full, ax1, ax2[EXAMPLE], EXAMPLE)
                elif EXP == 8:
                    plot_exp_no_variation(json_full, ax1, ax2[EXAMPLE], EXAMPLE)
                elif EXP == 9:
                    plot_exp_no_variation(json_full, ax1, ax2[EXAMPLE], EXAMPLE)
                elif EXP == 10:
                    plot_exp_variation(json_full, ax1, ax2[EXAMPLE], EXAMPLE, "difficulty")
                elif EXP == 11:
                    plot_exp_variation(json_full, ax1, ax2[EXAMPLE], EXAMPLE, "difficulty")
                elif EXP == 12:
                    plot_exp_no_variation(json_full, ax1, ax2[EXAMPLE], EXAMPLE)
                else:
                    raise Exception("Experiment not implemented")
                    

            # Adjust the layout and show
            fig1.tight_layout(pad=.5)
            fig2.tight_layout(pad=.5)

            if SAVE:
                # Save as pdf
                fig1.savefig(f"figures/exp{EXP}_{MODEL.split('/')[0]}.pdf")
                #fig2.savefig(f"figures/exp{EXP}_{MODEL}_extra.pdf")
                plt.close()
            else:
                plt.show()
