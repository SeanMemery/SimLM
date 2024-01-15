import guidance, os, json
import guidance_setup
from path import ROOT_FOLDER
from tqdm import tqdm

### Setup prompt
summarise_reasoning_header = """\
Below is a transcript of an experiment where a user was asked to predict the initial conditions of a ball, such that the third bounce of the ball lands near a target. \
You are tasked with summarising this experiment such that it maintains its reasoning but is more concise. \
The transcript is as follows:
\n\n####### TRANSCRIPT START #######
"""
TO_REPLACE = [("..", "."), ("\"",""), ("\n",""), ("  ", " "), (".,.", "."), ("..", "."), (".,,.", "."), (". .", ".")]

def summarise_single(summarise_reasoning_header, EXAMPLE):
    TARGET = EXAMPLE["target"]
    REASONING = EXAMPLE["example"]
    
    # Extract all valid jsons from the reasoning, i.e. the text bounded by { and }
    valid_jsons = []
    for i in range(len(REASONING)):
        if REASONING[i] == "{":
            for j in range(i+1, len(REASONING)):
                if REASONING[j] == "}":
                    json_extract = REASONING[i:j+1]
                    # Skip if example json
                    if "\"height\": h" in json_extract or "\"horizontal_velocity\": v" in json_extract:
                        continue
                    json_extract = json_extract.replace("\n", "")
                    valid_jsons.append(json_extract)
                    break

    # Extract all bounces
    bounces = []
    bounce_start = "bounces occurred at:\n"
    bounce_end = "\nGive a critique"
    bounce_start_i = REASONING.find(bounce_start)
    while bounce_start_i != -1:
        bounce_end_i = REASONING.find(bounce_end, bounce_start_i)
        bounces.append(REASONING[bounce_start_i:bounce_end_i].split("\n")[1])
        bounce_start_i = REASONING.find(bounce_start, bounce_end_i)

    # If there are not equal number of valid jsons and bounces, then the example is invalid
    if len(valid_jsons)-1 != len(bounces):
        print("Invalid example as there are not equal number of valid jsons and bounce")
        print(f"valid_jsons: {valid_jsons}")
        print(f"bounces: {bounces}")
        return ""

    # Summarise the example
    PROMPT = summarise_reasoning_header + REASONING + "\n####### TRANSCRIPT END #######\n\nSummarise the above experiment transcript section by section to fit the template below. This means for example to summarise the second CRITIQUE section to CRITIQUE_2 etc. Be sure to match the REASONING section with the REASONING entry below, and each CRITIQUE seciton with the CRITIQUE entries below. Be sure to write the summaries as one CONCISE paragraph with no new lines.\n\n"
    PROMPT += f"""
        Reasoning: For a target of {TARGET}""" + """, {{gen 'REASONING' max_tokens=128 stop="\n" temperature=0.7 token_healing=False}}"""
    PROMPT += f"""
        Prediction + Simulation 1: Therefore, the new prediction is {valid_jsons[0]}. SIMULATION: {bounces[0]}."""
    PROMPT += """
        Critique 1: Based on this, {{gen 'CRITIQUE_1' max_tokens=64 stop="\n" temperature=0.7 token_healing=False}}"""
    for i in range(len(valid_jsons)-2):
        PROMPT += f"""
            Prediction + Simulation {i+2}: Therefore, the new prediction is {valid_jsons[i+1]}. SIMULATION: {bounces[i+1]}."""
        PROMPT += f"""
        Critique {i+2}: Based on this, {{{{gen 'CRITIQUE_{i+2}' max_tokens=64 stop="\n" temperature=0.7 token_healing=False}}}}"""
    PROMPT += "}"
    PROMPT = guidance(PROMPT, caching=False)().variables()

    initial_reasoning = PROMPT["REASONING"]
    critiques = [PROMPT[f"CRITIQUE_{i+1}"] for i in range(len(valid_jsons)-1)]

    # Initialise prompt
    SUMMARISED_EXAMPLE = f"For a target of {TARGET}, {initial_reasoning}. Therefore, the prediction is {valid_jsons[0]}. SIMULATION: {bounces[0]}."

    # Use list of jsons and bounces to setup template
    for critique, prediction, simulation in zip(critiques[:-1], valid_jsons[1:-1], bounces[1:]):
        SUMMARISED_EXAMPLE += f" Based on this, {critique}. Therefore, the new prediction is {prediction}. SIMULATION: {simulation}."

    # Add final critique
    SUMMARISED_EXAMPLE += f" Based on this, {critiques[-1]}."

    # Add final prediction
    SUMMARISED_EXAMPLE += f" Therefore, the final prediction is {valid_jsons[-1]}."

    # Remove TO_REMOVE
    for (r1,r2) in TO_REPLACE:
        SUMMARISED_EXAMPLE = SUMMARISED_EXAMPLE.replace(r1, r2)

    return SUMMARISED_EXAMPLE

def load_guidance():
    ### Model Settings
    MODEL_SETTINGS = {
        "experiment": -1,
        "bits": 8,
        "skip": False,
        "palm_key": "/home/sean/Desktop/palm2api-401803-8fc20d576f82.json",
        "model_repo": "",
    }
    
    ### Load model
    guidance_setup.load_model("text-bison", MODEL_SETTINGS)

if __name__ == "__main__":

    ### Load the examples
    raw_examples = json.load(open(f"{ROOT_FOLDER}/examples/example_reasoning.json", "r"))

    ### Add baseline experiments as exceptions
    EXCLUDED_EXPERIMENTS = [
        "experiment_8",
        "experiment_9",
        "experiment_11",
    ]
    INCLUDED_MODELS = [
        "gpt-3.5-turbo",
        "chat-bison",
        "gpt-4",
        "Llama-2-13B-Chat-fp16",
        "Llama-2-7b-chat-hf",
        "Llama-2-13b-chat-hf",
        "Llama-2-70b-chat-hf",
    ]

    ### Load guidance
    load_guidance()

    ### Initialise dictionary
    SUMMARISED_EXAMPLES_DICT = json.load(open(f"{ROOT_FOLDER}/examples/summarised_example_reasoning.json"))
    
    ### Loop through the models
    for MODEL in tqdm(raw_examples.keys()):
        print(f"MODEL: {MODEL}")
        SUMMARISED_EXAMPLES_DICT[MODEL] = SUMMARISED_EXAMPLES_DICT.get(MODEL, {})

        ### Loop through the experiments
        for EXPERIMENT in tqdm(raw_examples[MODEL].keys()):
            print(f"EXPERIMENT: {EXPERIMENT}")

            ### Skip if not the experiment we want or if the model is not included
            if EXPERIMENT in EXCLUDED_EXPERIMENTS or MODEL not in INCLUDED_MODELS:
                SUMMARISED_EXAMPLES_DICT[MODEL][EXPERIMENT] = raw_examples[MODEL][EXPERIMENT]
                continue
            else:
                SUMMARISED_EXAMPLES_DICT[MODEL][EXPERIMENT] = SUMMARISED_EXAMPLES_DICT[MODEL].get(EXPERIMENT, [])

            ### Loop through the examples
            for EXAMPLE in tqdm(raw_examples[MODEL][EXPERIMENT], desc="Iterating over Examples"):
            
                SKIP = False
                ### Check if example has already been summarised
                for EXAMPLE_SUMMARISED in SUMMARISED_EXAMPLES_DICT[MODEL][EXPERIMENT]:
                    if EXAMPLE_SUMMARISED["score"] == EXAMPLE["score"] and EXAMPLE_SUMMARISED["target"] == EXAMPLE["target"]:
                        SKIP = True

                ### If example has already been summarised, skip
                if SKIP:
                    continue

                SUMMARISED_EXAMPLE = summarise_single(summarise_reasoning_header, EXAMPLE)

                SCORE = EXAMPLE["score"]
                TARGET = EXAMPLE["target"]
                REASONING = EXAMPLE["example"]

                # Add to dictionary
                SUMMARISED_EXAMPLES_DICT[MODEL][EXPERIMENT].append({
                    "score": SCORE,
                    "experiment": int(EXPERIMENT.split("_")[-1]),
                    "model": MODEL,
                    "target": TARGET,
                    "raw_example": REASONING,
                    "example": SUMMARISED_EXAMPLE,
                })

            # Save to file
            with open(f"{ROOT_FOLDER}/examples/summarised_example_reasoning.json", "w") as f:
                json.dump(SUMMARISED_EXAMPLES_DICT, f, indent=4)
                    
                               

                
