import argparse, os, yaml, guidance, json, torch, gc, random
import numpy as np
from tqdm import tqdm
import guidance_setup
from experiments import Experiment

### Dictionary to map experiment number to varying parameter
VARYING_PARAMS_DICT = {
    3: "FREQUENCIES",
    4: "AMPLITUDES",
    5: "COEFF_R",
    6: "MAX_ATTEMPTS",
    10: "DIFFICULTIES",
    11: "DIFFICULTIES",
}

### For difficulty experiment
FIXED_TARGET = 50

def SETUP_GUIDANCE(MODEL, MODEL_SETTINGS):
    print(f"Loading LLM: {MODEL}, Bits: {BITS}" if BITS else f"Loading LLM: {MODEL}")
    guidance.llms.Transformers.cache.clear()

    RESULTS_DIR, MODEL_DIR = guidance_setup.load_model(MODEL, MODEL_SETTINGS)
            
    ### If results directory doesn't exist, create it
    if RESULTS_DIR and not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)        
    
    return RESULTS_DIR, MODEL_DIR

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Collect experiment results.')
    parser.add_argument("--exp", nargs='+', type=int, default=[1], help="Experiment to collect results for.")
    parser.add_argument("--rm", action="store_true", help="Remove model after experiment.")
    parser.add_argument("--models", type=str, default="models", help="Partition of model_list.yaml to get models list.")
    parser.add_argument("--examples", nargs='+', type=int, default=[0, 1, 2], help="Number of examples to use for reasoning.")
    args = parser.parse_args()
    config = yaml.load(open(f"config.yaml", "r"), Loader=yaml.FullLoader)

    REMOVE = args.rm
    EXPERIMENTS = args.exp
    BITS = config["bits"]
    EXAMPLES = args.examples
    EXAMPLE_GEN = config["example_gen"]
    TARGET = config["target"]
    ITERATIONS = config["iterations"]

    ### Varying Paramaters
    FREQUENCIES = [float(i) for i in list(np.arange(config["frequencies"][0], config["frequencies"][1], config["frequencies"][2]))]
    AMPLITUDES  = [float(i) for i in list(np.arange(config["amplitudes"][0], config["amplitudes"][1], config["amplitudes"][2]))]
    COEFF_R     = [float(i) for i in list(np.arange(config["coeff_r"][0], config["coeff_r"][1], config["coeff_r"][2]))]
    MAX_ATTEMPTS = [int(i) for i in  list(np.arange(config["max_attempts"][0], config["max_attempts"][1], config["max_attempts"][2]))]
    DIFFICULTIES = [int(i) for i in list(np.arange(config["difficulties"][0], config["difficulties"][1], config["difficulties"][2]))]

    TARGET_BOUNCE = config["target_bounce"]
    MODELS = yaml.load(open(config["model_list"], "r"), Loader=yaml.FullLoader)[args.models]
    PALM_KEY = config["palm_key"]

    for EXP in EXPERIMENTS:

        print("EXPERIMENT:", EXP)

        ### Run experiment for each model
        for MODEL_REPO in MODELS:
            MODEL = MODEL_REPO.split("/")[-1]
            
            ### Setup guidance library
            MODEL_SETTINGS = {
                "experiment": EXP,
                "bits": BITS,
                "palm_key": PALM_KEY,
                "model_repo": MODEL_REPO
            }
            RESULTS_DIR, MODEL_DIR = SETUP_GUIDANCE(MODEL, MODEL_SETTINGS)

            ### If setup failed, skip
            if not RESULTS_DIR:
                print(f"ERROR: {MODEL} failed to load, skipping.")
                continue
            
            ### Save a copy of the config to results directory
            with open(f"{RESULTS_DIR}/config.yaml", "w") as f:
                f.write(yaml.dump(config.copy(), default_flow_style=False))

            ### Run experiment on MODEL for each number of examples
            for NUM_EXAMPLES in EXAMPLES:
                JSON_OUTPUTS = {}

                ### If analogical
                ANALOGICAL = NUM_EXAMPLES=="analogical"

                ### Setup experiment
                VARYING_PARAMS = eval(VARYING_PARAMS_DICT[EXP]) if EXP in VARYING_PARAMS_DICT.keys() else None
                EXPERIMENT = Experiment(EXP, MODEL, TARGET_BOUNCE, ANALOGICAL, VARYING_PARAMS, EXAMPLE_GEN)

                ### Run experiment for each target position between N_MIN and N_MAX
                for _ in tqdm(range(ITERATIONS), desc="Running LLM Simulations"):

                    ### Perform LLM reasoning
                    res_list = EXPERIMENT.reasoning(TARGET, NUM_EXAMPLES)  
                    for i, res in enumerate(res_list):
                        res["score"]           = abs(res["bounces"][TARGET_BOUNCE]-TARGET)
                        res["target"]          = TARGET
                        res["target_bounce"]   = TARGET_BOUNCE
                        res["model"]           = MODEL
                        res["examples"]        = NUM_EXAMPLES
                        res["bits"]            = BITS 
                        res["frequency"]       = VARYING_PARAMS[i] if EXP == 3 else 1
                        res["amplitude"]       = VARYING_PARAMS[i] if EXP == 4 else 1
                        res["coeff_r"]         = VARYING_PARAMS[i] if EXP == 5 else 0.9
                        res["max_attempts"]    = VARYING_PARAMS[i] if EXP == 6 else 5
                        res["difficulty"]      = VARYING_PARAMS[i] if EXP == 10 or EXP == 11 else 1
                        res["experiment"]      = EXP
                        res["examples_gen"]    = EXAMPLE_GEN

                    ### Record Results
                    JSON_OUTPUTS[str(TARGET)] = JSON_OUTPUTS.get(str(TARGET),[]) + res_list

                    # Write JSON entry
                    with open(f"{RESULTS_DIR}/examples_{NUM_EXAMPLES}.json", "w") as f:
                        f.write(json.dumps(JSON_OUTPUTS, indent=4))
                    
                    ### Clear GPU memory if possible
                    torch.cuda.empty_cache()
                    gc.collect()

            # Remove model to save space
            if MODEL_DIR and REMOVE:
                os.system(f"rm -rf {MODEL_DIR}")

            # Free Memory
            guidance_setup.flush_guidance()
