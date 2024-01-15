import json, os
import argparse
from path import ROOT_FOLDER

# Script to load jsons with reasoning examples and collect them into a single file
# Usage: python collect_examples.py --limit 3 --attempts 0

def parse_jsons(jsons, limit, attempts):
    results = []
    for j in jsons:

        # Entry or key as older results have a single entry for a target rather than a list
        for entry_or_key in j:

            if isinstance(entry_or_key, str):
                entries = j[entry_or_key]
            else:
                entries = [entry_or_key]

            for entry in entries:

                # Skip if keys arent present
                keys = ["score", "target", "target_bounce", "response", "examples"]
                if not all([k in entry for k in keys]):
                    continue

                if entry["score"] != entry["target"] and entry["score"] < limit:

                    full_reasoning = entry['response']

                    assert isinstance(full_reasoning, str), f"Failed to find reasoning in {entry}"
                    if not "### REAL CASE:\n" in full_reasoning:
                        continue

                    TARGET = entry["target"]
                    TARGET_BOUNCE = entry["target_bounce"]
                    SCORE = entry["score"]
                    EXPERIMENT = entry["experiment"]
                    MODEL = entry["model"]

                    full_reasoning = full_reasoning.split("### REAL CASE:\n")[1]
                    full_reasoning = full_reasoning.replace("<|im_start|>user\n", "")
                    full_reasoning = full_reasoning.replace("<|im_start|>assistant\n", "")
                    full_reasoning = full_reasoning.replace("<|im_end|>", "")
                    full_reasoning = full_reasoning.replace(" and output below as a JSON object in this format:\n{\n\"height\": h,\n\"horizontal_velocity\": v\n}\nOutput nothing else, just the JSON object", "")
                    full_reasoning = full_reasoning.replace("  ", "")
                    full_reasoning = full_reasoning.replace("\n\n", "\n")
                    full_reasoning = full_reasoning.replace("```json", "")
                    full_reasoning = full_reasoning.replace("```", "")

                    results.append({
                        "target": TARGET,
                        "target_bounce": TARGET_BOUNCE,
                        "score": SCORE,
                        "experiment": EXPERIMENT,
                        "example": full_reasoning,
                        "model": MODEL
                    })

    return results

def recursive_file_search(folder, jsons):
    for f in os.listdir(folder):
        if os.path.isdir(os.path.join(folder, f)):
            recursive_file_search(os.path.join(folder, f), jsons)
        elif os.path.isfile(os.path.join(folder, f)) and f.endswith(".json") and f.startswith("examples"):
            # Ignore analogical, no longer considering this
            if "analogical" in f:
                continue
            with open(os.path.join(folder, f)) as f:
                try:
                    jsons.append(json.load(f))
                except:
                    print(f"Failed to load {f}")
                    pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, default=3)
    parser.add_argument('--attempts', type=int, default=0)
    args = parser.parse_args()

    jsons = []
    folder = f"{ROOT_FOLDER}/results/"
    recursive_file_search(folder, jsons)

    jsons = parse_jsons(jsons, args.limit, args.attempts)
    EXPERIMENTS = list(set([j["experiment"] for j in jsons]))
    MODELS = list(set([j["model"] for j in jsons]))

    jsons_final = {MODEL: {} for MODEL in MODELS}
    for MODEL in MODELS:
        for EXP in EXPERIMENTS:
            print(f"{MODEL}/{EXP}: {len([j for j in jsons if j['experiment'] == EXP  and j['model'] == MODEL])}")
            jsons_final[MODEL][f"experiment_{EXP}"] = [j for j in jsons if j["experiment"] == EXP and j["model"] == MODEL]

    # Save to file
    with open(f"{ROOT_FOLDER}/examples/example_reasoning.json", 'w') as f:
        json.dump(jsons_final, f, indent=4)