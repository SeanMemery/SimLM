# Script to collect all the tasks for the evaluation set and output as json object to "./evaluation_sets/{current_date}.json"

import json
import os

if __name__ == "__main__":
    json_data = {}

    # Set conditions for evaluation set
    conditions = {
        "rack": ["oneball"],
        "complexity": [1],
    }

    # Add details to json
    json_data["details"] = {}
    json_data["details"]["conditions"] = conditions
    json_data["details"]["name"] = input("Evaluation set name: ")
    json_data["details"]["description"] = input("Evaluation set description: ")

    # Set current prompts 
    json_data["prompts"] = {}

    template_dir = "templates"
    for filename in os.listdir(template_dir):
        if filename.endswith(".j"):
            prompt_name = os.path.splitext(filename)[0]
            with open(os.path.join(template_dir, filename), "r") as f:
                json_data["prompts"][prompt_name] = f.read()

    # Get task json 
    task_pool = json.load(open("task_pool.json"))

    # Create evaluation set
    amounts = {
        "total": 0,
    }
    json_data["tasks"] = []
    for task_description in task_pool:

        # Check conditions
        skip = False
        for condition, values in conditions.items():
            if task_description[condition] not in values:
                skip = True

        # Skip if conditions not met
        if skip:
            continue

        json_data["tasks"].append({
            "prompt":     task_description["prompt"],
            "rack":       task_description["rack"],
            "pass":       task_description["pass"],
            "fail":       task_description["fail"],
            "conditions": task_description["conditions"],
            "task_id":    amounts["total"],
        })
        amounts["total"] += 1
        amounts[task_description["rack"]] = amounts.get(task_description["rack"], 0) + 1

    # Print details
    print("------------Task Amounts------------")
    for rack, amount in amounts.items():
        print(f"{rack}: {amount}")
    print("------------------------------------")

    # Add count to details
    json_data["details"]["amounts"] = amounts

    # Write to file
    with open(f"evaluation_sets/{json_data['details']['name']}.json", "w") as f:
        json.dump(json_data, f, indent=4)

    



