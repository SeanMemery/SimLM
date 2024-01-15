import os, json, glob


if __name__ == '__main__':
    results_dir = os.path.join(os.getcwd(), 'results')

    # find all json files in results directory and subdirectories
    json_files = glob.glob(os.path.join(results_dir, '**/*.json'), recursive=True)

    total = 0

    # load json files
    for json_file in json_files:
        with open(json_file, 'r') as f:
            data = json.load(f)

        for TARGET in data.keys():
            for ENTRY in data[TARGET]:
                if ENTRY['score'] == ENTRY['target']:
                    data[TARGET].remove(ENTRY)
                total += 1

                if all(ex in ENTRY.keys() for ex in ["example_target_1","example_target_2"]):
                    continue
                else:
                    text = "a target of "
                    if ENTRY["examples"] == 1:
                        to_search = ENTRY["response"]
                        target = to_search[to_search.find(text)+len(text):]
                        target = target[:target.find(" ")].split("m")[0].split(",")[0]
                        try:
                            ENTRY["example_target_1"] = float(target)
                            ENTRY["example_target_2"] = 0  
                        except:
                            data[TARGET].remove(ENTRY)
                    elif ENTRY["examples"] == 2:
                        to_search = ENTRY["response"]
                        loc_1 = to_search.find(text)+len(text)
                        target_1 = to_search[loc_1:]
                        target_1 = target_1[:target_1.find(" ")].split("m")[0].split(",")[0]
                        

                        to_search = ENTRY["response"][loc_1:]
                        loc_2 = to_search.find(text)+len(text)
                        target_2 = to_search[loc_2:]
                        target_2 = target_2[:target_2.find(" ")].split("m")[0].split(",")[0]
                        try:
                            ENTRY["example_target_1"] = float(target_1)
                            ENTRY["example_target_2"] = float(target_2)
                        except:
                            print(ENTRY)
                            data[TARGET].remove(ENTRY)
                    else:
                        ENTRY["example_target_1"] = 0
                        ENTRY["example_target_2"] = 0


        # write cleaned json file
        with open(json_file, 'w') as f:
            json.dump(data, f, indent=4)