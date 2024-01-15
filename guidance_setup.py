import guidance, json, os, torch, gc, time
import path

def flush_guidance():
    del guidance.llm
    torch.cuda.empty_cache()
    gc.collect()

def empty_cache():
    torch.cuda.empty_cache()
    gc.collect()

def reload_model():
    empty_cache()
    try:
        del guidance.llm.model_obj
    except:
        pass
    load_model(path.CURRENT_MODEL, path.CURRENT_MODEL_SETTINGS)

def config_changes(MODEL_DIR):
    ### Change config.json to increase max_position_embeddings to 4096
    config_path = f"{MODEL_DIR}/config.json"
    config = json.load(open(config_path, "r"))
    config["max_position_embeddings"] = 4096
    json.dump(config, open(config_path, "w"), indent=4)

    ### Change tokenzier_config.json to increase model_max_length to 4096
    tokenizer_config_path = f"{MODEL_DIR}/tokenizer_config.json"
    tokenizer_config = json.load(open(tokenizer_config_path, "r"))
    tokenizer_config["model_max_length"] = 4096
    json.dump(tokenizer_config, open(tokenizer_config_path, "w"), indent=4)

def load_model(MODEL, MODEL_SETTINGS):

    ### Model Settings
    TIME = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime(time.time()))
    EXP = MODEL_SETTINGS["experiment"]
    BITS = MODEL_SETTINGS["bits"]
    MODEL_REPO = MODEL_SETTINGS["model_repo"]
    PALM_KEY = MODEL_SETTINGS["palm_key"]

    path.CURRENT_MODEL = MODEL
    path.CURRENT_MODEL_SETTINGS = MODEL_SETTINGS.copy()

    ### Correctly set up the model
    if "gpt" in MODEL:
        import openai
        try:
            openai.api_key = os.getenv("OPENAI_API_KEY")
        except:
            print("ERROR: OPENAI_API_KEY environment variable not set.")
            exit()
        guidance.llm = guidance.llms.OpenAI(MODEL) 
        RESULTS_DIR = f"{path.ROOT_FOLDER}/results/experiment_{EXP}/{MODEL}/{TIME}"
        MODEL_DIR = None
    elif any([i in MODEL for i in ["bison","gemini"]]):
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = PALM_KEY
        guidance.llm = guidance.llms.PaLM(MODEL) 
        RESULTS_DIR = f"{path.ROOT_FOLDER}/results/experiment_{EXP}/{MODEL}/{TIME}"
        MODEL_DIR = None
    else:
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
        if MODEL == "random":
            RESULTS_DIR = f"{path.ROOT_FOLDER}/results/experiment_{EXP}/{MODEL}/{TIME}"
        else:
            try:
                MODEL_DIR = f"{path.MODELS_FOLDER}/{MODEL}"   
                RESULTS_DIR = f"{path.ROOT_FOLDER}/results/experiment_{EXP}/{MODEL}/{BITS}_bit/{TIME}"    
                               
                try:
                    guidance.llm = guidance.llms.Transformers(MODEL_REPO, load_in_8bit=BITS==8, load_in_4bit=BITS==4, device_map="auto")
                except Exception as e:
                    print("ERROR: ", e)
                    from huggingface_hub import snapshot_download
                    os.makedirs(MODEL_DIR) if not os.path.exists(MODEL_DIR) else None
                    snapshot_download(repo_id=MODEL_REPO, local_dir=MODEL_DIR)
                    guidance.llm = guidance.llms.Transformers(MODEL_DIR, load_in_8bit=BITS==8, load_in_4bit=BITS==4, device_map="auto")

                config_changes(MODEL_DIR)

                print(f"Loaded LLM: {MODEL}!")
            except Exception as e:  
                print(f"ERROR: {MODEL} failed to load, {e}.")
                return None, None
            
    return RESULTS_DIR, MODEL_DIR