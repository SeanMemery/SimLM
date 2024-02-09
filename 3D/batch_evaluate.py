from evaluate_model import *

DEFAULT_CONFIG = {
    "llm": {
        "ctx_n": 4096,
        "temperature": 0.7,
        "top_k": 40,
        "top_p": 0.9,
        "num_predict": 256,
    },

    "method_iterations": {
        "random":   0,
        "simlm":    0,
        "baseline": 200,
        "discrete": 0,
    },

    "profile": False,
    "few_shot": 2,
    "rel_boards": 1,
    "evaluation_set": "latest",
    "visualizable": False,
    "max_shots": 3,

}

MODELS = [
    #"gpt-4-turbo-preview",
    #"gpt-3.5-turbo",
    #"chat-bison",
    #"llama2:7b-chat",
    "llama2:13b-chat",
]

FEW_SHOTS = [1, 2]

CONFIGS = [
    DEFAULT_CONFIG,
]
# CONFIGS += [
#     {
#         **DEFAULT_CONFIG,
#         "few_shot": fs,
#     } for fs in FEW_SHOTS
# ]

if __name__ == "__main__":

    for model_name in MODELS:

        for config in CONFIGS:

            evaluate_setup(model_name, config)
