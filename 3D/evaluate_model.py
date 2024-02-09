import yaml, json, os, datetime, sys
from tqdm import tqdm
from path import *

sys.path.append("src/")
import solvers
from task import Task
from profiler import *

class Evaluation():
    
    def __init__(self, model_name, method, iterations, config) -> None:

        self.config = config

        # Visualise
        solvers.VISUALISE = config["visualizable"]

        # Method
        self.method = method

        # Reset profiler 
        reset_profiles()

        # Load evaluation set
        self.evaluation_set = self.load_evaluation_set(self.config["evaluation_set"])

        # Create solver
        with Profile("create_solver"):
            if self.method == "random":
                self.solver = solvers.RandomPredictor()
            elif self.method == "baseline":
                self.solver = solvers.Baseline(model_name, self.config, self.evaluation_set["prompts"])
            elif self.method == "simlm":
                self.solver = solvers.SimLM(model_name, self.config, self.evaluation_set["prompts"])
            elif self.method == "discrete":
                self.solver = solvers.DiscreteSimLM(model_name, self.config, self.evaluation_set["prompts"])
            else:
                raise ValueError(f"Method {config['method']} not implemented.")
        
        # Params
        self.model_name = model_name
        current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        results_dir = f"{RESULTS_DIR}/{self.evaluation_set['details']['name']}"
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        self.results_file = f"{results_dir}/{current_time}.json"
        self.N = self.config["max_shots"]
        self.iterations = iterations

    def load_evaluation_set(self, set_name):
        """
        Load the evaluation set
        """

        if set_name == "latest":
            files = os.listdir(EVALUATION_SET_DIR)
            sorted_files = sorted(files, key=lambda x: os.path.getctime(f"{EVALUATION_SET_DIR}/{x}"), reverse=True)
            evaluation_set = f"{EVALUATION_SET_DIR}/{sorted_files[0]}"
        else:
            evaluation_set = f"{EVALUATION_SET_DIR}/{set_name}.json"

        if not os.path.exists(evaluation_set):
            raise ValueError(f"Evaluation set {evaluation_set} does not exist.")

        with open(evaluation_set, "r") as f:
            evaluation_set = json.load(f)

        return evaluation_set
    
    def save_trial(self, trial_data):
        """
        Save trial data
        """
        if not os.path.exists(self.results_file):
            total_results = []
        else:
            with open(self.results_file, "r") as f:
                total_results = json.load(f)
        total_results.append(trial_data)
        with open(self.results_file, "w") as f:
            json.dump(total_results, f, indent=4)

    def run(self):

        num_tasks = len(self.evaluation_set["tasks"])
        for iter in tqdm(range(self.iterations), desc=f"Running {self.method} for model {self.model_name}..."):
            for ind, task_description in enumerate(self.evaluation_set["tasks"]):

                trial_data = self.run_trial(task_description)

                if self.config["profile"]:
                    make_report()

                if len(trial_data["shots"]) == 0:
                    print(f"Trial {iter*num_tasks + ind}/{self.iterations*num_tasks} failed, no shots taken.")
                    continue
                
                self.save_trial(trial_data)

    def run_trial(self, task_description):
        """
        Run a single trial
        """

        with Profile("single_trial"):

        ### 0. Setup
            trial_data = {
                **self.config["llm"],
                "model": self.model_name,
                "method": self.method,
                "task_id": task_description["task_id"],
                "evaluation_set": self.evaluation_set["details"]["name"],
                "task": task_description,
                "few_shot": self.config["few_shot"],
                "rel_boards": self.config["rel_boards"],
                "passed": False,
                "shots": [],
                "events": [],
                "context": "",
                "board_states": [],
                "outcome": ""
            }

            task = Task(
                task_description["prompt"],
                task_description["rack"],
                task_description["pass"],
                task_description["fail"],
                task_description["conditions"]
            )

            trial =  self.solver.solve(task, trial_data, self.N)


            return trial

if __name__ == "__main__":
    model_name = os.environ.get("MODEL_NAME")

    if model_name is None:
        raise ValueError("MODEL_NAME environment variable not set.")

    with open(CONFIG_FILE, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    methods = config["method_iterations"]

    for method, iterations in methods.items():
        if not iterations > 0:
            continue
        evaluation = Evaluation(model_name, method, iterations, config)
        evaluation.run()


# Evaluate a single setup of model and config
def evaluate_setup(model_name, config):
    for method, iterations in config["method_iterations"].items():
        if iterations > 0:
            evaluation = Evaluation(model_name, method, iterations, config)
            evaluation.run()