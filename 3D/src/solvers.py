import random

from pool import *
from llm import *
from jinja import *
from task import *
from loader import *
from profiler import *

VISUALISE = False

class Solver():
    def take_shot(self, prediction):
        with Profile("take_shot"):
            try:
                self.pool.strike(**prediction)
            except Exception as e:
                print("Failed to take shot")
                return [], {"text": "", "balls": {}}
            events = self.pool.get_events_desc()
            board_state = self.pool.get_board_state()
            if VISUALISE:
                self.pool.visualise()
            return events, board_state
    
    def show_results(self, trial_data, verify_state):
        raise NotImplementedError

    def solve(task, trial_data, max_attempts):
        raise NotImplementedError

# Random predictor: simply predicts a random shot and takes it 
class RandomPredictor(Solver):
    def __init__(self) -> None:
        self.pool = Pool(visualizable=VISUALISE)
        
    def setup(self, task, examples=[]):
        self.pool.new_state(task.rack)

    def random_prediction(self):
        return {
            "V0": random.uniform(0.0, 2.5),
            "phi": random.uniform(0.0, 360.0),
            "theta": random.uniform(0.0, 90.0),
            "a": random.uniform(-1.0, 1.0),
            "b": random.uniform(-1.0, 1.0),
        }

    def show_results(self, trial_data, verify_state):
        trial_data["passed"] = verify_state == VerifyState.PASS
        trial_data["outcome"] = verify_state.get_name()
        return trial_data

    def solve(self, task, trial_data, max_attempts):
        self.setup(task)
        trial_data["board_states"].append(self.pool.get_board_state())

    ### 1. Random prediction
        prediction = self.random_prediction()
        trial_data["shots"].append(prediction)

        # Take shot
        events, board_state = self.take_shot(prediction)
        trial_data["events"].append(events)
        trial_data["board_states"].append(board_state)

    ### 2. Success check
        verify_state = task.verify(events, board_state)

        if verify_state != VerifyState.INCOMPLETE:
            return self.show_results(trial_data, verify_state)
        
    ### 3. Reset board to initial state 
        self.pool.reset()
        
    ### 4. Critique and predict loop
        for _ in range(max_attempts):

            # Critique and predict
            prediction = self.random_prediction()
            trial_data["shots"].append(prediction)

            # Take shot
            events, board_state = self.take_shot(prediction)
            trial_data["events"].append(events)
            trial_data["board_states"].append(board_state)

            # Success check
            verify_state = task.verify(events, board_state)

            if verify_state != VerifyState.INCOMPLETE:
                return self.show_results(trial_data, verify_state)
            
            # Reset board state
            self.pool.reset()
            
    ### 5. Return if no success
        return self.show_results(trial_data, verify_state)

# Simulation Language Model: uses a language model to generate reasoning and critique from simulation feedback and make a prediction
class SimLM(Solver):
    def __init__(self, model_name, config, prompts) -> None:
        self.jinja = Jinja(prompts)
        self.pool = None
        with Profile("llm_init"):
            self.llm = LLM(model_name, config["llm"])
        with Profile("loader_init"):
            self.loader = Loader()
        self.examples = []

        # Init RAG
        self.examples_K = config["few_shot"]
        self.boards_K = config["rel_boards"]
            
    def setup(self, task):

        with Profile("solver_setup"):

            # Create board and task
            with Profile("create_board"):
                self.llm.reset()
                if self.pool is not None:
                    self.pool.new_state(task.rack)
                else:
                    self.pool = Pool(game_type=task.rack, visualizable=VISUALISE)

            if self.examples_K > 0:
                # Get K examples from the datastore
                with Profile("retrieve_examples"):
                    self.examples = self.loader.retrieve_examples(
                        self.__class__.__name__,
                        task.prompt,
                        self.examples_K
                    )
                    if len(self.examples) == 0:
                        print(f"No examples found for task {task.prompt}")
                        self.examples = []

            # Set system prompt
            system_prompt = self.jinja.get_system_text(self.pool.get_board_state(), self.examples, task.prompt)
            self.llm.set_system_prompt(system_prompt)


    def reason_predict(self):
        # Reasoning
        with Profile("reason"):
            reasoning = self.jinja.get_reasoning_text()
            self.llm.generate(reasoning)

        # Relevant board states
        relevant = []
        if self.boards_K > 0:
            with Profile("retrieve_board_states"):
                relevant = self.loader.retrieve_board_states(
                    self.pool.game_type,
                    self.pool.get_board_state()["balls"],
                    self.boards_K
                )

        # Prediction
        with Profile("predict"):
            prediction = self.jinja.get_prediction_text(relevant)
            prediction = self.llm.generate(prediction)
            prediction = self.llm.parse_prediction(prediction)

        return prediction
    
    def critique_predict(self, events, board_state):
        # Get the events and make text
        simulation = self.jinja.get_simulation_text(events, board_state)

        # Get critique text
        with Profile("critique"):
            critique = self.jinja.get_critique_text()
            sim_critique = simulation + "\n" + critique
            self.llm.generate(sim_critique)

        # Relevant board states
        relevant = []
        if self.boards_K > 0:
            relevant = self.loader.retrieve_board_states(
                self.pool.game_type,
                self.pool.get_board_state()["balls"],
                self.boards_K
            )

        # Get new prediction
        prediction = self.jinja.get_prediction_text(relevant)
        prediction = self.llm.generate(prediction)
        prediction = self.llm.parse_prediction(prediction)

        return prediction
    
    def show_results(self, trial_data, verify_state):
        trial_data["passed"] = verify_state == VerifyState.PASS
        trial_data["outcome"] = verify_state.get_name()
        trial_data["context"] = self.llm.get_context()
        return trial_data

    def solve(self, task, trial_data, max_attempts):

    ### 0. Setup 
        self.setup(task)
        trial_data["board_states"].append(self.pool.get_board_state())

    ### 1. Reason and predict
        prediction = self.reason_predict()

        # Fail if no prediction
        if len(prediction) == 0:
            return trial_data
        trial_data["shots"].append(prediction)

        # Take shot
        events, board_state = self.take_shot(prediction)
        trial_data["events"].append(events)
        trial_data["board_states"].append(board_state)

    ### 2. Success check
        verify_state = task.verify(events, board_state)

        if verify_state != VerifyState.INCOMPLETE:
            return self.show_results(trial_data, verify_state)
        
    ### 3. Reset board to initial state
        self.pool.reset()
        
    ### 4. Critique and predict loop
        with Profile("predict_loop"):
            for _ in range(max_attempts):
                with Profile("predict_loop_iteration"):
                    # Critique and predict
                    prediction = self.critique_predict(events, board_state)

                    # Fail if no prediction
                    if len(prediction) == 0:
                        return self.show_results(trial_data, VerifyState.FAIL)
                    trial_data["shots"].append(prediction)

                    # Take shot
                    events, board_state = self.take_shot(prediction)
                    trial_data["events"].append(events)
                    trial_data["board_states"].append(board_state)

                    # Success check
                    verify_state = task.verify(events, board_state)

                    if verify_state != VerifyState.INCOMPLETE:
                        return self.show_results(trial_data, verify_state)
                    
                    self.pool.reset()
            
    ### 5. Return if no success
        return self.show_results(trial_data, verify_state)

# Baseline CoT: uses a language model to generate reasoning and make a prediction with no simulation feedback
#   - Still has the board state after each shot, so is still some feedback just not the events of a shot
class Baseline(SimLM):

    def solve(self, task, trial_data, max_attempts):

    ### 0. Setup 
        self.setup(task)
        trial_data["board_states"].append(self.pool.get_board_state())

    ### 1. Reason and predict
        prediction = self.reason_predict()

        # Fail if no prediction
        if len(prediction) == 0:
            return trial_data
        trial_data["shots"].append(prediction)

        # Take shot
        events, board_state = self.take_shot(prediction)
        trial_data["events"].append(events)
        trial_data["board_states"].append(board_state)

    ### 2. Success check
        verify_state = task.verify(events, board_state)

        if verify_state != VerifyState.INCOMPLETE:
            return self.show_results(trial_data, verify_state)
        
    ### 3. Reset board to initial state
        self.pool.reset()
        
    ### 4. Critique and predict loop
        with Profile("predict_loop"):
            for _ in range(max_attempts):
                with Profile("predict_loop_iteration"):
                    # Reason and predict (not critique, so no simulation feedback just current board state)
                    self.llm.reset() # To remove previous simulation feedback
                    prediction = self.reason_predict()

                    # Fail if no prediction
                    if len(prediction) == 0:
                        return self.show_results(trial_data, VerifyState.FAIL)
                    trial_data["shots"].append(prediction)

                    # Take shot
                    events, board_state = self.take_shot(prediction)
                    trial_data["events"].append(events)
                    trial_data["board_states"].append(board_state)

                    # Success check
                    verify_state = task.verify(events, board_state)

                    if verify_state != VerifyState.INCOMPLETE:
                        return self.show_results(trial_data, verify_state)
                    
                    self.pool.reset()
            
    ### 5. Return if no success
        return self.show_results(trial_data, verify_state)

# Discrete SimLM: uses a language model to generate reasoning and critique from simulation feedback and make a prediction of discrete values that are mapped to a shot
class DiscreteSimLM(SimLM):
    
    def setup(self, task):

       with Profile("solver_setup"):

        # Create board and task
        with Profile("create_board"):
            self.llm.reset()
            if self.pool is not None:
                self.pool.new_state(task.rack)
            else:
                self.pool = Pool(game_type=task.rack, visualizable=VISUALISE)

        if self.examples_K > 0:
            # Get K examples from the datastore
            with Profile("retrieve_examples"):
                self.examples = self.loader.retrieve_examples(
                    self.__class__.__name__,
                    task.prompt,
                    self.examples_K
                )
                if len(self.examples) == 0:
                    print(f"No examples found for task {task.prompt}")
                    self.examples = []

        # Set system prompt
        system_prompt = self.jinja.get_discrete_system_text(self.pool.get_board_state(), self.examples, task.prompt)
        self.llm.set_system_prompt(system_prompt)

    def reason_predict(self):
        
        with Profile("reason_predict"):

            board_state = self.pool.get_board_state()

            # Reasoning
            with Profile("reason"):
                reasoning = self.jinja.get_discrete_reasoning_text()
                self.llm.generate(reasoning)

            # Relevant board states
            relevant = []
            if self.boards_K > 0:
                with Profile("retrieve_board_states"):
                    relevant = self.loader.retrieve_board_states(
                        self.pool.game_type,
                        board_state["balls"],
                        self.boards_K
                    )

            with Profile("predict"):
                # Prediction
                prediction = self.jinja.get_discrete_prediction_text(board_state, relevant)
                prediction = self.llm.generate(prediction)
                prediction = self.parse_discrete_prediction(prediction)

        return prediction
    
    def critique_predict(self, events, board_state):

        with Profile("critique_predict"):

            # Get the events and make text
            simulation = self.jinja.get_simulation_text(events, board_state)

            with Profile("critique"):
                # Get critique text
                critique = self.jinja.get_critique_text()
                sim_critique = simulation + "\n" + critique
                self.llm.generate(sim_critique)

            # Relevant board states
            relevant = []
            if self.boards_K > 0:
                with Profile("retrieve_board_states"):
                    relevant = self.loader.retrieve_board_states(
                        self.pool.game_type,
                        board_state["balls"],
                        self.boards_K
                    )

            with Profile("predict"):
                # Get new prediction
                prediction = self.jinja.get_discrete_prediction_text(board_state, relevant)
                prediction = self.llm.generate(prediction)
                prediction = self.parse_discrete_prediction(prediction)


        return prediction  
    
    def parse_discrete_prediction(self, prediction):


        pattern = r"^([A-Z]+)\s*=\s*(\w+)(?=\s|$)"
        p_clean = prediction.strip()
        remove_list = ["[", "]", "(", ")", "{", "}", ":", "'", '"', ",", "."]
        for char in remove_list:
            p_clean = p_clean.replace(char, "")
        if "\n" in p_clean:
            p_clean = p_clean.split("\n")
        elif "," in p_clean:
            p_clean = p_clean.split(",")
        p_clean = [x.strip() for x in p_clean if x != ""]
        p_clean = "\n".join(p_clean)
        matches = re.findall(pattern, p_clean, re.MULTILINE)

        parsed_prediction = {
            "theta": 2.5,
            "a": 0.0,
            "b": 0.0,
            "V0": 0.25,
            "phi": 0,
        }
        phi_add = 0

        for key, value in matches:

            if key == "BALL" and value != "cue":
                parsed_prediction["phi"] = self.pool.get_azimuth_for_target_ball(value)

            elif key == "SPEED":
                if value == "LOW":
                    parsed_prediction["V0"] = 0.25
                elif value == "MEDIUM":
                    parsed_prediction["V0"] = 1.0
                elif value == "HIGH":
                    parsed_prediction["V0"] = 2.0

            elif key == "SPIN":
                if value == "NONE":
                    parsed_prediction["a"] = 0.0
                    parsed_prediction["b"] = 0.0
                    parsed_prediction["theta"] = 2.5

                elif value == "BACK":
                    parsed_prediction["a"] = 0.0
                    parsed_prediction["b"] = -0.5
                    parsed_prediction["theta"] = 22.5

                elif value == "FRONT":
                    parsed_prediction["a"] = 0.0
                    parsed_prediction["b"] = 0.5
                    parsed_prediction["theta"] = 2.5

                elif value == "LEFT":
                    parsed_prediction["a"] = -0.5
                    parsed_prediction["b"] = 0.0
                    parsed_prediction["theta"] = 2.5

                elif value == "RIGHT":
                    parsed_prediction["a"] = 0.5
                    parsed_prediction["b"] = 0.0
                    parsed_prediction["theta"] = 2.5

            elif key == "SIDE": #TODO: Improve this to implement a "cut" shot
                if value == "CENTER":
                    pass
                elif value == "LEFT":
                    phi_add = 2.5
                elif value == "RIGHT":
                    phi_add = -2.5

        parsed_prediction["phi"] += phi_add


        return parsed_prediction
