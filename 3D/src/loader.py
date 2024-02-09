import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datastore.examples import *
from datastore.board_states import *

SOLVERS = {
    "SimLM": "simlm",
    "Baseline": "baseline",
    "DiscreteSimLM": "discrete",
    "RandomPredictor": "random",
}

class Loader():
    def __init__(self):
        self.example_store = ExampleStore()
        self.board_store = BoardStore()

    def retrieve_examples(self, solver_name, prompt, K):
        filters = {
            "method": SOLVERS[solver_name],
        }
        return self.example_store.retrieve_examples(filters, prompt, K)
    
    def retrieve_board_states(self, game_type, current_balls, K):
        return self.board_store.get_relevant(game_type, current_balls, K)





