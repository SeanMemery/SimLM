import sys, json
from pathlib import Path
from tqdm import tqdm

sys.path.append(str(Path('.').absolute().parent))
from path import *
from src.pool import *


# For each number of balls 
# Generate random board states 
# For each board state, save the outcome of shooting at each ball on the board 

BOARD_STATES_FILE = DATASTORE_DIR + "/board_store/board_states.json"
INITIAL_BOARD_STATES = ["oneball"]
GEN_N = 100000

class BoardStore():
    def __init__(self):
        self.BOARDS = []
        self.load()

    def load(self):
        if os.path.exists(BOARD_STATES_FILE):
            with open(BOARD_STATES_FILE, "r") as f:
                self.BOARDS = json.load(f)
        else:
            self.BOARDS = []

    def generate(self):
        generate_board_states()
        self.load()

    def get_board_states(self):
        return self.BOARDS
    
    def get_similar_board_states(self, game_type, current_balls, K):

        similar_board_states = []
        for bs in self.BOARDS:
            if bs["game_type"] == game_type:
                
                sum_sq_euclidean_dist = sum([
                    ((bs["balls"][id][0] - current_balls[id][0])**2 + 
                    (bs["balls"][id][1] - current_balls[id][1])**2)
                    for id in current_balls.keys()
                ])
                if len(similar_board_states) < K:
                    similar_board_states.append((bs, sum_sq_euclidean_dist))
                    similar_board_states.sort(key=lambda x: x[1])
                elif sum_sq_euclidean_dist < similar_board_states[-1][1]:
                    similar_board_states.append((bs, sum_sq_euclidean_dist))
                    similar_board_states.sort(key=lambda x: x[1])
                    similar_board_states = similar_board_states[:K]

        return [x[0] for x in similar_board_states]

    def get_relevant(self, game_type, current_balls, K):
        similar_board_states = self.get_similar_board_states(game_type, current_balls, K)

        boards = []
        for i in range(K):
            to_show_llm = ""
            bs = similar_board_states[i]
            to_show_llm += f"{bs['initial_board_state']}\n"
            to_show_llm += f"Outcomes:\n"
            for outcome in bs["outcomes"]:
                to_show_llm += f"{outcome['shot_string']}\n"
                to_show_llm += f"{outcome['events']}\n"
                to_show_llm += f"{outcome['final_board_state']}\n"
            boards.append(to_show_llm)

        return boards

def generate_board_states():

    BOARDS = []

    for game_type in INITIAL_BOARD_STATES:
        pool = Pool(game_type)
        skip = False

        for _ in tqdm(range(GEN_N)):
            
            if skip:
                skip = False
                continue

            pool.randomize_positions()

            BOARDS.append({
                "game_type": game_type,
                "balls": pool.get_board_state()["balls"],
                "initial_board_state": pool.get_board_state()["text"],
                "outcomes": []
            })

            for id in pool.balls.keys():
                if id == "cue":
                    continue

                shot = {
                    "V0": 1,
                    "theta": 2.5,
                    "phi": pool.get_azimuth_for_target_ball(id),
                }
                try:
                    pool.strike(**shot)
                except:
                    #print(f"Failed to strike {shot} on board {BOARDS[-1]['balls']}")
                    skip = True 
                    pool.new_state(game_type)
                    BOARDS = BOARDS[:-1]
                    continue

                events = pool.get_events_desc()
                final_board_state = pool.get_board_state()["text"]

                # print(
                #     f"""
                #     =========================================
                #     BOARD STATE 0: {BOARDS[-1]["balls"]}
                #     SHOT: {shot_string}, {shot}
                #     EVENTS: {events}
                #     BOARD STATE 1: {board_state}
                #     =========================================
                # """
                # )

                BOARDS[-1]["outcomes"].append({
                    "target": id,
                    "shot_string": f"BALL={id}\nSPEED=MEDIUM\nSPIN=NONE\n,SIDE=CENTER",
                    "events": events,
                    "final_board_state": final_board_state
                })

                pool.reset()

    with open(BOARD_STATES_FILE, "w") as f:
        print(
            f"Generated {len(BOARDS)} board states for {len(INITIAL_BOARD_STATES)} game types"
        )
        json.dump(BOARDS, f, indent=4)
        
