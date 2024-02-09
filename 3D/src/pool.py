import pooltool as pt
import math, random
import numpy as np

DEFAULTS = {
    "V0": 1,
    "theta": 0,
    "phi": 0,
    "a": 0,
    "b": 0
}

class Pool():

    def __init__(self, game_type=None, visualizable=False) -> None:

        if game_type is None:
            game_type = "oneball"

        assert game_type.upper() in pt.GameType._member_names_, f"Invalid game type {game_type}"
        self.visualizable = visualizable
        self.interface = pt.ShotViewer() if self.visualizable else None
        self.table = pt.Table.default()
        self.game_type = game_type
        self.balls = self.setup_balls(game_type)
        self.cue = pt.Cue(cue_ball_id="cue")
        self.shot = pt.System(table=self.table, balls=self.balls, cue=self.cue)

    def setup_balls(self, game_type):

        if game_type == "oneball":
            ang = 2 * np.pi * np.random.rand()
            x = 0.5   + 0.2  * np.cos(ang)
            y = 0.5   + 0.2 * np.sin(ang)
            return {
                "cue": pt.Ball.create("cue", xy=(0.5, 1)),
                "red": pt.Ball.create("red", xy=(x, y), ballset=pt.game.layouts.DEFAULT_SNOOKER_BALLSET),
            }
        
        elif game_type == "twoball":
            ang1 = 2 * np.pi * np.random.rand()
            x1 = 0.5   + 0.2  * np.cos(ang1)
            y1 = 0.5   + 0.2 * np.sin(ang1)

            ang2 = 2 * np.pi * np.random.rand()
            x2 = 0.5   + 0.2  * np.cos(ang2)
            y2 = 0.5   + 0.2 * np.sin(ang2)

            return {
                "cue": pt.Ball.create("cue", xy=(0.5, 1)),
                "red": pt.Ball.create("red", xy=(x1, y1), ballset=pt.game.layouts.DEFAULT_SNOOKER_BALLSET),
                "yellow": pt.Ball.create("yellow", xy=(x2, y2), ballset=pt.game.layouts.DEFAULT_SNOOKER_BALLSET),
            }
        
    def randomize_positions(self):
        return self.shot.randomize_positions()

    def get_cue_and_balls(self):
        return self.shot.cue, self.shot.balls

    def new_state(self, game_type):
        assert game_type.upper() in pt.GameType._member_names_, f"Invalid game type {game_type}"

        self.shot.reset_history()
        self.shot.reset_balls()

        self.table = pt.Table.default()
        self.balls = self.setup_balls(game_type)
        self.cue = pt.Cue(cue_ball_id="cue")

        del self.shot.table 
        del self.shot.balls
        del self.shot.cue

        self.shot.table = self.table
        self.shot.balls = self.balls
        self.shot.cue = self.cue

    def reset(self):
        self.shot.reset_balls()
        self.shot.reset_history()

    def strike(self, V0=DEFAULTS["V0"], phi=DEFAULTS["phi"], theta=DEFAULTS["theta"], a=DEFAULTS["a"], b=DEFAULTS["b"]):
        self.shot.strike(V0=V0, phi=phi, theta=theta, a=a, b=b)
        pt.simulate(self.shot, inplace=True)

    def get_board_state(self):
        return self.shot.get_board_state()

    def visualise(self):
        if self.visualizable:
            self.interface.show(self.shot)
        else:
            raise Exception("Cannot visualize without visualizable=True")

    def get_events_desc(self):
        null_events = ["Null event", "sliding to rolling"]
        event_descriptions = []
        for event in self.shot.events:
            if any([x in event.description for x in null_events]):
                continue
            event_descriptions.append(event.description )
        event_descriptions = list(dict.fromkeys(event_descriptions))
        return "\n".join(event_descriptions)
    
    def get_azimuth_for_target_ball(self, target_ball_id):
        target_ball_id = target_ball_id.lower()
        if not target_ball_id in self.balls.keys():
            print(f"Ball {target_ball_id} not found in {self.balls.keys()}")
            return 0

        target_ball_pos = self.balls[target_ball_id].xyz
        cue_position = self.balls["cue"].xyz

        direction = pt.ptmath.unit_vector(np.array(target_ball_pos) - np.array(cue_position))
        direction = math.atan2(direction[1], direction[0])
        direction = direction if direction >= 0 else 2 * np.pi + direction

        return direction * 180 / np.pi