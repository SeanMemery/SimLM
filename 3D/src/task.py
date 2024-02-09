from enum import Enum
from profiler import *

POCKETS = {
    "lb": (0.00, 0.00),
    "lc": (0.00, 1.00),
    "lt": (0.00, 2.00),
    "rb": (1.00, 0.00),
    "rc": (1.00, 1.00),
    "rt": (1.00, 2.00),
}

class VerifyState(Enum):
    PASS = 0
    FAIL = 1
    INCOMPLETE = 2

    def get_name(self):
        return self.name.lower()

def near_pocket(args, board_state):
    """
    Check if the ball is near a pocket
        - args = [ball_id, pocket_id]
    """
    ball_id = args[0]
    if ball_id not in board_state["balls"].keys():
        return False
    ball_pos = board_state["balls"][ball_id]

    pocket_id = args[1]

    # Check if the ball is near any pocket
    if pocket_id == "any":
        for p_id, p_pos in POCKETS.items():
            if near_pocket([ball_id, p_id], board_state):
                return True
        return False
    elif pocket_id == "none":
        for p_id, p_pos in POCKETS.items():
            if near_pocket([ball_id, p_id], board_state):
                return False
        return True
    
    assert pocket_id in POCKETS.keys(), "Pocket ID does not exist"
    pocket_pos = POCKETS[pocket_id]

    radius = 0.2

    return (ball_pos[0] - pocket_pos[0])**2 + (ball_pos[1] - pocket_pos[1])**2 < radius**2


# FUNCTION NAME : FUNCTION
FUNCTIONS = {
    "NEAR_POCKET" : near_pocket,
}

class Event():
    def __init__(self, text):
        self.text = text
        self.function = None
        self.function_name = None
        
        # Check if the event is a function
        for function_name, function in FUNCTIONS.items():
            if function_name in text:
                self.function_name = function_name
                self.function = function
                break

    def verify_event(self, log, board_state):
        """
        Verify that the event has occurred 
        """

        if self.function is None:
            return self.text in log

        # Get text between parentheses in self.text, separated by comma
        args = self.text[self.text.find("(")+1:self.text.find(")")].split(",")
        args = [arg.strip() for arg in args]
        return self.function(args, board_state)

class Task():
    def __init__(self, prompt, rack, pass_events, fail_events, conditions):
        self.prompt = prompt
        self.rack = rack
        self.pass_events = pass_events
        self.fail_events = fail_events
        self.pass_conditions = conditions["pass"]
        self.fail_conditions = conditions["fail"]

    def verify(self, log, board_state):
        """
        Verify that the task was successful by checking the log
            - Make sure each element of verify_text is in the log in the correct order
            - Returns True if the task was successful, False otherwise
        """


        def verify_all_ordered(events, log, board_state):
            """
            Verify that all events are in the log in the correct order
            """
            for i, text in enumerate(events):
                event = Event(text)
                if not event.verify_event(log, board_state):
                    return False
                if i > 0:
                    if log.index(text) < log.index(events[i-1]):
                        return False
            return True
        
        def verify_all_unordered(events, log, board_state):
            """
            Verify that all events are in the log in any order
            """
            for text in events:
                event = Event(text)
                if not event.verify_event(log, board_state):
                    return False
            return True
        
        def verify_any(events, log, board_state):
            """
            Verify that any event is in the log
            """
            for text in events:
                event = Event(text)
                if event.verify_event(log, board_state):
                    return True
            return False

        def general_verify(events, conditions, log, board_state):
            # all ordered 
            if "all" in conditions and "ordered" in conditions:
                return verify_all_ordered(events, log, board_state)

            # all unordered
            if "all" in conditions and "ordered" not in conditions:
                return verify_all_unordered(events, log, board_state)
            
            # any
            if "any" in conditions:
                return verify_any(events, log, board_state)

            # Ill posed task
            raise ValueError("Task conditions are ill posed")


        if general_verify(self.fail_events, self.fail_conditions, log, board_state):
            return VerifyState.FAIL
        elif general_verify(self.pass_events, self.pass_conditions, log, board_state):
            return VerifyState.PASS
        else:
            return VerifyState.INCOMPLETE