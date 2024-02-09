from jinja2 import Template

class Jinja():
    def __init__(self, prompts) -> None:
        self.templates = prompts

    def get_system_text(self, board_state, examples, task):
        template = Template(self.templates["system"])
        return template.render(
            board_state = board_state["text"],
            examples = examples,
            task = task
        )
    def get_discrete_system_text(self, board_state, examples, task):
        template = Template(self.templates["discrete_system"])
        ball_ids = [ball_id for ball_id in board_state["balls"].keys() if ball_id != "cue"]
        ball_ids = ", ".join(ball_ids)
        return template.render(
            ball_ids = ball_ids,
            board_state = board_state["text"],
            examples = examples,
            task = task
        )
    def get_reasoning_text(self):
        template = Template(self.templates["reasoning"])
        return template.render()
    def get_discrete_reasoning_text(self):
        template = Template(self.templates["discrete_reasoning"])
        return template.render()
    
    def get_prediction_text(self, rel_boards):
        template = Template(self.templates["prediction"])
        return template.render(
            rel_boards = rel_boards
        )
    def get_discrete_prediction_text(self, board_state, rel_boards):
        template = Template(self.templates["discrete_prediction"])
        ball_ids = [ball_id for ball_id in board_state["balls"].keys() if ball_id != "cue"]
        ball_ids = ", ".join(ball_ids)
        return template.render(
            ball_ids = ball_ids,
            rel_boards = rel_boards
        )

    def get_simulation_text(self, events, board_state):
        template = Template(self.templates["simulation"])
        return template.render(
            events = events,
            board_state = board_state["text"]
        )
    
    def get_critique_text(self):
        template = Template(self.templates["critique"])
        return template.render()
    
    def get_continue_text(self):
        template = Template(self.templates["continue"])
        return template.render()