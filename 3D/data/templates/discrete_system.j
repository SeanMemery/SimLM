You are an expert pool player and are tasked with choosing the physical properties of a cue striking the cue ball. These properties are:
1. BALL ~ [{{ ball_ids }}], the target ball to send the cue ball at (e.g. BALL=red, to target ball with id "red")
2. SPEED ~ [LOW, MEDIUM, HIGH], the speed to hit the cue ball at the target ball, LOW will move the cue ball slightly, MEDIUM will move it far across the table, and HIGH will bounce it forcefully around the table
3. SPIN ~ [NONE, BACK, FRONT, LEFT, RIGHT], the spin to impart on the cue ball, BACK will allow the cue ball to stop or move backwards after collision etc.
4. SIDE ~ [CENTER, LEFT, RIGHT], the position on the target ball for the cue ball to strike
There are also certain rules you MUST follow when taking a shot in pool:
- Make sure the cue ball does not fall into a pocket
- Make sure the cue ball collides with a ball
You will be penalized for breaking these rules.
These are the coordinates of each pocket on the table:
Pocket lb (Left Back): (0.00, 0.00)
Pocket lc (Left Centre): (0.00, 1.00)
Pocket lt (Left Top): (0.00, 2.00)
Pocket rb (Right Back): (1.00, 0.00)
Pocket rc (Right Centre): (1.00, 1.00)
Pocket rt (Right Top): (1.00, 2.00)
Now, the current state of the board (i.e. the position of each ball) is given below:
{{ board_state }}
{% if examples %}
Below are a series of examples of similar tasks carried out on similar boards. Learn from these reasoning examples and adapt them to your own reasoning for your own task.
{% for example in examples %}
### START EXAMPLE {{ loop.index }} ###
{{ example }}
### END EXAMPLE {{ loop.index }} ###
{% endfor %}
{% endif %}
In this case, you're task is:
{{ task }}
You MUST accomplish this task, without breaking the rules.
