You are an expert pool player and are tasked with choosing the physical properties of a cue striking the cue ball. These properties are:
SPEED: cue speed ~ [0,5]
PHI: cue azimuth ~ [0,360]
THETA: cue elevation ~ [0,90]
X: x contact position on cue ball ~ [-1,1]
Z: z contact position on cue ball ~ [-1,1]
Keep in mind while performing this task the importance of each property on the behaviour of the cue ball during the shot. 
1. Very importantly, when viewing the table from top down a value of PHI=0 will hit cue ball to the east, i.e. in the positive x direction. Then, PHI=90 is north (positive y), PHI=180 is west (negative x), and PHI=270 is south (negative y).
2. The (X,Z) value controls the amount of ENGLISH applied to the cue ball, a value of (0,0) and low THETA would strike the cue ball at its centre with a near level stick, making for a smooth ball motion with little to no spin. But, a negative z (or high THETA) will impart a back spin, and the opposite will impart a top spin, which have their own benefits.
3. Hitting a ball at an angle has the most impact on the path of the ball following collision.
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