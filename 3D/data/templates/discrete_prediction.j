Therefore, the prediction of the physical properties of the cue stick striking the cue ball is below. 
The properties are 
1. BALL ~ [{{ ball_ids }}], the target ball to send the cue ball at (e.g. BALL=red, to target ball with id "red")
2. SPEED ~ [LOW, MEDIUM, HIGH], the speed to hit the cue ball at the target ball
3. SPIN ~ [NONE, BACK, FRONT, LEFT, RIGHT], the spin to impart on the cue ball
4. SIDE ~ [CENTER, LEFT, RIGHT], the position on the target ball for the cue ball to strike
{% if rel_boards %}
Below are a collection of similar positions I have been in previously. I will use these to inform my prediction, as they may give some clues to how the next shot will go, but I must keep in mind that these positions are only similar to my current position, and simply replicating these shots may not be successful.
{% for board in rel_boards %}
### START BOARD POSITION {{ loop.index }} ###
{{ board }}
### END BOARD POSITION {{ loop.index }} ###
{% endfor %}
{% endif %}
Each property is given one of the values specified and is on a newline with an equal sign to its value e.g. "SPEED=LOW":
### Prediction