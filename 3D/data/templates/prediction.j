{% if rel_boards %}
Below are a collection of similar positions I have been in previously. I will use these to inform my prediction, as they may give some clues to how the next shot will go, but I must keep in mind that these positions are only similar to my current position, and simply replicating these shots may not be successful.
{% for board in rel_boards %}
### START BOARD POSITION {{ loop.index }} ###
{{ board }}
### END BOARD POSITION {{ loop.index }} ###
{% endfor %}
{% endif %}
Therefore, the prediction of the physical properties of the cue stick striking the cue ball is below, given in the EXACT values, with each property in caps, on a newline, and an equal sign to its value e.g. "SPEED=5":
### Prediction