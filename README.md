# [SimLM: Can Language Models Infer Parameters of Physical Systems?](https://arxiv.org/abs/2312.14215)

Code for the **SimLM** method of _LLMs performing parameter estimation in 2D and 3D scenes_.

## 2D 
Estimating the initial __height__ and __horizontal velocity__ of a ball in 2D as it bounces on different surfaces, with the goal of having the third bounce of the ball occur at a target.

## 3D 
Estimating the parameters needed to take a shot in pool (using the PoolTool simulator):

__V__: The speed at which the cue ball will travel, V ∼ [0, 5].

__theta__: The angle of the cue, in order to impart different levels of spin, theta ∼ [0, 90].

__phi__: The azimuth of the shot, the angle correspondingto the direction the cue ball will travel in, phi ∼ [0, 360].

__a__: The east-to-west coordinate of the point of contact of the cue to the cue ball, a ∼ [−1, 1].

__b__: The north-to-south coordinate of the point of contact of the cue to the cue ball, b ∼ [−1, 1].
