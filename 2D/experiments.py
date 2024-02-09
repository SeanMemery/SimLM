import json, os, time, guidance, random
import numpy as np
from tqdm import tqdm
from sim import EvenSimulation, UnevenSimulation, CustomSimulation
import guidance_setup

def sample_surface(difficulty):
    ### Generate a random surface using a difficulty level that varies from 1 to 10
    ### Difficulty 1: almost flat surface
    ### Difficulty 10: very rough and uneven surface

    d = (difficulty-1) / 9.0

    EASY_FUNC = lambda x: 0.15*np.sin(0.25 * x)
    HARD_FUNC = lambda x: 0.6*np.sin(0.9*x) + 0.15*np.sin(2.25 * x) + 0.05*np.sin(4.5*x)
    FUNC = lambda x: (1-d)*EASY_FUNC(x) + d*HARD_FUNC(x)
    
    return FUNC

### FLAT SURFACE - Forward Problem ###
class ExperimentForward():
    def __init__(self, target_bounce, analogical, varying_params):
        self.sim = EvenSimulation()
        self.INITIAL_CONDITIONS = "{\"height\": 8.1, \"horizontal_velocity\": 10.15}"
        self.target = 50

        self.surface_descriptions = ["A simple flat surface at y = 0."]

    def reasoning_api(self, target, examples):
        n0_reasoning = guidance("""
{{#system~}}
You're an expert physicist running an experiment. You have a ball and it's initial height and horizontal velocity in 2D. You want to \
predict the position of the ball's third bounce. \
Below is a description of the surface the ball will bounce on: 
SURFACE:
{{surface_description}} 
{{examples}}
### REAL CASE:
You want to predict a single value for the horizontal distance of the third bounce of the ball. The known initial conditions of the ball are:
{{INITIAL_CONDITIONS}}
What is the horizontal distance of the third bounce of the ball? Keep in mind the coefficient of restitution here is 0.9 and you want to predict the THIRD bounce of the ball. Be sure to be concise and not include any extra information.
{{~/system}}
{{#user~}}
Give your reasoning and calculations for the task in a single paragraph.
REASONING:
{{~/user}} 
{{#assistant~}}
{{gen 'reasoning' temperature=1 max_tokens=384}}
{{~/assistant}}
{{#user~}}
Therefore, make a prediction for what the horizontal distance value should be and output below as a JSON object in this format:
{
    "horizontal_distance": d
}    
Output nothing else, just the JSON object.
{{~/user}}
{{#assistant~}}
{{gen 'prediciton' temperature=0.7 max_tokens=64}}
{{~/assistant}}
""", caching=False) 

        t0 = time.time()

        response = n0_reasoning(examples=examples, surface_description=self.surface_descriptions[0], INITIAL_CONDITIONS=self.INITIAL_CONDITIONS)
        try:
            ic_final = response["prediciton"]
            ic_final = ic_final[ic_final.index("{") : ic_final.rindex("}") + 1]
            ic_final = json.loads(ic_final)
        except Exception as e:
            print(f"ERROR: exception: {e}, for reponse {response}")
            guidance_setup.reload_model()
            return [{"ic":{"horizontal_distance":0}, "bounces": [0,0,0], "response":str(response), "attempts":0, "time":0}]

        t = time.time() - t0    
        return [{"ic":ic_final, "bounces": [0,0,ic_final["horizontal_distance"]], "response":str(response), "attempts":1, "time":t}]

    def reasoning_local(self, target, examples):
        n0_reasoning = guidance("""You're an expert physicist running an experiment. You have a ball and it's initial height and horizontal velocity in 2D. You want to \
predict the position of the ball's third bounce. \
Below is a description of the surface the ball will bounce on: 
SURFACE:
{{surface_description}} 
{{examples}}
### REAL CASE:
You want to predict a single value for the horizontal distance of the third bounce of the ball. The known initial conditions of the ball are:
{{INITIAL_CONDITIONS}}
What is the horizontal distance of the third bounce of the ball? Keep in mind the coefficient of restitution here is 0.9 and you want to predict the THIRD bounce of the ball.
{{gen 'reasoning' max_tokens=256 do_sample=True temperature=0.4}}
Therefore, the horizontal distance value should be:
```json
{
    "horizontal_distance": {{gen 'prediction' pattern='\d*(\.\d(3))?\,' stop=',' do_sample=False}}
}```
""", caching=False)   

        t0 = time.time()

        response = n0_reasoning(examples=examples, surface_description=self.surface_descriptions[0], INITIAL_CONDITIONS=self.INITIAL_CONDITIONS)
        try:
            ic_final = {
                "horizontal_distance": float(response["prediction"]),
            }
        except Exception as e:
            print(f"ERROR: exception: {e}, for reponse {response}")
            guidance_setup.reload_model()
            return [{"ic":{"horizontal_distance":0}, "bounces": [0,0,0], "response":str(response), "attempts":0, "time":0}]

        t = time.time() - t0    
        return [{"ic":ic_final, "bounces": [0,0,ic_final["horizontal_distance"]], "response":str(response), "attempts":1, "time":t}]

### FLAT SURFACE ###
class Experiment1():
    def __init__(self, target_bounce, analogical, varying_params):
        self.sim = EvenSimulation()
        self.TARGET_BOUNCE = target_bounce
        self.ANALOGICAL = analogical
        self.MAX_ATTEMPTS = 5

        self.surface_descriptions = ["A simple flat surface at y = 0."]

    def reasoning_api(self, target, examples):

        ### n0_reasoning
        n0_reasoning = guidance("""
{{#system~}}
You're an expert physicist running an experiment. You have a ball and want to \
predict initial height and horizontal velocity values such that the {{target_bounce}} bounce of the ball lands at a specific horizontal position. \
Once you predict the initial conditions a computer simulation will give you the exact positions of the ball's bounces. Below is a description of the surface the ball will bounce on: 
SURFACE:
{{surface_description}} 
{{examples}}
### REAL CASE:
You want the horizontal distance of the {{target_bounce}} bounce of the ball to be within 3m of {{target}}m, this is THE MOST IMPORTANT part of the task. What height and horizontal velocity values should you use? \
Be VERY careful when deciding if a bounce position is within 3m of the target, for a target of {{target}}m the {{target_bounce}} bounce must occur between {{t1}}m and {{t2}}m in order to be 3m from {{target}}m.  
{{~/system}}
{{#user~}}
Give your reasoning for the task in a single paragraph.
REASONING:
{{~/user}} 
{{#assistant~}}
{{gen 'reasoning' temperature=0.4 max_tokens=128}}
{{~/assistant}}
{{#user~}}
Therefore, make a prediction for what the height and horizontal velocity values should be and output below as a JSON object in this format:
{
    "height": h,
    "horizontal_velocity": v
}    
Output nothing else, just the JSON object.
{{~/user}}
{{#assistant~}}
{{gen 'IC' temperature=0.7 max_tokens=64}}
{{~/assistant}}
{{#user~}}
SIMULATION:
After running the simulation, the three bounces occurred at:
{{get_bounces IC}}
Give a critique of the reasoning and simulation results, in a single paragraph. Be sure to think on why the simulation results may differ from what you expected, and what you can learn from them.
CRITIQUE:
{{~/user}}
{{#assistant~}}
{{gen 'critique' temperature=0.4 max_tokens=128}}
{{~/assistant}}
{{#user~}}
Do the simulation results match the target? Answer either Yes or No.
Answer:
{{~/user}}
{{#assistant~}}
{{gen "answer" temperature=0.7 max_tokens=8}}
{{~/assistant}}
""", caching=False) 

        ### n_reasoning
        n_reasoning = guidance("""
{{context}}
{{#user~}}
Therefore, make a new prediction for what the updated height and horizontal velocity values should be and output below as a JSON object in this format:
{
    "height": h,
    "horizontal_velocity": v
}    
Output nothing else, just the JSON object.
{{~/user}} 
{{#assistant~}}
{{gen 'IC' temperature=0.7 max_tokens=64}}
{{~/assistant}} 
{{#user~}}
SIMULATION:
After running the simulation, the three bounces occurred at:
{{get_bounces IC}}
Give a critique of the reasoning and simulation results, in a single paragraph.
CRITIQUE:
{{~/user}}
{{#assistant~}}
{{gen 'critique' temperature=0.4 max_tokens=128}}
{{~/assistant}}
{{#user~}}
Do the simulation results match the target? Answer either Yes or No.
Answer:
{{~/user}}
{{#assistant~}}
{{gen "answer" temperature=0.7 max_tokens=8}}
{{~/assistant}}
""", caching=False)
                            
        ### nn_reasoning
        nn_reasoning = guidance("""
{{context}}                  
{{#user~}}
FINAL ANSWER:
After verifying the reasoning with the simulation results, output the final answer below as a JSON object in this format:
{
    "height": h,
    "horizontal_velocity": v
}
Output nothing else, just the JSON object.     
{{~/user}}
                        
{{#assistant~}}
{{gen 'final_IC' temperature=0.7 max_tokens=64}}
{{~/assistant}}   
""", caching=False)

        t0 = time.time()

        response = n0_reasoning(examples=examples, get_bounces=self.sim.get_bounces_as_string_IC, surface_description=self.surface_descriptions[0], target_bounce=self.TARGET_BOUNCE[1], target=target, t1=round(target-3,2), t2=round(target+3,2))
        try:
            ### REPEAT REASONING UNTIL TARGET/LIMIT IS REACHED
            C = 0

            while "No" in response["answer"]:
                C += 1
                response = n_reasoning(context=str(response), get_bounces=self.sim.get_bounces_as_string_IC)
                if C >= self.MAX_ATTEMPTS:
                    break
                ### Sleep to avoid API limit
                #time.sleep(5)
            ### REPEAT REASONING UNTIL TARGET/LIMIT IS REACHED
            response = nn_reasoning(context=str(response))
            ic_final = response["final_IC"]
            ic_final = ic_final[ic_final.index("{") : ic_final.rindex("}") + 1]
            ic_final = json.loads(ic_final)
            bounces = self.sim.get_bounces(ic_final["height"], ic_final["horizontal_velocity"])
        except Exception as e:
            print(f"ERROR: exception: {e}")
            guidance_setup.reload_model()
            return [{"ic":{"height":0, "horizontal_velocity":0}, "bounces": [0,0,0], "response":str(response), "attempts":0, "time":0}]

        t = time.time() - t0    
        return [{"ic":ic_final, "bounces": bounces, "response":str(response), "attempts":C, "time":t}]

    def reasoning_local(self, target, examples):
        """
        Perform a guided simulation, where the LLM reasons about the query.
        Q -> (QRPCA)Q -> (QRPCA)QR -> (QRPCA)QRPC -> (QRPCA)QRPCA

        1) The LLM attempts an initial reasoning step R
        2) The LLM decided if a simulation is needed and new set of IC is generated
        3) Simulation results are retrieved 
        4) The LLM critiques its reasoning based on the results
        5) Repeat 2-4 until the target is reached
        6) The LLM outputs a final set of IC
        
        Parameters:
        - sim: The simulation to use.
        - target: The target distance.

        Returns:
        - ic_final: The final set of initial conditions.
        """ 

        n0_reasoning = guidance("""You're an expert physicist running an experiment. You have a ball and want to \
predict initial height and horizontal velocity values such that the {{target_bounce}} bounce of the ball lands at a specific horizontal position. \
Once you predict the initial conditions a computer simulation will give you the exact positions of the ball's bounces. Below is a description of the surface the ball will bounce on: 
SURFACE:
{{surface_description}} 
Give your reasoning for the task in a single paragraph.
REASONING:
{{gen 'reasoning' max_tokens=128 do_sample=True temperature=0.4}}
Therefore, the height and horizontal velocity values should be:
```json
{
    "height": {{gen 'height' pattern='\d*(\.\d(3))?\,' stop=',' do_sample=False}},
    "horizontal_velocity": {{gen 'horizontal_velocity' pattern='\d*(\.\d(3))?\,' stop=',' do_sample=False}},
}```
SIMULATION:
After running the simulation, the three bounces occurred at:
{{get_bounces height horizontal_velocity}}
Give a critique of the reasoning and simulation results, in a single paragraph. Be sure to think on why the simulation results may differ from what you expected, and what you can learn from them.
CRITIQUE:
{{gen 'critique' max_tokens=128 do_sample=True temperature=0.4}}
Do the simulation results match the target? (Yes/No)
Answer:{{gen "answer" max_tokens=8 do_sample=False}}
""", caching=False)   
    
        n_reasoning = guidance("""{{context}}
Therefore, the height and horizontal velocity values must be adjusted, the new values are:
```json\
{
    "height": {{gen 'height' pattern='\d*(\.\d(3))?\,' stop=',' do_sample=False}},
    "horizontal_velocity": {{gen 'horizontal_velocity' pattern='\d*(\.\d(3))?\,' stop=',' do_sample=False}},
}```
SIMULATION:
After running the simulation, the three bounces occurred at:
{{get_bounces height horizontal_velocity}}
CRITIQUE:
{{gen 'critique' max_tokens=128 do_sample=True temperature=0.4}}
Do the simulation results match the target? (Yes/No)
Answer:{{gen "answer" max_tokens=8 do_sample=False}}
""", caching=False)
                        
        nn_reasoning = guidance("""{{context}}
FINAL ANSWER:
After verifying the reasoning with the simulation results, the final answer is:
```json
{
    "height": {{gen 'height_final' pattern='\d*(\.\d(3))?\,' stop=',' do_sample=False}},
    "horizontal_velocity": {{gen 'horizontal_velocity_final' pattern='\d*(\.\d(3))?\,' stop=',' do_sample=False}},
}```""", caching=False)

        t0 = time.time()

        response = n0_reasoning(examples=examples, get_bounces=self.sim.get_bounces_as_string, surface_description=self.surface_descriptions[0], target_bounce=self.TARGET_BOUNCE[1], target=target, t1=round(target-3,2), t2=round(target+3,2), analogical=self.ANALOGICAL)

        try:
            ### REPEAT REASONING UNTIL TARGET/LIMIT IS REACHED
            C = 0

            while "No" in response["answer"]:
                C += 1
                response = n_reasoning(context=response, get_bounces=self.sim.get_bounces_as_string)
                if C >= self.MAX_ATTEMPTS:
                    break
            ### REPEAT REASONING UNTIL TARGET/LIMIT IS REACHED

            response = nn_reasoning(context=response)
            ic_final = {
                "height": float(response["height_final"]),
                "horizontal_velocity": float(response["horizontal_velocity_final"]),
            }
            bounces = self.sim.get_bounces(ic_final["height"], ic_final["horizontal_velocity"])
        except Exception as e:
            print("ERROR: ", e)
            print("RESPONSE:", str(response))
            guidance_setup.reload_model()
            return [{"ic":{"height":0, "horizontal_velocity":0}, "bounces": [0,0,0], "response":str(response), "attempts":0, "time":0}]

        t = time.time() - t0    
        return [{"ic":ic_final, "bounces": bounces, "response":str(response), "attempts":C, "time":t}]

### UNEVEN SURFACE - No Variation ###
class Experiment2():
    def __init__(self, target_bounce, analogical, varying_params):
        self.sims = [UnevenSimulation()]
        self.TARGET_BOUNCE = target_bounce
        self.ANALOGICAL = analogical
        self.MAX_ATTEMPTS = 5

        self.surface_descriptions = ["A simple sine wave with a frequency of 1 and amplitude of 1."]

    def reasoning_api(self, target, examples):

        results = []
        for i, SIM_STEP in tqdm(enumerate(self.sims)):

            ### n0_reasoning
            if self.ANALOGICAL:
                n0_reasoning = guidance("""
{{#system~}}
You're an expert physicist running an experiment. You have a ball and want to \
predict initial height and horizontal velocity values such that the {{target_bounce}} bounce of the ball lands at a specific horizontal position. \
Once you predict the initial conditions a computer simulation will give you the exact positions of the ball's bounces. Below is a description of the surface the ball will bounce on: 
SURFACE:
{{surface_description}}
The surface the ball bounces on is uneven so the ball's trajectory may be difficult to predict without repeated simulations and trial and error. 
Problem: You want the horizontal distance of the {{target_bounce}} bounce of the ball to be within 3m of {{target}}m, this is THE MOST IMPORTANT part of the task. What height and horizontal velocity values should you use? \
Be VERY careful when deciding if a bounce position is within 3m of the target, for a target of {{target}}m the {{target_bounce}} bounce must occur between {{t1}}m and {{t2}}m in order to be 3m from {{target}}m. 
Relevant problems: Recall three relevant and distinct problems. For each problem, describe it and explain the solution.
{{~/system}}
{{#assistant~}}
{{gen 'rel_probs' max_tokens=768 temperature=0.9}}
{{~/assistant}}
{{#user~}}
Solve the initial problem:
Give your reasoning for the task in a single paragraph.
REASONING:
{{~/user}}
{{#assistant~}}
{{gen 'reasoning' temperature=0.4 max_tokens=128}}
{{~/assistant}}
{{#user~}}
Therefore, make a prediction for what the height and horizontal velocity values should be and output below as a JSON object in this format:
{
    "height": h,
    "horizontal_velocity": v
}    
Output nothing else, just the JSON object.
{{~/user}}
{{#assistant~}}
{{gen 'IC' temperature=0.7 max_tokens=64}}
{{~/assistant}}
{{#user~}}
SIMULATION:
After running the simulation, the three bounces occurred at:
{{get_bounces IC}}
Give a critique of the reasoning and simulation results, in a single paragraph. Be sure to think on why the simulation results may differ from what you expected, and what you can learn from them.
CRITIQUE:
{{~/user}}
{{#assistant~}}
{{gen 'critique' temperature=0.4 max_tokens=128}}
{{~/assistant}}
{{#user~}}
Do the simulation results match the target? Answer either Yes or No.
Answer:
{{~/user}}
{{#assistant~}}
{{gen "answer" temperature=0.7 max_tokens=8}}
{{~/assistant}}
""", caching=False) 
            else:
                n0_reasoning = guidance("""
{{#system~}}
You're an expert physicist running an experiment. You have a ball and want to \
predict initial height and horizontal velocity values such that the {{target_bounce}} bounce of the ball lands at a specific horizontal position. \
Once you predict the initial conditions a computer simulation will give you the exact positions of the ball's bounces. Below is a description of the surface the ball will bounce on: 
SURFACE:
{{surface_description}} 
The surface the ball bounces on is uneven so the ball's trajectory may be difficult to predict without repeated simulations and trial and error. 
{{examples}}
### REAL CASE:
You want the horizontal distance of the {{target_bounce}} bounce of the ball to be within 3m of {{target}}m, this is THE MOST IMPORTANT part of the task. What height and horizontal velocity values should you use? \
Be VERY careful when deciding if a bounce position is within 3m of the target, for a target of {{target}}m the {{target_bounce}} bounce must occur between {{t1}}m and {{t2}}m in order to be 3m from {{target}}m.  
{{~/system}}
{{#user~}}
Give your reasoning for the task in a single paragraph.
REASONING:
{{~/user}}
{{#assistant~}}
{{gen 'reasoning' temperature=0.4 max_tokens=128}}
{{~/assistant}}
{{#user~}}
Therefore, make a prediction for what the height and horizontal velocity values should be and output below as a JSON object in this format:
{
    "height": h,
    "horizontal_velocity": v
}    
Output nothing else, just the JSON object.
{{~/user}}
{{#assistant~}}
{{gen 'IC' temperature=0.7 max_tokens=64}}
{{~/assistant}}
{{#user~}}
SIMULATION:
After running the simulation, the three bounces occurred at:
{{get_bounces IC}}
Give a critique of the reasoning and simulation results, in a single paragraph. Be sure to think on why the simulation results may differ from what you expected, and what you can learn from them.
CRITIQUE:
{{~/user}}
{{#assistant~}}
{{gen 'critique' temperature=0.4 max_tokens=128}}
{{~/assistant}}
{{#user~}}
Do the simulation results match the target? Answer either Yes or No.
Answer:
{{~/user}}
{{#assistant~}}
{{gen "answer" temperature=0.7 max_tokens=8}}
{{~/assistant}}
    """, caching=False) 

            ### n_reasoning
            n_reasoning = guidance("""
{{context}}
{{#user~}}
Therefore, make a new prediction for what the updated height and horizontal velocity values should be and output below as a JSON object in this format:
{
    "height": h,
    "horizontal_velocity": v
}    
Output nothing else, just the JSON object.
{{~/user}}
{{#assistant~}}
{{gen 'IC' temperature=0.7 max_tokens=64}}
{{~/assistant}} 
{{#user~}}
SIMULATION:
After running the simulation, the three bounces occurred at:
{{get_bounces IC}}
Give a critique of the reasoning and simulation results, in a single paragraph. 
CRITIQUE:
{{~/user}}
{{#assistant~}}
{{gen 'critique' temperature=0.4 max_tokens=128}}
{{~/assistant}}
{{#user~}}
Do the simulation results match the target? Answer either Yes or No.
Answer:
{{~/user}}
{{#assistant~}}
{{gen "answer" temperature=0.7 max_tokens=8}}
{{~/assistant}}""", caching=False)

            ### nn_reasoning
            nn_reasoning = guidance("""
{{context}}                  
{{#user~}}
FINAL ANSWER:
After verifying the reasoning with the simulation results, output the final answer below as a JSON object in this format:
{
    "height": h,
    "horizontal_velocity": v
}
Output nothing else, just the JSON object.
{{~/user}}
{{#assistant~}}
{{gen 'final_IC' temperature=0.7 max_tokens=64}}
{{~/assistant}}""", caching=False)

            t0 = time.time()

            response = n0_reasoning(examples=examples, get_bounces=SIM_STEP.get_bounces_as_string_IC, surface_description=self.surface_descriptions[i], target_bounce=self.TARGET_BOUNCE[1], target=target, t1=round(target-3,2), t2=round(target+3,2), analogical=self.ANALOGICAL)
            try:
                ### REPEAT REASONING UNTIL TARGET/LIMIT IS REACHED
                C = 0
    
                while "No" in response["answer"]:
                    C += 1
                    response = n_reasoning(context=str(response), get_bounces=SIM_STEP.get_bounces_as_string_IC)
                    if C >= self.MAX_ATTEMPTS:
                        break
                    ### Sleep to avoid API limit
                    #time.sleep(5)
                ### REPEAT REASONING UNTIL TARGET/LIMIT IS REACHED
                response = nn_reasoning(context=str(response))
                ic_final = response["final_IC"]
                ic_final = ic_final[ic_final.index("{") : ic_final.rindex("}") + 1]
                ic_final = json.loads(ic_final)
                bounces = SIM_STEP.get_bounces(ic_final["height"], ic_final["horizontal_velocity"])
            except Exception as e:
                print(f"ERROR: exception: {e}")
                guidance_setup.reload_model()
                continue

            t = time.time() - t0    
            results.append({"ic":ic_final, "bounces": bounces, "response":str(response), "attempts":C, "time":t})

        return results

    def reasoning_local(self, target, examples):
        """
        Perform a guided simulation, where the LLM reasons about the query.
        Q -> (QRPCA)Q -> (QRPCA)QR -> (QRPCA)QRPC -> (QRPCA)QRPCA

        1) The LLM attempts an initial reasoning step R
        2) The LLM decided if a simulation is needed and new set of IC is generated
        3) Simulation results are retrieved 
        4) The LLM critiques its reasoning based on the results
        5) Repeat 2-4 until the target is reached
        6) The LLM outputs a final set of IC
        
        Parameters:
        - sim: The simulation to use.
        - target: The target distance.

        Returns:
        - ic_final: The final set of initial conditions.
        """ 
        results = []
        for i, SIM_STEP in tqdm(enumerate(self.sims)):

            n0_reasoning = guidance("""You're an expert physicist running an experiment. You have a ball and want to \
predict initial height and horizontal velocity values such that the {{target_bounce}} bounce of the ball lands at a specific horizontal position. \
Once you predict the initial conditions a computer simulation will give you the exact positions of the ball's bounces. Below is a description of the surface the ball will bounce on: 
SURFACE:
{{surface_description}} 
The surface the ball bounces on is uneven so the ball's trajectory may be difficult to predict without repeated simulations and trial and error.  
{{#if analogical}}Problem: You want the horizontal distance of the {{target_bounce}} bounce of the ball to be within 3m of {{target}}m, this is THE MOST IMPORTANT part of the task. What height and horizontal velocity values should you use? \
Be VERY careful when deciding if a bounce position is within 3m of the target, for a target of {{target}}m the {{target_bounce}} bounce must occur between {{t1}}m and {{t2}}m in order to be 3m from {{target}}m. 
Relevant problems: Recall three relevant and distinct problems. For each problem, describe it and explain the solution.
1. {{gen 'rel_prob1' max_tokens=128 do_sample=True temperature=0.4}}
2. {{gen 'rel_prob2' max_tokens=128 do_sample=True temperature=0.4}}
3. {{gen 'rel_prob3' max_tokens=128 do_sample=True temperature=0.4}}
Solve the initial problem: {{else}}{{examples}}
### REAL CASE:
You want the horizontal distance of the {{target_bounce}} bounce of the ball to be within 3m of {{target}}m, this is THE MOST IMPORTANT part of the task. What height and horizontal velocity values should you use? \
Be VERY careful when deciding if a bounce position is within 3m of the target, for a target of {{target}}m the {{target_bounce}} bounce must occur between {{t1}}m and {{t2}}m in order to be 3m from {{target}}m.  
{{/if}} 
Give your reasoning for the task in a single paragraph.
REASONING:
{{gen 'reasoning' max_tokens=128 do_sample=True temperature=0.4}}
Therefore, the height and horizontal velocity values should be:
```json
{
    "height": {{gen 'height' pattern='\d*(\.\d(3))?\,' stop=',' do_sample=False}},
    "horizontal_velocity": {{gen 'horizontal_velocity' pattern='\d*(\.\d(3))?\,' stop=',' do_sample=False}},
}```
SIMULATION:
After running the simulation, the bounces occurred at:
{{get_bounces height horizontal_velocity}}
Give a critique of the reasoning and simulation results, in a single paragraph. Be sure to think on why the simulation results may differ from what you expected, and what you can learn from them.
CRITIQUE:
{{gen 'critique' max_tokens=128 do_sample=True temperature=0.4}}
Do the simulation results match the target? (Yes/No)
Answer:{{gen "answer" max_tokens=8 do_sample=False}}""", caching=False)   
            
            n_reasoning = guidance("""{{context}}
Therefore, the height and horizontal velocity values must be adjusted, the new values are:
```json\
{
    "height": {{gen 'height' pattern='\d*(\.\d(3))?\,' stop=',' do_sample=False}},
    "horizontal_velocity": {{gen 'horizontal_velocity' pattern='\d*(\.\d(3))?\,' stop=',' do_sample=False}},
}```
SIMULATION:
After running the simulation, the bounces occurred at:
{{get_bounces height horizontal_velocity}}
Give a critique of the reasoning and simulation results, in a single paragraph. Be sure to think on why the simulation results may differ from what you expected, and what you can learn from them.
CRITIQUE:
{{gen 'critique' max_tokens=128 do_sample=True temperature=0.4}}
Do the simulation results match the target? (Yes/No)
Answer:{{gen "answer" max_tokens=8 do_sample=False}}""", caching=False)
                                
            nn_reasoning = guidance("""{{context}}
FINAL ANSWER:
After verifying the reasoning with the simulation results, the final answer is:
```json
{
    "height": {{gen 'height_final' pattern='\d*(\.\d(3))?\,' stop=',' do_sample=False}},
    "horizontal_velocity": {{gen 'horizontal_velocity_final' pattern='\d*(\.\d(3))?\,' stop=',' do_sample=False}},
}```""", caching=False)

            t0 = time.time()

            response = n0_reasoning(examples=examples, get_bounces=SIM_STEP.get_bounces_as_string, surface_description=self.surface_descriptions[i], target_bounce=self.TARGET_BOUNCE[1], target=target, t1=round(target-3,2), t2=round(target+3,2), analogical=self.ANALOGICAL)

            try:
                ### REPEAT REASONING UNTIL TARGET/LIMIT IS REACHED
                C = 0
    
                while "No" in response["answer"]:
                    C += 1
                    response = n_reasoning(context=response, get_bounces=SIM_STEP.get_bounces_as_string)
                    if C >= self.MAX_ATTEMPTS:
                        break
                ### REPEAT REASONING UNTIL TARGET/LIMIT IS REACHED

                response = nn_reasoning(context=response)
                ic_final = {
                    "height": float(response["height_final"]),
                    "horizontal_velocity": float(response["horizontal_velocity_final"]),
                }
                bounces = SIM_STEP.get_bounces(ic_final["height"], ic_final["horizontal_velocity"])
            except Exception as e:
                print("ERROR: ", e)
                print("RESPONSE:", str(response))
                guidance_setup.reload_model()
                continue

            t = time.time() - t0    
            results.append({"ic":ic_final, "bounces": bounces, "response":str(response), "attempts":C, "time":t})

        return results

### UNEVEN SURFACE - Frequency Variation ###
class Experiment3(Experiment2):
    def __init__(self, target_bounce, analogical, varying_params):
        super().__init__(target_bounce, analogical, varying_params)
        self.sims = [UnevenSimulation(F=f) for f in varying_params]

        self.surface_descriptions = [f"A simple sine wave with an amplitude of 1 and frequency {i}." for i in varying_params]

### UNEVEN SURFACE - Amplitude Variation ###
class Experiment4(Experiment2):
    def __init__(self, target_bounce, analogical, varying_params):
        super().__init__(target_bounce, analogical, varying_params)
        self.sims = [UnevenSimulation(A=a) for a in varying_params]

        self.surface_descriptions = [f"A simple sine wave with a frequency of 1 and amplitude {i}." for i in varying_params]

### UNEVEN SURFACE - Coefficient Variation ###
class Experiment5(Experiment2):
    def __init__(self, target_bounce, analogical, varying_params):
        super().__init__(target_bounce, analogical, varying_params)
        self.sims = [UnevenSimulation(C_R=c) for c in varying_params]
        self.surface_descriptions = ["A simple sine wave with a frequency of 1 and amplitude of 1."]*len(varying_params)

### Uneven Surface - Max Attempts Variation ###
class Experiment6(Experiment2):
    def __init__(self, target_bounce, analogical, varying_params):
        super().__init__(target_bounce, analogical, varying_params)
        self.MAX_ATTEMPTS = varying_params
        self.surface_descriptions = ["A simple sine wave with a frequency of 1 and amplitude of 1."]*len(varying_params)

    def reasoning_api(self, target, examples):

        results = []
        SIM_STEP = self.sims[0]
        for i, MAX_ATTEMPTS in tqdm(enumerate(self.MAX_ATTEMPTS)):

            ### n0_reasoning
            if self.ANALOGICAL:
                n0_reasoning = guidance("""
{{#system~}}
You're an expert physicist running an experiment. You have a ball and want to \
predict initial height and horizontal velocity values such that the {{target_bounce}} bounce of the ball lands at a specific horizontal position. \
Once you predict the initial conditions a computer simulation will give you the exact positions of the ball's bounces. Below is a description of the surface the ball will bounce on: 
SURFACE:
{{surface_description}}
The surface the ball bounces on is uneven so the ball's trajectory may be difficult to predict without repeated simulations and trial and error. 
Problem: You want the horizontal distance of the {{target_bounce}} bounce of the ball to be within 3m of {{target}}m, this is THE MOST IMPORTANT part of the task. What height and horizontal velocity values should you use? \
Be VERY careful when deciding if a bounce position is within 3m of the target, for a target of {{target}}m the {{target_bounce}} bounce must occur between {{t1}}m and {{t2}}m in order to be 3m from {{target}}m. 
Relevant problems: Recall three relevant and distinct problems. For each problem, describe it and explain the solution.
{{~/system}}
{{#assistant~}}
{{gen 'rel_probs' max_tokens=768 temperature=0.9}}
{{~/assistant}}
{{#user~}}
Solve the initial problem:
Give your reasoning for the task in a single paragraph.
REASONING:
{{~/user}}
{{#assistant~}}
{{gen 'reasoning' temperature=0.4 max_tokens=128}}
{{~/assistant}}
{{#user~}}
Therefore, make a prediction for what the height and horizontal velocity values should be and output below as a JSON object in this format:
{
    "height": h,
    "horizontal_velocity": v
}    
Output nothing else, just the JSON object.
{{~/user}}
{{#assistant~}}
{{gen 'IC' temperature=0.7 max_tokens=64}}
{{~/assistant}}
{{#user~}}
SIMULATION:
After running the simulation, the three bounces occurred at:
{{get_bounces IC}}
Give a critique of the reasoning and simulation results, in a single paragraph. Be sure to think on why the simulation results may differ from what you expected, and what you can learn from them.
CRITIQUE:
{{~/user}}
{{#assistant~}}
{{gen 'critique' temperature=0.4 max_tokens=128}}
{{~/assistant}}
{{#user~}}
Do the simulation results match the target? Answer either Yes or No.
Answer:
{{~/user}}
{{#assistant~}}
{{gen "answer" temperature=0.7 max_tokens=8}}
{{~/assistant}}
""", caching=False) 
            else:
                n0_reasoning = guidance("""
{{#system~}}
You're an expert physicist running an experiment. You have a ball and want to \
predict initial height and horizontal velocity values such that the {{target_bounce}} bounce of the ball lands at a specific horizontal position. \
Once you predict the initial conditions a computer simulation will give you the exact positions of the ball's bounces. Below is a description of the surface the ball will bounce on: 
SURFACE:
{{surface_description}} 
The surface the ball bounces on is uneven so the ball's trajectory may be difficult to predict without repeated simulations and trial and error. 
{{examples}}
### REAL CASE:
You want the horizontal distance of the {{target_bounce}} bounce of the ball to be within 3m of {{target}}m, this is THE MOST IMPORTANT part of the task. What height and horizontal velocity values should you use? \
Be VERY careful when deciding if a bounce position is within 3m of the target, for a target of {{target}}m the {{target_bounce}} bounce must occur between {{t1}}m and {{t2}}m in order to be 3m from {{target}}m.  
{{~/system}}
{{#user~}}
Give your reasoning for the task in a single paragraph.
REASONING:
{{~/user}}
{{#assistant~}}
{{gen 'reasoning' temperature=0.4 max_tokens=128}}
{{~/assistant}}
{{#user~}}
Therefore, make a prediction for what the height and horizontal velocity values should be and output below as a JSON object in this format:
{
    "height": h,
    "horizontal_velocity": v
}    
Output nothing else, just the JSON object.
{{~/user}}
{{#assistant~}}
{{gen 'IC' temperature=0.7 max_tokens=64}}
{{~/assistant}}
{{#user~}}
SIMULATION:
After running the simulation, the three bounces occurred at:
{{get_bounces IC}}
Give a critique of the reasoning and simulation results, in a single paragraph. Be sure to think on why the simulation results may differ from what you expected, and what you can learn from them.
CRITIQUE:
{{~/user}}
{{#assistant~}}
{{gen 'critique' temperature=0.4 max_tokens=128}}
{{~/assistant}}
{{#user~}}
Do the simulation results match the target? Answer either Yes or No.
Answer:
{{~/user}}
{{#assistant~}}
{{gen "answer" temperature=0.7 max_tokens=8}}
{{~/assistant}}
    """, caching=False) 

            ### n_reasoning
            n_reasoning = guidance("""
{{context}}
{{#user~}}
Therefore, make a new prediction for what the updated height and horizontal velocity values should be and output below as a JSON object in this format:
{
    "height": h,
    "horizontal_velocity": v
}    
Output nothing else, just the JSON object.
{{~/user}}
{{#assistant~}}
{{gen 'IC' temperature=0.7 max_tokens=64}}
{{~/assistant}} 
{{#user~}}
SIMULATION:
After running the simulation, the three bounces occurred at:
{{get_bounces IC}}
Give a critique of the reasoning and simulation results, in a single paragraph.
CRITIQUE:
{{~/user}}
{{#assistant~}}
{{gen 'critique' temperature=0.4 max_tokens=128}}
{{~/assistant}}
{{#user~}}
Do the simulation results match the target? Answer either Yes or No.
Answer:
{{~/user}}
{{#assistant~}}
{{gen "answer" temperature=0.7 max_tokens=8}}
{{~/assistant}}""", caching=False)

            ### nn_reasoning
            nn_reasoning = guidance("""
{{context}}                  
{{#user~}}
FINAL ANSWER:
After verifying the reasoning with the simulation results, output the final answer below as a JSON object in this format:
{
    "height": h,
    "horizontal_velocity": v
}
Output nothing else, just the JSON object.
{{~/user}}
{{#assistant~}}
{{gen 'final_IC' temperature=0.7 max_tokens=64}}
{{~/assistant}}""", caching=False)

            t0 = time.time()

            response = n0_reasoning(examples=examples, get_bounces=SIM_STEP.get_bounces_as_string_IC, surface_description=self.surface_descriptions[i], target_bounce=self.TARGET_BOUNCE[1], target=target, t1=round(target-3,2), t2=round(target+3,2), analogical=self.ANALOGICAL)
            try:
                ### REPEAT REASONING UNTIL TARGET/LIMIT IS REACHED
                C = 0
    
                while "No" in response["answer"]:
                    C += 1
                    response = n_reasoning(context=str(response), get_bounces=SIM_STEP.get_bounces_as_string_IC)
                    if C >= MAX_ATTEMPTS:
                        break
                    ### Sleep to avoid API limit
                    #time.sleep(5)
                ### REPEAT REASONING UNTIL TARGET/LIMIT IS REACHED
                response = nn_reasoning(context=str(response))
                ic_final = response["final_IC"]
                ic_final = ic_final[ic_final.index("{") : ic_final.rindex("}") + 1]
                ic_final = json.loads(ic_final)
                bounces = SIM_STEP.get_bounces(ic_final["height"], ic_final["horizontal_velocity"])
            except Exception as e:
                print(f"ERROR: exception: {e}")
                guidance_setup.reload_model()
                continue

            t = time.time() - t0    
            results.append({"ic":ic_final, "bounces": bounces, "response":str(response), "attempts":C, "time":t})

        return results

    def reasoning_local(self, target, examples):
        """
        Perform a guided simulation, where the LLM reasons about the query.
        Q -> (QRPCA)Q -> (QRPCA)QR -> (QRPCA)QRPC -> (QRPCA)QRPCA

        1) The LLM attempts an initial reasoning step R
        2) The LLM decided if a simulation is needed and new set of IC is generated
        3) Simulation results are retrieved 
        4) The LLM critiques its reasoning based on the results
        5) Repeat 2-4 until the target is reached
        6) The LLM outputs a final set of IC
        
        Parameters:
        - sim: The simulation to use.
        - target: The target distance.

        Returns:
        - ic_final: The final set of initial conditions.
        """ 
        results = []
        SIM_STEP = self.sims[0]
        for i, MAX_ATTEMPTS in tqdm(enumerate(self.MAX_ATTEMPTS)):

            n0_reasoning = guidance("""You're an expert physicist running an experiment. You have a ball and want to \
predict initial height and horizontal velocity values such that the {{target_bounce}} bounce of the ball lands at a specific horizontal position. \
Once you predict the initial conditions a computer simulation will give you the exact positions of the ball's bounces. Below is a description of the surface the ball will bounce on: 
SURFACE:
{{surface_description}} 
The surface the ball bounces on is uneven so the ball's trajectory may be difficult to predict without repeated simulations and trial and error.  
{{#if analogical}}Problem: You want the horizontal distance of the {{target_bounce}} bounce of the ball to be within 3m of {{target}}m, this is THE MOST IMPORTANT part of the task. What height and horizontal velocity values should you use? \
Be VERY careful when deciding if a bounce position is within 3m of the target, for a target of {{target}}m the {{target_bounce}} bounce must occur between {{t1}}m and {{t2}}m in order to be 3m from {{target}}m. 
Relevant problems: Recall three relevant and distinct problems. For each problem, describe it and explain the solution.
1. {{gen 'rel_prob1' max_tokens=128 do_sample=True temperature=0.4}}
2. {{gen 'rel_prob2' max_tokens=128 do_sample=True temperature=0.4}}
3. {{gen 'rel_prob3' max_tokens=128 do_sample=True temperature=0.4}}
Solve the initial problem: {{else}}{{examples}}
### REAL CASE:
You want the horizontal distance of the {{target_bounce}} bounce of the ball to be within 3m of {{target}}m, this is THE MOST IMPORTANT part of the task. What height and horizontal velocity values should you use? \
Be VERY careful when deciding if a bounce position is within 3m of the target, for a target of {{target}}m the {{target_bounce}} bounce must occur between {{t1}}m and {{t2}}m in order to be 3m from {{target}}m.  
{{/if}} 
Give your reasoning for the task in a single paragraph.
REASONING:
{{gen 'reasoning' max_tokens=128 do_sample=True temperature=0.4}}
Therefore, the height and horizontal velocity values should be:
```json
{
    "height": {{gen 'height' pattern='\d*(\.\d(3))?\,' stop=',' do_sample=False}},
    "horizontal_velocity": {{gen 'horizontal_velocity' pattern='\d*(\.\d(3))?\,' stop=',' do_sample=False}},
}```
SIMULATION:
After running the simulation, the bounces occurred at:
{{get_bounces height horizontal_velocity}}
Give a critique of the reasoning and simulation results, in a single paragraph. Be sure to think on why the simulation results may differ from what you expected, and what you can learn from them.
CRITIQUE:
{{gen 'critique' max_tokens=128 do_sample=True temperature=0.4}}
Do the simulation results match the target? (Yes/No)
Answer:{{gen "answer" max_tokens=8 do_sample=False}}""", caching=False)   
            
            n_reasoning = guidance("""{{context}}
Therefore, the height and horizontal velocity values must be adjusted, the new values are:
```json\
{
    "height": {{gen 'height' pattern='\d*(\.\d(3))?\,' stop=',' do_sample=False}},
    "horizontal_velocity": {{gen 'horizontal_velocity' pattern='\d*(\.\d(3))?\,' stop=',' do_sample=False}},
}```
SIMULATION:
After running the simulation, the bounces occurred at:
{{get_bounces height horizontal_velocity}}
CRITIQUE:
{{gen 'critique' max_tokens=128 do_sample=True temperature=0.4}}
Do the simulation results match the target? (Yes/No)
Answer:{{gen "answer" max_tokens=8 do_sample=False}}""", caching=False)
                                
            nn_reasoning = guidance("""{{context}}
FINAL ANSWER:
After verifying the reasoning with the simulation results, the final answer is:
```json
{
    "height": {{gen 'height_final' pattern='\d*(\.\d(3))?\,' stop=',' do_sample=False}},
    "horizontal_velocity": {{gen 'horizontal_velocity_final' pattern='\d*(\.\d(3))?\,' stop=',' do_sample=False}},
}```""", caching=False)

            t0 = time.time()

            response = n0_reasoning(examples=examples, get_bounces=SIM_STEP.get_bounces_as_string, surface_description=self.surface_descriptions[i], target_bounce=self.TARGET_BOUNCE[1], target=target, t1=round(target-3,2), t2=round(target+3,2), analogical=self.ANALOGICAL)

            try:
                ### REPEAT REASONING UNTIL TARGET/LIMIT IS REACHED
                C = 0
    
                while "No" in response["answer"]:
                    C += 1
                    response = n_reasoning(context=response, get_bounces=SIM_STEP.get_bounces_as_string)
                    if C >= MAX_ATTEMPTS:
                        break
                ### REPEAT REASONING UNTIL TARGET/LIMIT IS REACHED

                response = nn_reasoning(context=response)
                ic_final = {
                    "height": float(response["height_final"]),
                    "horizontal_velocity": float(response["horizontal_velocity_final"]),
                }
                bounces = SIM_STEP.get_bounces(ic_final["height"], ic_final["horizontal_velocity"])
            except Exception as e:
                print("ERROR: ", e)
                print("RESPONSE:", str(response))
                guidance_setup.reload_model()
                continue

            t = time.time() - t0    
            results.append({"ic":ic_final, "bounces": bounces, "response":str(response), "attempts":C, "time":t})

        return results

### Custom Surface ###
class Experiment7(Experiment2):
    def __init__(self, target_bounce, analogical, varying_params):
        super().__init__(target_bounce, analogical, None)
        # CUSTOM_SURFACE = lambda x: x + np.sin(x)
        # self.sims = [CustomSimulation(CUSTOM_SURFACE)]
        # self.surface_descriptions = "A diagonal curve, defined by y=x, with some periodic variation."

        CUSTOM_SURFACE = lambda x: 0 if x < 10 else 10 if x < 20 else 0
        self.sims = [CustomSimulation(CUSTOM_SURFACE)]
        self.surface_descriptions = ["A flat surface with a step from x=10 to x=20 to y=10."]

### Baseline - Flat Surface ###
class Experiment8(Experiment1):
    def __init__(self, target_bounce, analogical, varying_params):
        super().__init__(target_bounce, analogical, varying_params)

    def reasoning_api(self, target, examples):
        n0_reasoning = guidance("""
{{#system~}}
You're an expert physicist running an experiment. You have a ball and want to \
predict initial height and horizontal velocity values such that the {{target_bounce}} bounce of the ball lands at a specific horizontal position. \
Below is a description of the surface the ball will bounce on: 
SURFACE:
{{surface_description}} 
{{examples}}
### REAL CASE:
You want the horizontal distance of the {{target_bounce}} bounce of the ball to be within 3m of {{target}}m, this is THE MOST IMPORTANT part of the task. What height and horizontal velocity values should you use? \
Be VERY careful when deciding if a bounce position is within 3m of the target, for a target of {{target}}m the {{target_bounce}} bounce must occur between {{t1}}m and {{t2}}m in order to be 3m from {{target}}m.  
{{~/system}}
{{#user~}}
Give your reasoning for the task in a single paragraph.
REASONING:
{{~/user}} 
{{#assistant~}}
{{gen 'reasoning' temperature=0.4 max_tokens=128}}
{{~/assistant}}
{{#user~}}
Therefore, make a prediction for what the height and horizontal velocity values should be and output below as a JSON object in this format:
{
    "height": h,
    "horizontal_velocity": v
}    
Output nothing else, just the JSON object.
{{~/user}}
{{#assistant~}}
{{gen 'IC' temperature=0.7 max_tokens=64}}
{{~/assistant}}
""", caching=False) 

        t0 = time.time()

        response = n0_reasoning(examples=examples, surface_description=self.surface_descriptions[0], target_bounce=self.TARGET_BOUNCE[1], target=target, t1=round(target-3,2), t2=round(target+3,2))
        try:
            ic_final = response["IC"]
            ic_final = ic_final[ic_final.index("{") : ic_final.rindex("}") + 1]
            ic_final = json.loads(ic_final)
            bounces = self.sim.get_bounces(ic_final["height"], ic_final["horizontal_velocity"])
        except Exception as e:
            print(f"ERROR: exception: {e}, for reponse {response}")
            guidance_setup.reload_model()
            return [{"ic":{"height":0, "horizontal_velocity":0}, "bounces": [0,0,0], "response":str(response), "attempts":0, "time":0}]

        t = time.time() - t0    
        return [{"ic":ic_final, "bounces": bounces, "response":str(response), "attempts":1, "time":t}]

    def reasoning_local(self, target, examples):
        n0_reasoning = guidance("""You're an expert physicist running an experiment. You have a ball and want to \
predict initial height and horizontal velocity values such that the {{target_bounce}} bounce of the ball lands at a specific horizontal position. \
Below is a description of the surface the ball will bounce on: 
SURFACE:
{{surface_description}} 
{{examples}}
### REAL CASE:
You want the horizontal distance of the {{target_bounce}} bounce of the ball to be within 3m of {{target}}m, this is THE MOST IMPORTANT part of the task. What height and horizontal velocity values should you use? \
Be VERY careful when deciding if a bounce position is within 3m of the target, for a target of {{target}}m the {{target_bounce}} bounce must occur between {{t1}}m and {{t2}}m in order to be 3m from {{target}}m.  
Give your reasoning for the task in a single paragraph.
REASONING:
{{gen 'reasoning' max_tokens=128 do_sample=True temperature=0.4}}
Therefore, the height and horizontal velocity values should be:
```json
{
    "height": {{gen 'height' pattern='\d*(\.\d(3))?\,' stop=',' do_sample=False}},
    "horizontal_velocity": {{gen 'horizontal_velocity' pattern='\d*(\.\d(3))?\,' stop=',' do_sample=False}},
}```
""", caching=False)   
        t0 = time.time()

        response = n0_reasoning(examples=examples, surface_description=self.surface_descriptions[0], target_bounce=self.TARGET_BOUNCE[1], target=target, t1=round(target-3,2), t2=round(target+3,2))
        try:
            ic_final = {
                "height": float(response["height"]),
                "horizontal_velocity": float(response["horizontal_velocity"]),
            }
            bounces = self.sim.get_bounces(ic_final["height"], ic_final["horizontal_velocity"])
        except Exception as e:
            print(f"ERROR: exception: {e}, for reponse {response}")
            guidance_setup.reload_model()
            return [{"ic":{"height":0, "horizontal_velocity":0}, "bounces": [0,0,0], "response":str(response), "attempts":0, "time":0}]

        t = time.time() - t0    
        return [{"ic":ic_final, "bounces": bounces, "response":str(response), "attempts":1, "time":t}]

### Baseline - Curved Surface ###
class Experiment9(Experiment8):
    def __init__(self, target_bounce, analogical, varying_params):
        super().__init__(target_bounce, analogical, varying_params)
        
        self.sim = UnevenSimulation()
        self.surface_descriptions = "A simple sine wave with a frequency of 1 and amplitude of 1."

### Difficulty Increment ###
class Experiment10(Experiment2):
    def __init__(self, target_bounce, analogical, varying_params):
        super().__init__(target_bounce, analogical, None)

        CUSTOM_SURFACES = [sample_surface(difficulty=i) for i in varying_params]
        self.sims = [CustomSimulation(CS) for CS in CUSTOM_SURFACES]
        self.surface_descriptions = [f"A randomly generated rough surface of difficulty {i}/10" for i in varying_params]

### Baseline Difficulty Increment ###
class Experiment11(Experiment2):
    def __init__(self, target_bounce, analogical, varying_params):
        super().__init__(target_bounce, analogical, None)

        CUSTOM_SURFACES = [sample_surface(difficulty=i) for i in varying_params]
        self.sims = [CustomSimulation(CS) for CS in CUSTOM_SURFACES]
        self.surface_descriptions = [f"A randomly generated rough surface of difficulty {i}/10" for i in varying_params]

    def reasoning_api(self, target, examples):
        results = []
        for i, SIM_STEP in tqdm(enumerate(self.sims)):

            n0_reasoning = guidance("""
    {{#system~}}
    You're an expert physicist running an experiment. You have a ball and want to \
    predict initial height and horizontal velocity values such that the {{target_bounce}} bounce of the ball lands at a specific horizontal position. \
    Below is a description of the surface the ball will bounce on: 
    SURFACE:
    {{surface_description}} 
    {{examples}}
    ### REAL CASE:
    You want the horizontal distance of the {{target_bounce}} bounce of the ball to be within 3m of {{target}}m, this is THE MOST IMPORTANT part of the task. What height and horizontal velocity values should you use? \
    Be VERY careful when deciding if a bounce position is within 3m of the target, for a target of {{target}}m the {{target_bounce}} bounce must occur between {{t1}}m and {{t2}}m in order to be 3m from {{target}}m.  
    {{~/system}}
    {{#user~}}
    Give your reasoning for the task in a single paragraph.
    REASONING:
    {{~/user}} 
    {{#assistant~}}
    {{gen 'reasoning' temperature=0.4 max_tokens=128}}
    {{~/assistant}}
    {{#user~}}
    Therefore, make a prediction for what the height and horizontal velocity values should be and output below as a JSON object in this format:
    {
        "height": h,
        "horizontal_velocity": v
    }    
    Output nothing else, just the JSON object.
    {{~/user}}
    {{#assistant~}}
    {{gen 'IC' temperature=0.7 max_tokens=64}}
    {{~/assistant}}
    """, caching=False) 

            t0 = time.time()

            response = n0_reasoning(examples=examples, surface_description=self.surface_descriptions[i], target_bounce=self.TARGET_BOUNCE[1], target=target, t1=round(target-3,2), t2=round(target+3,2))
            try:
                ic_final = response["IC"]
                ic_final = ic_final[ic_final.index("{") : ic_final.rindex("}") + 1]
                ic_final = json.loads(ic_final)
                bounces = SIM_STEP.get_bounces(ic_final["height"], ic_final["horizontal_velocity"])
            except Exception as e:
                print(f"ERROR: exception: {e}, for reponse {response}")
                guidance_setup.reload_model()
                continue

            t = time.time() - t0    
            results.append({"ic":ic_final, "bounces": bounces, "response":str(response), "attempts":1, "time":t})
        return results

    def reasoning_local(self, target, examples):
        results = []
        for i, SIM_STEP in tqdm(enumerate(self.sims)):
            n0_reasoning = guidance("""You're an expert physicist running an experiment. You have a ball and want to \
predict initial height and horizontal velocity values such that the {{target_bounce}} bounce of the ball lands at a specific horizontal position. \
Below is a description of the surface the ball will bounce on: 
SURFACE:
{{surface_description}} 
{{examples}}
### REAL CASE:
You want the horizontal distance of the {{target_bounce}} bounce of the ball to be within 3m of {{target}}m, this is THE MOST IMPORTANT part of the task. What height and horizontal velocity values should you use? \
Be VERY careful when deciding if a bounce position is within 3m of the target, for a target of {{target}}m the {{target_bounce}} bounce must occur between {{t1}}m and {{t2}}m in order to be 3m from {{target}}m.  
Give your reasoning for the task in a single paragraph.
REASONING:
{{gen 'reasoning' max_tokens=128 do_sample=True temperature=0.4}}
Therefore, the height and horizontal velocity values should be:
```json
{
    "height": {{gen 'height' pattern='\d*(\.\d(3))?\,' stop=',' do_sample=False}},
    "horizontal_velocity": {{gen 'horizontal_velocity' pattern='\d*(\.\d(3))?\,' stop=',' do_sample=False}},
}```
""", caching=False)   
            t0 = time.time()

            response = n0_reasoning(examples=examples, surface_description=self.surface_descriptions[i], target_bounce=self.TARGET_BOUNCE[1], target=target, t1=round(target-3,2), t2=round(target+3,2))
            try:
                ic_final = {
                    "height": float(response["height"]),
                    "horizontal_velocity": float(response["horizontal_velocity"]),
                }
                bounces = SIM_STEP.get_bounces(ic_final["height"], ic_final["horizontal_velocity"])
            except Exception as e:
                print(f"ERROR: exception: {e}, for reponse {response}")
                guidance_setup.reload_model()
                continue

            t = time.time() - t0    
            results.append({"ic":ic_final, "bounces": bounces, "response":str(response), "attempts":1, "time":t})
        return results


EXPERIMENTS = {
    "flat": Experiment1,
    "uneven": Experiment2,
    "frequency": Experiment3,
    "amplitude": Experiment4,
    "coefficient": Experiment5,
    "max_attempts": Experiment6,
    "custom": Experiment7,
    "baseline": Experiment8,
    "baseline_curved": Experiment9,
    "difficulty_increment": Experiment10,
    "baseline_difficulty_increment": Experiment11,
    "forward": ExperimentForward,
}

class Experiment():
    def __init__(self, EXP_NUM, MODEL_NAME, TARGET_BOUNCE, ANALOGICAL, VARYING_PARAMS, EXAMPLE_GEN):
        self.EXP_NUM = EXP_NUM
        self.TARGET_BOUNCE = (TARGET_BOUNCE, ["first", "second", "third", "fourth", "fifth"][TARGET_BOUNCE])  

        # Choose experiment
        assert EXP_NUM in range(1, len(EXPERIMENTS.keys())+1), "Invalid experiment number."
        self.EXP = EXPERIMENTS[list(EXPERIMENTS.keys())[EXP_NUM-1]](self.TARGET_BOUNCE, ANALOGICAL, VARYING_PARAMS)
        
        self.MODEL_NAME = MODEL_NAME
        self.API = any([api in MODEL_NAME for api in ["gpt", "chat-bison"]])
        self.EXAMPLE_GEN = EXAMPLE_GEN

        ### Gen Examples
        if self.EXAMPLE_GEN=="gen":
            self.examples = json.load(open("examples/summarised_example_reasoning.json", "r"))
        else:
            ### Fixed Examples
            self.fixed_examples = json.load(open("examples/fixed_examples.json", "r"))
            if self.EXP_NUM==1:
                self.fixed_examples = self.fixed_examples["flat"]
            elif self.EXP_NUM==7:
                self.fixed_examples = self.fixed_examples["custom"]
            else:
                self.fixed_examples = self.fixed_examples["curved"]
        
    def reasoning_random(self, target, SIM):

        N1 = 100 # Total trials
        N2 = 10    # Number of initial conditions to generate per trial
        
        average_best_ic = {
            "height": 0,
            "horizontal_velocity": 0,
        }

        # Generate N1 random trials
        for _ in tqdm(range(N1), desc="Running Random Simulations"):
            ics = []
            # Generate N2 random initial conditions
            for _ in range(N2):
                ### TODO: Think more on the ranges of the initial conditions
                ics.append({
                    "height": np.random.uniform(0, target),
                    "horizontal_velocity": np.random.uniform(0, target),
                })

            # Choose the best initial condition
            bounces = [SIM.get_bounces(ic["height"], ic["horizontal_velocity"]) for ic in ics]
            scores = [abs(bounces[i][self.TARGET_BOUNCE]-target) for i in range(N2)]
            ic = ics[np.argmin(scores)]

            # Update the average best initial condition
            average_best_ic["height"] += ic["height"] / N1
            average_best_ic["horizontal_velocity"] += ic["horizontal_velocity"] / N1
        
        return average_best_ic, "random", 0, 0

    def gen_examples(self, num_examples):
        if num_examples==0:
            return ""
        EXP_STRING = f"experiment_{self.EXP_NUM}"
        assert self.MODEL_NAME in self.examples.keys(), f"Invalid model for examples: {self.MODEL_NAME}"
        assert EXP_STRING in self.examples[self.MODEL_NAME].keys(), f"Invalid experiment number: {self.EXP_NUM}, for model: {self.MODEL_NAME}"
        assert len(self.examples[self.MODEL_NAME][EXP_STRING]) >= num_examples, f"Not enough stored examples for target examples: {num_examples}"
        examples = list(self.examples[self.MODEL_NAME][EXP_STRING])
        random.shuffle(examples)

        # Perform stratified sampling for bins of 10

        # Convert list of dictionaries to a NumPy array
        data = np.array([[item['target'], item["example"]] for item in examples])  

        # Define bins
        bins = np.arange(0, 101, 10)  # Bins from 0 to 100 in steps of 10
        bin_indices = np.digitize(data[:, 0], bins)  # Assuming 'target' is in the first column

        # Number of samples per bin 
        n_samples_per_bin = 1

        # Sample from each bin
        stratified_samples = []
        for bin_index in range(1, len(bins)):  # bin indices start from 1
            bin_data = data[bin_indices == bin_index]
            if len(bin_data) < n_samples_per_bin:
                continue
            sampled_indices = np.random.choice(bin_data.shape[0], n_samples_per_bin, replace=False)
            stratified_samples.append(bin_data[sampled_indices][0][1])

        final_examples = np.random.choice(stratified_samples, num_examples, replace=False)
        for i in range(num_examples):
            final_examples[i] = f"### EXAMPLE {i+1}:\n" + final_examples[i]
        return "\n".join(final_examples)
            
    def reasoning(self, target, num_examples):
        examples = self.gen_examples(num_examples)

        if self.API:
            return self.EXP.reasoning_api(target, examples)
        elif self.MODEL_NAME == "random":
            return self.reasoning_random(target, self.EXP.sim)
        else:
            return self.EXP.reasoning_local(target, examples)
