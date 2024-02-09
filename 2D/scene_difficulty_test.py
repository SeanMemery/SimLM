import math
from sim import Simulation
import numpy as np
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import json

from sim import CustomSimulation
LATEX_FIG_SCALE = 0.48

functions = {
    "random_sines_1": lambda x: 0.3*math.sin(x/10) + 0.3*math.sin(x/5) + 0.3*math.sin(x/2) + 0.3*math.sin(x),
    "random_sines_2": lambda x: 0.3*math.sin(x/10) + 0.3*math.sin(x/5) + 0.3*math.sin(x/2) + 0.3*math.sin(x) + 0.3*math.sin(x/20),
    "random_sines_3": lambda x: 0.3*math.sin(x/10) + 0.3*math.sin(x/5) + 0.3*math.sin(x/2) + 0.3*math.sin(x) + 0.3*math.sin(x/20) + 0.3*math.sin(x/40),
    "random_sines_4": lambda x: 0.3*math.sin(x/10) + 0.3*math.sin(x/5) + 0.3*math.sin(x/2) + 0.3*math.sin(x) + 0.3*math.sin(x/20) + 0.3*math.sin(x/40) + 0.3*math.sin(x/80),
    "spike": lambda x: 0 if x < 10 else 20 if x < 20 else 0,
    "diagonal": lambda x: x,
    "diagonal_curved": lambda x: x + 0.75*math.sin(x/10),
}


def plot_surface(function, N, TARGET, H_LIM=[1,100], V_LIM=[1,100], SHOW=True, SAVE=False, NAME=""):
    RESULTS = []
    sim = CustomSimulation(function)

    if SHOW:
        sim.visualise()
    
    # # Grid of random initial condition values to test
    # ICS = [{
    #     "height": np.random.uniform(H_LIM[0], H_LIM[1]),
    #     "horizontal_velocity": np.random.uniform(V_LIM[0], V_LIM[1])
    # } for _ in range(N)]

    # Add a single IC for each HxV combination
    ICS = []
    for h in np.linspace(H_LIM[0], H_LIM[1], N):
        for v in np.linspace(V_LIM[0], V_LIM[1], N):
            ICS.append({
                "height": h,
                "horizontal_velocity": v
            })

    for IC in tqdm(ICS, desc="Running Simulations"):
        bounces = sim.get_bounces(IC["height"],IC["horizontal_velocity"])
        RESULTS.append({
            "score": abs(bounces[-1]-TARGET), 
            "bounces": bounces, 
            "target": TARGET, 
            "IC": IC
        })

    df = pd.DataFrame(RESULTS)

    best = df.iloc[np.argmin(df["score"])]["IC"]

    FINAL_RESULTS = {
        "mean": np.mean(df["score"]),
        "median": np.median(df["score"]),
        "std": np.std(df["score"]),
        "best": best,
        "worst": df.iloc[np.argmax(df["score"])]["IC"],
        "less_than_5": len([s for s in df["score"] if s < 5])/N,
    }

    for res in FINAL_RESULTS:
        print(f"{res}: {FINAL_RESULTS[res]}")

    if SHOW:
        sim.visualise(best["height"], best["horizontal_velocity"])

    # Plot heatmap of scores
    v = df["IC"].apply(lambda x: x["horizontal_velocity"])
    h = df["IC"].apply(lambda x: x["height"])
    s = df["score"]

    # Set a max score of 100 
    s = s.apply(lambda x: min(x, 100))

    from matplotlib import rcParams
    # Set the font family and size to match your LaTeX document
    rcParams['font.family'] = 'serif'
    rcParams['font.size'] = int(24/LATEX_FIG_SCALE)

    fig1, ax1 = plt.subplots(1, 1, figsize=( int(12/(LATEX_FIG_SCALE)), int(12/(LATEX_FIG_SCALE))))

    hb = ax1.hexbin(v, h, C=s, gridsize=100, cmap=plt.colormaps["viridis"].reversed(), vmin=0, vmax=100)

    #plt.title(f"Heatmap of Scores for target of {TARGET}")
    ax1.set_xlabel("Horizontal Velocity")
    ax1.set_ylabel("Height")
    cb = fig1.colorbar(hb)
    cb.set_ticks([20, 40, 60, 80, 100])
    fig1.tight_layout(pad=.5)

    if SAVE:
        if NAME:
            fig1.savefig(f"figures/{NAME}.pdf")
        else:
            fig1.savefig(f"figures/{function.__name__}_N{N}.pdf")
    else:
        fig1.show()

def test_surface(function, N, TARGET):
    RESULTS = []
    sim = CustomSimulation(function)
    
    # Grid of random initial condition values to test
    H_MAX = 25
    V_MAX = 25
    ICS = [{
        "height": np.random.uniform(1, H_MAX), 
        "horizontal_velocity": np.random.uniform(1, V_MAX)
    } for _ in range(N)]

    for IC in ICS:
        bounces = sim.get_bounces(IC["height"],IC["horizontal_velocity"])
        RESULTS.append({
            "score": abs(bounces[-1]-TARGET), 
            "bounces": bounces, 
            "target": TARGET, 
            "IC": IC
        })

    df = pd.DataFrame(RESULTS)

    return {
        "mean": np.mean(df["score"]),
        "median": np.median(df["score"]),
        "std": np.std(df["score"]),
        "best": df.iloc[np.argmin(df["score"])]["IC"],
        "worst": df.iloc[np.argmax(df["score"])]["IC"],
        "less_than_5": len([s for s in df["score"] if s < 5])/N,
    }

if __name__ == "__main__":
    N = 500
    T = 50
    plot_surface(lambda x: 0, N, T, H_LIM=[1,50], V_LIM=[1,50], SHOW=False, SAVE=True, NAME=f"exp1")
    plot_surface(lambda x: np.sin(x), N, T, H_LIM=[1,50], V_LIM=[1,50], SHOW=False, SAVE=True, NAME=f"exp2")