"""
Script to run batch experiments for Questions 3.c, 3.d, and 3.e,
and generate publication-ready plots with Seaborn and Matplotlib.

---

AI GENERATED na krub, Ajaarn.

"""

from __future__ import annotations
import os
import random
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from pacman_module.pacman import runGame
from pacman_module.ghostAgents import ConfusedGhost, AfraidGhost, ScaredGhost
from bayesfilter import BeliefStateAgent
from sherlockpacman import PacmanAgent


GHOST_CLASSES = {
    "confused": ConfusedGhost,
    "afraid": AfraidGhost,
    "scared": ScaredGhost,
}


def run_single_trial(
    layout: str,
    ghost_type: str,
    sensor_variance: float,
    seed: int,
    max_steps: int = 50,
) -> tuple[list[float], list[float]]:
    """Run a single game simulation and return uncertainty and error histories."""
    # Set seeds for reproducibility
    np.random.seed(seed)
    random.seed(seed)

    args = argparse.Namespace(
        seed=seed,
        agentfile="sherlockpacman.py",
        ghostagent=ghost_type,
        layout=layout,
        nghosts=1,
        silentdisplay=True,
        bsagentfile="bayesfilter.py",
        oraclebsagentfile=None,
        edibleghosts=True,
        hiddenghosts=False,
        sensorvariance=sensor_variance,
    )

    agent = PacmanAgent(args)
    bsagent = BeliefStateAgent(args)
    ghost_cls = GHOST_CLASSES[ghost_type]
    ghosts = [ghost_cls(1, args)]

    # Run the game headlessly
    runGame(
        layout,
        agent,
        ghosts,
        bsagent,
        displayGraphics=False,
        expout=0,
        hiddenGhosts=False,
        edibleGhosts=True,
        startingIndex=2,
    )

    # Return logged metrics up to max_steps
    uncertainty = bsagent.history_uncertainty[:max_steps]
    error = bsagent.history_error[:max_steps]
    return uncertainty, error


def run_experiment_3c(num_trials: int = 15, max_steps: int = 40) -> pd.DataFrame:
    """Run simulations for Question 3.c and 3.d on both layouts across all 3 ghosts."""
    print("=" * 60)
    print("Running Experiment 3.c (Layouts & Ghost Types)...")
    print("=" * 60)

    records = []
    layouts = ["large_filter", "large_filter_walls"]
    ghost_types = ["confused", "afraid", "scared"]

    for layout in layouts:
        for ghost in ghost_types:
            print(f"--> Simulating layout='{layout}', ghost='{ghost}' over {num_trials} trials...")
            for seed in range(1, num_trials + 1):
                unc, err = run_single_trial(
                    layout=layout,
                    ghost_type=ghost,
                    sensor_variance=1.0,
                    seed=seed * 17,
                    max_steps=max_steps,
                )
                for t, (u, e) in enumerate(zip(unc, err)):
                    records.append({
                        "step": t,
                        "uncertainty": u,
                        "error": e,
                        "layout": layout,
                        "ghost": ghost,
                        "trial": seed,
                    })

    df = pd.DataFrame(records)
    df.to_csv("experiment_3c_data.csv", index=False)
    print(f"Experiment 3.c data saved to experiment_3c_data.csv ({len(df)} data points).")
    return df


def run_experiment_3e(num_trials: int = 15, max_steps: int = 40) -> pd.DataFrame:
    """Run simulations for Question 3.e testing different sensor variances."""
    print("=" * 60)
    print("Running Experiment 3.e (Sensor Variances)...")
    print("=" * 60)

    records = []
    variances = [0.25, 1.0, 4.0]
    layout = "large_filter"
    ghost = "scared"

    for var in variances:
        print(f"--> Simulating variance={var} on layout='{layout}', ghost='{ghost}'...")
        for seed in range(1, num_trials + 1):
            unc, err = run_single_trial(
                layout=layout,
                ghost_type=ghost,
                sensor_variance=var,
                seed=seed * 23,
                max_steps=max_steps,
            )
            for t, (u, e) in enumerate(zip(unc, err)):
                records.append({
                    "step": t,
                    "uncertainty": u,
                    "error": e,
                    "variance": f"var = {var}",
                    "trial": seed,
                })

    df = pd.DataFrame(records)
    df.to_csv("experiment_3e_data.csv", index=False)
    print(f"Experiment 3.e data saved to experiment_3e_data.csv ({len(df)} data points).")
    return df


def plot_results_3c(df: pd.DataFrame):
    """Generate and save 2x2 comparison plots for Question 3.c with Seaborn."""
    sns.set_theme(style="whitegrid", font_scale=1.1)
    palette = {"confused": "#3498db", "afraid": "#e67e22", "scared": "#e74c3c"}

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True)

    # 1. large_filter - Uncertainty
    sns.lineplot(
        data=df[df["layout"] == "large_filter"],
        x="step",
        y="uncertainty",
        hue="ghost",
        palette=palette,
        errorbar="sd",
        ax=axes[0, 0],
    )
    axes[0, 0].set_title("Open Maze (large_filter): Uncertainty", fontweight="bold")
    axes[0, 0].set_ylabel("Shannon Entropy (bits)")

    # 2. large_filter - Quality / Error
    sns.lineplot(
        data=df[df["layout"] == "large_filter"],
        x="step",
        y="error",
        hue="ghost",
        palette=palette,
        errorbar="sd",
        ax=axes[0, 1],
    )
    axes[0, 1].set_title("Open Maze (large_filter): Quality Error", fontweight="bold")
    axes[0, 1].set_ylabel("Expected Manhattan Error (tiles)")

    # 3. large_filter_walls - Uncertainty
    sns.lineplot(
        data=df[df["layout"] == "large_filter_walls"],
        x="step",
        y="uncertainty",
        hue="ghost",
        palette=palette,
        errorbar="sd",
        ax=axes[1, 0],
    )
    axes[1, 0].set_title("Walled Maze (large_filter_walls): Uncertainty", fontweight="bold")
    axes[1, 0].set_ylabel("Shannon Entropy (bits)")
    axes[1, 0].set_xlabel("Time Step (t)")

    # 4. large_filter_walls - Quality / Error
    sns.lineplot(
        data=df[df["layout"] == "large_filter_walls"],
        x="step",
        y="error",
        hue="ghost",
        palette=palette,
        errorbar="sd",
        ax=axes[1, 1],
    )
    axes[1, 1].set_title("Walled Maze (large_filter_walls): Quality Error", fontweight="bold")
    axes[1, 1].set_ylabel("Expected Manhattan Error (tiles)")
    axes[1, 1].set_xlabel("Time Step (t)")

    plt.tight_layout()
    plt.savefig("fig_question_3c.png", dpi=300)
    plt.savefig("fig_question_3c.pdf")
    plt.close()
    print("Saved fig_question_3c.png and fig_question_3c.pdf successfully!")


def plot_results_3e(df: pd.DataFrame):
    """Generate and save 1x2 comparison plots for Question 3.e with Seaborn."""
    sns.set_theme(style="whitegrid", font_scale=1.1)
    palette = {"var = 0.25": "#2ecc71", "var = 1.0": "#3498db", "var = 4.0": "#e74c3c"}

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharex=True)

    # 1. Variance effect on Uncertainty
    sns.lineplot(
        data=df,
        x="step",
        y="uncertainty",
        hue="variance",
        palette=palette,
        errorbar="sd",
        ax=axes[0],
    )
    axes[0].set_title("Effect of Sensor Variance on Uncertainty", fontweight="bold")
    axes[0].set_ylabel("Shannon Entropy (bits)")
    axes[0].set_xlabel("Time Step (t)")

    # 2. Variance effect on Error
    sns.lineplot(
        data=df,
        x="step",
        y="error",
        hue="variance",
        palette=palette,
        errorbar="sd",
        ax=axes[1],
    )
    axes[1].set_title("Effect of Sensor Variance on Quality Error", fontweight="bold")
    axes[1].set_ylabel("Expected Manhattan Error (tiles)")
    axes[1].set_xlabel("Time Step (t)")

    plt.tight_layout()
    plt.savefig("fig_question_3e.png", dpi=300)
    plt.savefig("fig_question_3e.pdf")
    plt.close()
    print("Saved fig_question_3e.png and fig_question_3e.pdf successfully!")


if __name__ == "__main__":
    df_3c = run_experiment_3c(num_trials=15, max_steps=40)
    plot_results_3c(df_3c)

    df_3e = run_experiment_3e(num_trials=15, max_steps=40)
    plot_results_3e(df_3e)

    print("\nAll experiments and figures completed successfully!")
