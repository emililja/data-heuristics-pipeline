"""Modular Metaheuristics Engine and Verification Module.

Enforces monotonic best-found tracking, temperature cooling decay verification,
spatial population diversity monitoring, and stagnation index auditing.
"""

from __future__ import annotations

import pathlib
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def calculate_spatial_diversity(population: np.ndarray) -> float:
    """Calculate average spatial dispersion of population from its centroid."""
    if population.ndim != 2 or len(population) == 0:
        raise ValueError("Population must be a non-empty 2D array.")

    population_centroid = population.mean(axis=0)
    spatial_dispersion = np.mean(np.abs(population - population_centroid))
    return float(spatial_dispersion)


def run_simulated_annealing(
    objective_function: Callable[[np.ndarray], float],
    initial_state: np.ndarray,
    neighbor_generator: Callable[[np.ndarray], np.ndarray],
    initial_temperature: float = 100.0,
    cooling_rate: float = 0.95,
    max_iterations: int = 400,
    random_seed: int = 42,
) -> tuple[np.ndarray, float, pd.DataFrame]:
    """Execute Simulated Annealing with temperature decay and monotonic best invariants."""
    if initial_temperature <= 0:
        raise ValueError("initial_temperature must be strictly positive.")

    if not (0.0 < cooling_rate < 1.0):
        raise ValueError("cooling_rate must be between 0.0 and 1.0.")

    np.random.seed(random_seed)
    current_state = initial_state.copy()
    current_energy = float(objective_function(current_state))

    best_state = current_state.copy()
    best_energy = current_energy

    current_temp = initial_temperature
    trajectory_records = []

    for iteration in range(1, max_iterations + 1):
        candidate_state = neighbor_generator(current_state)
        candidate_energy = float(objective_function(candidate_state))
        delta_energy = candidate_energy - current_energy

        # Metropolis acceptance criterion
        accept_move = False
        if delta_energy <= 0:
            accept_move = True
        else:
            exponent = -delta_energy / current_temp
            # Guard against underflow
            acceptance_probability = np.exp(exponent) if exponent > -700 else 0.0
            assert 0.0 <= acceptance_probability <= 1.0, (
                f"Metropolis probability out of bounds: {acceptance_probability:.2e}"
            )
            accept_move = bool(np.random.rand() < acceptance_probability)

        if accept_move:
            current_state = candidate_state.copy()
            current_energy = candidate_energy

        if current_energy < best_energy:
            best_state = current_state.copy()
            best_energy = current_energy

        trajectory_records.append(
            {
                "iteration": iteration,
                "temperature": current_temp,
                "current_energy": current_energy,
                "best_energy": best_energy,
            }
        )

        # Geometric cooling update
        next_temp = current_temp * cooling_rate
        assert next_temp < current_temp, "Cooling schedule invariant violated."
        current_temp = next_temp

    history_dataframe = pd.DataFrame(trajectory_records)

    # Invariant assertion: best energy must be monotonically non-increasing
    energy_deltas = history_dataframe["best_energy"].diff().dropna()
    assert (energy_deltas <= 1e-9).all(), "Monotonic best-found invariant violated in SA!"

    initial_energy = history_dataframe["current_energy"].iloc[0]
    percentage_improvement = (
        ((initial_energy - best_energy) / abs(initial_energy)) * 100
        if abs(initial_energy) > 1e-6
        else 0.0
    )

    delimiter = "=" * 65
    print(f"\n{delimiter}\nSIMULATED ANNEALING CONVERGENCE PROFILE\n{delimiter}")
    print(f"Initial Energy: {initial_energy:.4f} -> Final Best Energy: {best_energy:.4f}")
    print(f"Total Improvement: {percentage_improvement:.2f}%")
    print(f"Final Temperature: {current_temp:.6f} (Initial: {initial_temperature:.2f})")
    print(f"{delimiter}\n")

    return best_state, best_energy, history_dataframe


def run_genetic_algorithm(
    objective_function: Callable[[np.ndarray], float],
    dimension: int = 10,
    bounds: tuple[float, float] = (-5.0, 5.0),
    population_size: int = 50,
    generations: int = 100,
    mutation_probability: float = 0.20,
    patience: int = 15,
    random_seed: int = 42,
) -> tuple[np.ndarray, float, pd.DataFrame]:
    """Execute Genetic Algorithm with spatial diversity and stagnation index monitoring."""
    if population_size < 4:
        raise ValueError("population_size must be at least 4.")

    np.random.seed(random_seed)
    lower_bound, upper_bound = bounds

    # Stage 1: Population initialization within domain bounds
    population = np.random.uniform(
        lower_bound,
        upper_bound,
        size=(population_size, dimension),
    )
    fitness_scores = np.array([float(objective_function(ind)) for ind in population])

    best_index = int(np.argmin(fitness_scores))
    best_individual = population[best_index].copy()
    best_fitness = float(fitness_scores[best_index])

    generational_records = []
    stagnation_counter = 0

    for generation in range(1, generations + 1):
        spatial_diversity = calculate_spatial_diversity(population)

        # Stage 2: Elitism (preserve top 2 individuals)
        elite_indices = np.argsort(fitness_scores)[:2]
        next_population = [population[idx].copy() for idx in elite_indices]

        # Stage 3: Selection, Crossover, and Mutation
        while len(next_population) < population_size:
            # Tournament selection (size 2)
            c1, c2 = np.random.choice(population_size, size=2, replace=False)
            parent1 = population[c1 if fitness_scores[c1] < fitness_scores[c2] else c2]

            c3, c4 = np.random.choice(population_size, size=2, replace=False)
            parent2 = population[c3 if fitness_scores[c3] < fitness_scores[c4] else c4]

            # Blend Crossover (BLX-alpha with alpha = 0.5)
            alpha_param = 0.5
            gamma = np.random.uniform(-alpha_param, 1.0 + alpha_param, size=dimension)
            offspring = gamma * parent1 + (1.0 - gamma) * parent2
            offspring = np.clip(offspring, lower_bound, upper_bound)

            # Gaussian mutation
            if np.random.rand() < mutation_probability:
                mutation_mask = np.random.rand(dimension) < 0.30
                if mutation_mask.any():
                    offspring[mutation_mask] += np.random.normal(
                        0.0,
                        0.50,
                        size=mutation_mask.sum(),
                    )
                    offspring = np.clip(offspring, lower_bound, upper_bound)

            next_population.append(offspring)

        population = np.array(next_population)
        fitness_scores = np.array([float(objective_function(ind)) for ind in population])

        # Stage 4: Stagnation tracking and best individual update
        current_best_index = int(np.argmin(fitness_scores))
        current_best_val = float(fitness_scores[current_best_index])

        if current_best_val < (best_fitness - 1e-6):
            best_fitness = current_best_val
            best_individual = population[current_best_index].copy()
            stagnation_counter = 0
        else:
            stagnation_counter += 1

        search_status = (
            "STAGNANT / LOCAL MINIMUM"
            if stagnation_counter >= patience
            else "HEALTHY SEARCH"
        )

        generational_records.append(
            {
                "generation": generation,
                "best_fitness": best_fitness,
                "mean_fitness": float(fitness_scores.mean()),
                "diversity": spatial_diversity,
                "stagnation": stagnation_counter,
                "status": search_status,
            }
        )

    history_dataframe = pd.DataFrame(generational_records)

    # Invariant assertion: best fitness must be monotonically non-increasing
    fitness_deltas = history_dataframe["best_fitness"].diff().dropna()
    assert (fitness_deltas <= 1e-9).all(), "Monotonic best-found invariant violated in GA!"

    initial_fitness = history_dataframe["mean_fitness"].iloc[0]
    improvement_rate = (
        ((initial_fitness - best_fitness) / abs(initial_fitness)) * 100
        if abs(initial_fitness) > 1e-6
        else 0.0
    )

    delimiter = "=" * 65
    print(f"\n{delimiter}\nGENETIC ALGORITHM CONVERGENCE PROFILE\n{delimiter}")
    print(f"Initial Mean Fitness: {initial_fitness:.4f} -> Final Best: {best_fitness:.4f}")
    print(f"Total Improvement: {improvement_rate:.2f}%")
    print(
        f"Final Diversity: {history_dataframe['diversity'].iloc[-1]:.4f} | "
        f"Final Stagnation: {stagnation_counter} (Patience: {patience})"
    )
    print(f"{delimiter}\n")

    return best_individual, best_fitness, history_dataframe


def save_metaheuristic_dashboard(
    history_data: pd.DataFrame,
    output_directory: str = "output",
    filename_prefix: str = "metaheuristic",
) -> pathlib.Path:
    """Save dual-panel convergence and diversity/temperature diagnostic dashboard."""
    output_path = pathlib.Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)

    figure, axes = plt.subplots(1, 2, figsize=(14, 5))

    x_column = "generation" if "generation" in history_data.columns else "iteration"
    best_column = "best_fitness" if "best_fitness" in history_data.columns else "best_energy"
    current_column = "mean_fitness" if "mean_fitness" in history_data.columns else "current_energy"

    # Subplot 1: Convergence Trajectory
    axes[0].plot(
        history_data[x_column],
        history_data[best_column],
        color="#2b5c8f",
        linewidth=2.2,
        label="Best Found (Monotonic)",
    )
    axes[0].plot(
        history_data[x_column],
        history_data[current_column],
        color="#d95f02",
        linewidth=1.2,
        alpha=0.6,
        label="Current / Generational Mean",
    )
    axes[0].set_title(
        f"Convergence Trajectory ({filename_prefix.upper()})",
        fontsize=11,
        fontweight="bold",
    )
    axes[0].set_xlabel(x_column.capitalize(), fontsize=10)
    axes[0].set_ylabel("Objective Cost / Energy", fontsize=10)
    axes[0].legend(loc="upper right")
    axes[0].grid(True, linestyle="--", alpha=0.5)

    # Subplot 2: Diversity Decay or Cooling Profile
    if "diversity" in history_data.columns:
        axes[1].plot(
            history_data[x_column],
            history_data["diversity"],
            color="#7570b3",
            linewidth=2.0,
            label="Spatial Dispersion",
        )
        axes[1].set_title("Population Diversity Decay Profile", fontsize=11, fontweight="bold")
        axes[1].set_ylabel("Spatial Dispersion from Centroid", fontsize=10)
        axes[1].legend(loc="upper right")
    elif "temperature" in history_data.columns:
        axes[1].plot(
            history_data[x_column],
            history_data["temperature"],
            color="#e7298a",
            linewidth=2.0,
            label="Temperature (T)",
        )
        axes[1].set_title("Cooling Temperature Decay Profile", fontsize=11, fontweight="bold")
        axes[1].set_ylabel("Temperature", fontsize=10)
        axes[1].legend(loc="upper right")

    axes[1].set_xlabel(x_column.capitalize(), fontsize=10)
    axes[1].grid(True, linestyle="--", alpha=0.5)

    figure.tight_layout()
    artifact_path = output_path / f"{filename_prefix}_diagnostics_dashboard.png"
    figure.savefig(artifact_path, dpi=300)
    plt.close(figure)

    print(f"[Visual Saved] Metaheuristic dashboard written to: {artifact_path.resolve()}")
    return artifact_path


if __name__ == "__main__":
    print("Initiating Modular Metaheuristics Verification Self-Test...")

    # Benchmark Function: Rastrigin Function (highly multimodal, global minimum at origin = 0)
    def rastrigin_function(x: np.ndarray) -> float:
        dimension = len(x)
        return float(10.0 * dimension + np.sum(x**2 - 10.0 * np.cos(2.0 * np.pi * x)))

    # Task 1: Simulated Annealing Verification
    print("\n--- Running Simulated Annealing Self-Test ---")
    initial_sa_point = np.random.uniform(-4.0, 4.0, size=5)

    def gaussian_neighbor_operator(current: np.ndarray) -> np.ndarray:
        perturbation = np.random.normal(0.0, 0.40, size=len(current))
        return np.clip(current + perturbation, -5.12, 5.12)

    best_sa_state, best_sa_energy, sa_history = run_simulated_annealing(
        objective_function=rastrigin_function,
        initial_state=initial_sa_point,
        neighbor_generator=gaussian_neighbor_operator,
        initial_temperature=120.0,
        cooling_rate=0.96,
        max_iterations=300,
        random_seed=42,
    )
    save_metaheuristic_dashboard(
        sa_history, output_directory="output", filename_prefix="sa_rastrigin"
    )

    # Task 2: Genetic Algorithm Verification
    print("\n--- Running Genetic Algorithm Self-Test ---")
    best_ga_ind, best_ga_fitness, ga_history = run_genetic_algorithm(
        objective_function=rastrigin_function,
        dimension=5,
        bounds=(-5.12, 5.12),
        population_size=60,
        generations=80,
        patience=12,
        random_seed=42,
    )
    save_metaheuristic_dashboard(
        ga_history, output_directory="output", filename_prefix="ga_rastrigin"
    )

    print("\nModular Metaheuristics self-test successfully verified for SA and GA.")
