---
name: metaheuristics-optimization-verification
description: >-
  Executes and verifies modular metaheuristics (Genetic Algorithms, Simulated Annealing,
  and Variable Neighborhood Search) in Python or R with strict convergence tracking,
  temperature cooling validation, spatial diversity monitoring, and stagnation audits.
---

# Modular Metaheuristics Verification Skill (Polyglot)

Use this skill whenever solving complex combinatorial, non-convex, or NP-hard optimization problems using metaheuristic search. This workflow enforces monotonic best-so-far tracking, temperature decay checks, population diversity monitoring to detect premature convergence, and visual diagnostic dashboards across **Python** and **R**.

---

## Universal Metaheuristic Verification Flow

```
[ Stage 1: Solution Encoding & Neighborhood Definition ]
                            │
                            ▼
[ Stage 2: Exploration vs. Exploitation Search Dynamics ]
       ┌────────────────────┼────────────────────┐
       ▼                    ▼                    ▼
[ Genetic Algorithm ]  [ Simulated Annealing ]  [ Variable Neighborhood Search ]
(Selection/Crossover)  (Temperature Cooling)    (Systematic Shaking N_k)
       └────────────────────┬────────────────────┘
                            │
                            ▼
[ Stage 3: Convergence, Diversity & Stagnation Auditing ]
                            │
                            ▼
[ Stage 4: Dual Verification (Terminal Tables & Saved Visuals) ]
                            │
                            ▼
[ Stage 5: Feasibility Assertion & Zero-Error Check ]
```

---

## Reference Implementation Scripts

- **Python Module**: [`scripts/metaheuristics_engine.py`](./scripts/metaheuristics_engine.py)
- **R Module**: [`scripts/metaheuristics_engine.R`](./scripts/metaheuristics_engine.R)

---

## Mathematical Invariants & Diagnostic Rules

### 1. Monotonic Best-Found Trajectory
Regardless of algorithm, the best-found objective value across iterations/generations $t$ must be non-increasing (for minimization):
$$f_{\text{best}}(t+1) \le f_{\text{best}}(t), \quad \forall t \in \{1, \dots, T_{\max}\}$$
*Assertion*: $f_{\text{best}}(t+1) \le f_{\text{best}}(t) + 10^{-9}$. Any increase in best-so-far cost is a fatal bug.

### 2. Simulated Annealing: Temperature & Acceptance Probability
For a candidate move $s \rightarrow s'$ with energy delta $\Delta E = f(s') - f(s)$:
$$P(\text{accept } s' \mid s, T_t) = \begin{cases} 1 & \text{if } \Delta E \le 0 \\ \exp\left(-\frac{\Delta E}{T_t}\right) & \text{if } \Delta E > 0 \end{cases}$$
- **Cooling Invariant**: Temperature decay schedule must strictly satisfy $0 < T_{t+1} < T_t$.
- **Acceptance Invariant**: For positive temperature $T > 0$, acceptance probability must lie strictly in $(0, 1]$.

### 3. Population Diversity & Premature Stagnation (GA / VNS)
- **Spatial Population Diversity**:
  $$\text{Div}(\mathcal{P}_t) = \frac{1}{N \cdot d} \sum_{i=1}^N \sum_{j=1}^d |s_{i,j}^t - \bar{s}_j^t|$$
- **Stagnation Counter ($I_{\text{stag}}$)**:
  Measures consecutive iterations without improvement over tolerance $\delta$:
  $$I_{\text{stag}}(t) = t - \max \{ \tau \le t : f_{\text{best}}(\tau) < f_{\text{best}}(\tau - 1) - \delta \}$$
- **Diagnostic State Classification**:
  - `HEALTHY SEARCH`: Steady convergence with active diversity ($\text{Div} > \text{threshold}$).
  - `PREMATURE CONVERGENCE / STAGNANT`: High stagnation counter ($I_{\text{stag}} \ge N_{\text{patience}}$) with collapsed diversity ($\text{Div} \approx 0$).

### 4. Hard Constraint Feasibility Invariant
The final solution $s^*$ must strictly satisfy all problem constraints:
$$g_i(s^*) \le 0, \quad \forall i \in \{1, \dots, m\}$$
*(e.g., valid permutation with no duplicate visits in TSP, or weight within capacity $\sum w_i \le W$ in Knapsack).*

---

## Language Implementation Patterns

### Pattern A: Python (`numpy`, `matplotlib`, `pandas`)

#### 1. Simulated Annealing Engine with Invariant Tracking
```python
import numpy as np
import pandas as pd

def run_simulated_annealing(
    objective_func,
    initial_state: np.ndarray,
    neighbor_operator,
    initial_temp: float = 100.0,
    cooling_rate: float = 0.95,
    max_iterations: int = 500,
) -> tuple[np.ndarray, float, pd.DataFrame]:
    """Execute Simulated Annealing with temperature and monotonic best invariants."""
    current_state = initial_state.copy()
    current_energy = objective_func(current_state)
    
    best_state = current_state.copy()
    best_energy = current_energy
    
    temperature = initial_temp
    history = []
    
    for iteration in range(1, max_iterations + 1):
        candidate_state = neighbor_operator(current_state)
        candidate_energy = objective_func(candidate_state)
        delta_energy = candidate_energy - current_energy
        
        # Metropolis acceptance criterion
        if delta_energy <= 0 or np.random.rand() < np.exp(-delta_energy / temperature):
            current_state = candidate_state.copy()
            current_energy = candidate_energy
            
        if current_energy < best_energy:
            best_state = current_state.copy()
            best_energy = current_energy
            
        # Record trajectory
        history.append({
            "iteration": iteration,
            "temperature": temperature,
            "current_energy": current_energy,
            "best_energy": best_energy,
        })
        
        # Cooling update
        temperature *= cooling_rate
        
    history_df = pd.DataFrame(history)
    
    # Invariant assertion: best energy must be monotonically non-increasing
    assert (history_df["best_energy"].diff().dropna() <= 1e-9).all(), "Monotonic best invariant violated!"
    print(f"[SA Complete] Initial Energy: {history_df['current_energy'].iloc[0]:.4f} -> Final Best: {best_energy:.4f}")
    return best_state, best_energy, history_df
```

#### 2. Genetic Algorithm with Diversity & Stagnation Tracking
```python
def compute_population_diversity(population: np.ndarray) -> float:
    """Calculate average spatial dispersion from population centroid."""
    centroid = population.mean(axis=0)
    return float(np.mean(np.abs(population - centroid)))

def run_genetic_algorithm(
    fitness_func,
    pop_size: int = 50,
    dim: int = 10,
    bounds: tuple[float, float] = (-5.0, 5.0),
    generations: int = 100,
    patience: int = 15,
) -> tuple[np.ndarray, float, pd.DataFrame]:
    """Execute Genetic Algorithm with diversity tracking and stagnation index."""
    lb, ub = bounds
    population = np.random.uniform(lb, ub, size=(pop_size, dim))
    fitness_scores = np.array([fitness_func(ind) for ind in population])
    
    best_idx = np.argmin(fitness_scores)
    best_ind = population[best_idx].copy()
    best_fit = fitness_scores[best_idx]
    
    history = []
    stagnation_counter = 0
    
    for gen in range(1, generations + 1):
        prev_best = best_fit
        diversity = compute_population_diversity(population)
        
        # Elitism: retain top 2
        elite_indices = np.argsort(fitness_scores)[:2]
        new_pop = [population[i].copy() for i in elite_indices]
        
        # Selection, Crossover & Mutation
        while len(new_pop) < pop_size:
            # Tournament selection
            idx1, idx2 = np.random.choice(pop_size, size=2, replace=False)
            parent1 = population[idx1 if fitness_scores[idx1] < fitness_scores[idx2] else idx2]
            
            idx3, idx4 = np.random.choice(pop_size, size=2, replace=False)
            parent2 = population[idx3 if fitness_scores[idx3] < fitness_scores[idx4] else idx4]
            
            # Blend Crossover (BLX-alpha)
            alpha = 0.5
            gamma = np.random.uniform(-alpha, 1 + alpha, size=dim)
            child = np.clip(gamma * parent1 + (1 - gamma) * parent2, lb, ub)
            
            # Mutation
            if np.random.rand() < 0.2:
                mutation_mask = np.random.rand(dim) < 0.3
                child[mutation_mask] += np.random.normal(0, 0.5, size=mutation_mask.sum())
                child = np.clip(child, lb, ub)
                
            new_pop.append(child)
            
        population = np.array(new_pop)
        fitness_scores = np.array([fitness_func(ind) for ind in population])
        
        curr_best_idx = np.argmin(fitness_scores)
        if fitness_scores[curr_best_idx] < best_fit:
            best_fit = fitness_scores[curr_best_idx]
            best_ind = population[curr_best_idx].copy()
            stagnation_counter = 0
        else:
            stagnation_counter += 1
            
        history.append({
            "generation": gen,
            "best_fitness": best_fit,
            "mean_fitness": float(fitness_scores.mean()),
            "diversity": diversity,
            "stagnation": stagnation_counter,
            "status": "STAGNANT" if stagnation_counter >= patience else "HEALTHY",
        })
        
    history_df = pd.DataFrame(history)
    assert (history_df["best_fitness"].diff().dropna() <= 1e-9).all(), "Monotonic best invariant violated!"
    return best_ind, best_fit, history_df
```

#### 3. Save Diagnostics Dashboard in `output/`
```python
import pathlib
import matplotlib.pyplot as plt

def save_metaheuristic_dashboard(history_df: pd.DataFrame, output_dir: str = "output", prefix: str = "ga") -> pathlib.Path:
    """Save dual-panel convergence and diversity diagnostic dashboard."""
    path = pathlib.Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Subplot 1: Convergence Trajectory
    x_col = "generation" if "generation" in history_df.columns else "iteration"
    y_best = "best_fitness" if "best_fitness" in history_df.columns else "best_energy"
    y_curr = "mean_fitness" if "mean_fitness" in history_df.columns else "current_energy"
    
    axes[0].plot(history_df[x_col], history_df[y_best], color="#2b5c8f", lw=2.2, label="Best Found (Monotonic)")
    axes[0].plot(history_df[x_col], history_df[y_curr], color="#d95f02", lw=1.2, alpha=0.6, label="Current / Mean")
    axes[0].set_title(f"Convergence Trajectory ({prefix.upper()})", fontweight="bold")
    axes[0].set_xlabel(x_col.capitalize())
    axes[0].set_ylabel("Objective Cost / Fitness")
    axes[0].legend(loc="upper right")
    axes[0].grid(True, linestyle="--", alpha=0.5)
    
    # Subplot 2: Diversity Decay or Temperature Profile
    if "diversity" in history_df.columns:
        axes[1].plot(history_df[x_col], history_df["diversity"], color="#7570b3", lw=2.0)
        axes[1].set_title("Population Diversity Decay", fontweight="bold")
        axes[1].set_ylabel("Spatial Dispersion")
    elif "temperature" in history_df.columns:
        axes[1].plot(history_df[x_col], history_df["temperature"], color="#e7298a", lw=2.0)
        axes[1].set_title("Cooling Temperature Decay", fontweight="bold")
        axes[1].set_ylabel("Temperature")
        
    axes[1].set_xlabel(x_col.capitalize())
    axes[1].grid(True, linestyle="--", alpha=0.5)
    
    fig.tight_layout()
    plot_file = path / f"{prefix}_diagnostics_dashboard.png"
    fig.savefig(plot_file, dpi=300)
    plt.close(fig)
    print(f"[Visual Saved] Metaheuristic dashboard: {plot_file.resolve()}")
    return plot_file
```

---

### Pattern B: R (`stats`, `graphics`)

```r
run_simulated_annealing_r <- function(obj_func, init_state, temp_init = 100, cooling = 0.95, iters = 500) {
  res <- optim(
    par = init_state,
    fn = obj_func,
    method = "SANN",
    control = list(maxit = iters, temp = temp_init, tmax = 10)
  )
  
  cat("\n==================================================\n")
  cat("SIMULATED ANNEALING RESULTS (R)\n")
  cat("==================================================\n")
  cat(sprintf("Optimal Objective Value: %.4f\n", res$value))
  cat(sprintf("Function Evaluations    : %d\n", res$counts["function"]))
  cat(sprintf("Convergence Code        : %d\n", res$convergence))
  cat("==================================================\n\n")
  return(res)
}
```

---

## Universal Verification Checklist

Before marking any metaheuristic search task complete:
1. **Monotonic Best Invariant**: Confirm the recorded best-found objective is strictly non-increasing across all iterations.
2. **Cooling Invariant (SA)**: Confirm temperature strictly decreases toward zero ($0 < T_{t+1} < T_t$).
3. **Diversity & Stagnation Audit**: Ensure spatial dispersion is logged and stagnation counter did not fail silently in early generations.
4. **Hard Constraint Feasibility**: Confirm the returned solution satisfies all problem boundaries and discrete encodings.
5. **Visual Dashboard**: Confirm `output/<prefix>_diagnostics_dashboard.png` is generated and saved.
