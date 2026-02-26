import numpy as np
from ..env.environment import StrategicLendingEnv


def find_unregulated_optimum(params, t_grid):
    """
    For a given parameter dict, create an environment, and find
    the profit-maximizing threshold t* at λ = 0. Returns (t_star, stats).
    """
    env = StrategicLendingEnv(**params)
    best_obj = -np.inf
    best_t = None
    best_stats = None

    lam = 0.0  # unregulated

    for t in t_grid:
        obj, stats = env.evaluate_threshold(t, lam)
        if obj > best_obj:
            best_obj = obj
            best_t = t
            best_stats = stats

    return best_t, best_stats


def main():
    N = 10_000
    z_mean, z_std = 0.0, 1.0
    theta = 0.0

    # Threshold search range
    t_grid = np.linspace(-1.5, 4.0, 281)

    # Parameter grids to explore
    p_L_grid   = [0.3, 0.5]           # share of low-cost borrowers
    b_grid     = [0.8, 1.0]           # loan benefit
    h_grid     = [0.1, 0.3, 0.5]      # harm if good borrower denied
    k_L_grid   = [0.1]                # low-cost adjustment cost
    k_H_grid   = [0.3, 0.7, 1.0]      # high-cost adjustment cost
    pi_G_grid  = [0.2]                # profit on good borrowers
    pi_B_grid  = [-0.2, -0.4, -0.6]   # profit on bad borrowers (negative)

    results = []

    # Simple nested loops over the grid
    for p_L in p_L_grid:
        for b in b_grid:
            for h in h_grid:
                for k_L in k_L_grid:
                    for k_H in k_H_grid:
                        for pi_G in pi_G_grid:
                            for pi_B in pi_B_grid:
                                params = dict(
                                    N=N,
                                    p_L=p_L,
                                    theta=theta,
                                    b=b,
                                    h=h,
                                    k_L=k_L,
                                    k_H=k_H,
                                    pi_G=pi_G,
                                    pi_B=pi_B,
                                    z_mean=z_mean,
                                    z_std=z_std,
                                    seed=0,     # fixed seed for comparability
                                )

                                t_star, stats = find_unregulated_optimum(params, t_grid)

                                H = stats["H"]
                                Pi = stats["Pi"]
                                acc_L = stats["acc_L"]
                                acc_H = stats["acc_H"]
                                cost_L = stats["avg_cost_L"]
                                cost_H = stats["avg_cost_H"]

                                results.append({
                                    "p_L": p_L,
                                    "b": b,
                                    "h": h,
                                    "k_L": k_L,
                                    "k_H": k_H,
                                    "pi_G": pi_G,
                                    "pi_B": pi_B,
                                    "t_star": t_star,
                                    "Pi": Pi,
                                    "H": H,
                                    "H_frac": H / N,
                                    "acc_L": acc_L,
                                    "acc_H": acc_H,
                                    "cost_L": cost_L,
                                    "cost_H": cost_H,
                                })

    # Sort by harm fraction (descending)
    results_sorted = sorted(results, key=lambda r: r["H_frac"], reverse=True)

    # Print top K regimes
    K = 15
    print(f"Top {K} parameter regimes by harm fraction (λ = 0):")
    print("idx  H     H_frac   Pi     t*    p_L  b   h   k_H  pi_B  acc_L  acc_H")
    for i, r in enumerate(results_sorted[:K]):
        print(
            f"{i:2d}  {r['H']:5d}  {r['H_frac']:.4f}  {r['Pi']:6.1f}  {r['t_star']:.2f}  "
            f"{r['p_L']:.2f}  {r['b']:.2f}  {r['h']:.2f}  {r['k_H']:.2f}  {r['pi_B']:.2f}  "
            f"{r['acc_L']:.3f}  {r['acc_H']:.3f}"
        )


if __name__ == "__main__":
    main()