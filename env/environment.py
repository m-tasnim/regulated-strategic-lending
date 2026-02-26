# environment.py
import numpy as np
from .borrower import Borrower


class StrategicLendingEnv:
    """
    Minimal strategic lending environment with:
    - Borrower agents (Borrower objects)
    - Lender (threshold t on reported scores s_i = z_i + a_i)
    - Regulator (penalty λ on good-but-denied borrowers)
    """

    def __init__(
            self,
            N=10_000,
            p_L=0.5,
            theta=0,
            b=1.0,
            h=0.5,
            k_L=0.1,
            k_H=0.3,
            pi_G=0.2,
            pi_B=-0.2,
            z_mean=0,
            z_std=1.0,
            seed=0,
    ):
        # Population parameters
        self.N = N
        self.p_L = p_L          # share of low-cost borrowers
        self.theta = theta      # creditworthiness cutoff (good if z >= theta)

        # Borrower payoff parameters
        self.b = b              # benefit from receiving a loan
        self.h = h              # harm if good borrower is denied

        # Cost parameters for low- and high-cost groups
        self.k_L = k_L
        self.k_H = k_H

        # Lender profits per accepted loan
        self.pi_G = pi_G        # expected profit on good borrower
        self.pi_B = pi_B        # expected profit on bad borrower

        # Distribution of true creditworthiness
        self.z_mean = z_mean
        self.z_std = z_std

        # RNG
        self.rng = np.random.default_rng(seed)

        # List of Borrower agents
        self.borrowers = []

        # Initialise population
        self.reset_population()

    def reset_population(self, seed=None):
        """
        Generate a new borrower population (can be called to resample).
        """
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.borrowers = []

        # True creditworthiness
        z_vals = self.rng.normal(self.z_mean, self.z_std, size=self.N)

        # Group labels: True = low-cost, False = high-cost
        is_low_cost = self.rng.uniform(size=self.N) < self.p_L

        # Create Borrower objects
        for z, low_cost in zip(z_vals, is_low_cost):
            k = self.k_L if low_cost else self.k_H
            borrower = Borrower(
                z=z,
                k=k,
                theta=self.theta,
                b=self.b,
                h=self.h,
                is_low_cost=low_cost,
            )
            self.borrowers.append(borrower)

    def evaluate_threshold(self, t, lam):
        """
        Given a threshold t and penalty weight λ, compute:
        - objective: Π(t) - λ H(t)
        - Π(t): total profit
        - H(t): number of good-but-denied borrowers
        - group-level acceptance rates and avg costs
        """
        Pi = 0.0
        H = 0
        n_L = 0
        n_H = 0
        acc_L = 0
        acc_H = 0
        cost_L = 0.0
        cost_H = 0.0

        for borrower in self.borrowers:
            # Borrower best response
            a = borrower.best_response(t)
            s = borrower.z + a  # reported score

            accepted = s >= t
            is_good = borrower.is_good

            # Lender profit contribution
            if accepted:
                if is_good:
                    Pi += self.pi_G
                else:
                    Pi += self.pi_B

            # Harm: good-but-denied
            if (not accepted) and is_good:
                H += 1

            # Group-level stats
            if borrower.is_low_cost:
                n_L += 1
                if accepted:
                    acc_L += 1
                cost_L += borrower.adjustment_cost(a)
            else:
                n_H += 1
                if accepted:
                    acc_H += 1
                cost_H += borrower.adjustment_cost(a)

        # Avoid divide-by-zero if a group is empty
        acc_L_rate = acc_L / n_L if n_L > 0 else 0.0
        acc_H_rate = acc_H / n_H if n_H > 0 else 0.0
        avg_cost_L = cost_L / n_L if n_L > 0 else 0.0
        avg_cost_H = cost_H / n_H if n_H > 0 else 0.0

        # Objective for the lender under regulation λ
        objective = Pi - lam * H

        stats = {
            "Pi": Pi,
            "H": H,
            "acc_L": acc_L_rate,
            "acc_H": acc_H_rate,
            "avg_cost_L": avg_cost_L,
            "avg_cost_H": avg_cost_H,
        }
        return objective, stats

    def sweep_lambda(self, lambda_grid, t_grid):
        """
        For each λ in lambda_grid, find the threshold t(λ) in t_grid
        that maximizes Π(t) - λ H(t), and return summary results.
        """
        results = []

        for lam in lambda_grid:
            best_obj = -np.inf
            best_t = None
            best_stats = None

            for t in t_grid:
                obj, stats = self.evaluate_threshold(t, lam)
                if obj > best_obj:
                    best_obj = obj
                    best_t = t
                    best_stats = stats

            results.append({
                "lambda": lam,
                "t_star": best_t,
                "Pi": best_stats["Pi"],
                "H": best_stats["H"],
                "acc_L": best_stats["acc_L"],
                "acc_H": best_stats["acc_H"],
                "avg_cost_L": best_stats["avg_cost_L"],
                "avg_cost_H": best_stats["avg_cost_H"],
            })

        return results