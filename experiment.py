import numpy as np
from env.environment import StrategicLendingEnv

# Create environment with default parameters
env = StrategicLendingEnv(
    N=10_000,
    p_L=0.3,
    theta=0.0,
    b=1.0,
    h=0.1,
    k_L=0.1,
    k_H=0.3,
    pi_G=0.2,
    pi_B=-0.6,
    z_mean=0.0,
    z_std=1.0,
    seed=0,
)
t_grid = np.linspace(-1.5, 4.0, 261)

# Define λ grid and threshold grid
lambda_grid = np.linspace(0.0, 2.0, 21)
# t_grid = np.linspace(-1.5, 3.0, 181)

# Run sweep
results = env.sweep_lambda(lambda_grid, t_grid)

# Print summary
print("lambda  t*     Pi      H    acc_L  acc_H  cost_L  cost_H")
for r in results:
    print(f"{r['lambda']:5.2f}  {r['t_star']:4.2f}  "
          f"{r['Pi']:6.3f}  {r['H']:4d}  "
          f"{r['acc_L']:6.3f}  {r['acc_H']:6.3f}  "
          f"{r['avg_cost_L']:6.3f}  {r['avg_cost_H']:6.3f}")


import pandas as pd
import matplotlib.pyplot as plt

# Convert results to DataFrame
df = pd.DataFrame(results)

# Optional: sort by lambda just to be safe
df = df.sort_values("lambda").reset_index(drop=True)