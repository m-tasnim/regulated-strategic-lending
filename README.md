# Strategic Lending with Regulation

This repository contains code for the working paper "The Role of Regulatory Institutions in Strategic Lending", a minimal simulation model for studying how a simple regulatory mechanism affects outcomes in a strategic lending environment with heterogeneous borrowers.

We consider a one-shot interaction between three types of agents:

- **Borrowers**: each has a true creditworthiness \(z_i\) and an adjustment cost parameter \(k_i\). A fixed threshold \(\theta\) separates *good* (creditworthy) from *bad* borrowers. Borrowers can pay a quadratic cost to adjust their reported score \(s_i = z_i + a_i\).
- **Lender**: applies a threshold rule \(t\) to reported scores. Loans to good borrowers yield positive expected profit \(\pi^G > 0\); loans to bad borrowers yield negative expected profit \(\pi^B < 0\).
- **Regulator**: penalizes the lender according to the number of good borrowers who are denied credit. The penalty weight \(\lambda\) captures the strength of regulation.

## Repository structure

---

## Requirements
