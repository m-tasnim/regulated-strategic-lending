from dataclasses import dataclass


@dataclass
class Borrower:
    """
    Borrower agent in the regulated strategic lending model.
    """
    z: float
    k: float
    theta: float
    b: float
    h: float
    is_low_cost: bool

    @property
    def is_good(self) -> bool:
        """Whether the borrower is creditworthy (good) according to theta."""
        return self.z >= self.theta

    def best_response(self, t: float) -> float:
        """
        Compute best-response adjustment a_i given lender threshold t.
        """
        # Already above threshold: no need to adjust
        if self.z >= t:
            return 0.0

        # Amount needed to just reach the threshold
        delta = t - self.z
        if delta <= 0:
            return 0.0  # numerical safety

        # Utility if adjust
        U_adjust = self.b - self.k * (delta ** 2)

        # Utility if not adjust
        U_no = -self.h if self.is_good else 0.0

        # Adjust only if it is at least as good as not adjusting
        if U_adjust >= U_no:
            return delta
        else:
            return 0.0

    def adjustment_cost(self, a: float) -> float:
        """Quadratic adjustment cost k * a^2."""
        return self.k * (a ** 2)