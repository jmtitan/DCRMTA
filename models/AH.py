import random
import copy
import numpy as np
from typing import List, Any, Dict, Tuple, DefaultDict
from collections import defaultdict, Counter
import arrow

def update_coefs(self, beta: float, omega: float) -> Tuple[float, float, float]:

        """
        return updated beta and omega
        """

        delta = 1e-3

        beta_num = defaultdict(float)
        beta_den = defaultdict(float)
        omega_den = defaultdict(float)

        for u, row in enumerate(self.data.itertuples()):

            p = self.pi(
                row.path, row.exposure_times, row.total_conversions, beta, omega
            )

            r = copy.deepcopy(row.path)

            dts = [
                (arrow.get(row.exposure_times[-1]) - arrow.get(t)).seconds
                for t in row.exposure_times
            ]

            while r:

                # pick channels starting from the last one
                c = r.pop()
                dt = dts.pop()

                beta_den[c] += 1.0 - np.exp(-omega[c] * dt)
                omega_den[c] += p[c] * dt + beta[c] * dt * np.exp(-omega[c] * dt)

                beta_num[c] += p[c]

        # now that we gone through every user, update coefficients for every channel

        beta0 = copy.deepcopy(beta)
        omega0 = copy.deepcopy(omega)

        df = []

        for c in self.channels:

            beta_num[c] = (beta_num[c] > 1e-6) * beta_num[c]
            beta_den[c] = (beta_den[c] > 1e-6) * beta_den[c]
            omega_den[c] = max(omega_den[c], 1e-6)

            if beta_den[c]:
                beta[c] = beta_num[c] / beta_den[c]

            omega[c] = beta_num[c] / omega_den[c]

            df.append(abs(beta[c] - beta0[c]) < delta)
            df.append(abs(omega[c] - omega0[c]) < delta)

        return (beta, omega, sum(df))

def additive_hazard(self, epochs: float = 20, normalize: bool = True):

        """
        additive hazard model as in Multi-Touch Attribution in On-line Advertising with Survival Theory
        """

        beta = {c: random.uniform(0.001, 1) for c in self.channels}
        omega = {c: random.uniform(0.001, 1) for c in self.channels}

        for _ in range(epochs):

            beta, omega, h = self.update_coefs(beta, omega)

            if h == 2 * len(self.channels):
                print(f"converged after {_ + 1} iterations")
                break

        # time window: take the max time instant across all journeys that converged

        additive_hazard = defaultdict(float)

        for u, row in enumerate(self.data.itertuples()):

            p = self.pi(
                row.path, row.exposure_times, row.total_conversions, beta, omega
            )

            for c in p:
                additive_hazard[c] += p[c]

        if normalize:
            additive_hazard = self.normalize_dict(additive_hazard)

        self.attribution["add_haz"] = additive_hazard

        return self
