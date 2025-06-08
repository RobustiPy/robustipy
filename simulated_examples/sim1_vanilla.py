import numpy as np
import pandas as pd
from robustipy.models import OLSRobust


def sim1(project_name):
    # 1. Setup
    np.random.seed(192735)
    n = 100
    rng = np.random.default_rng(0)
    z = [f"z{i}" for i in range(1, 5)]

    # 2. Simulate covariates
    X = rng.standard_normal((n, 1))
    Z = rng.standard_normal((n, len(z)))
    df = pd.DataFrame(np.hstack([X, Z]), columns=["x1"] + z)

    # 3. Generate outcome y₁
    df["y1"] = (1 + 2 * df["x1"] + 0.5 * df[z].sum(axis=1)
                + rng.standard_normal(n) * 0.5)

    # 4. Fit robust OLS with cross‐validation
    vanilla_obj = OLSRobust(y=["y1"], x=["x1"], data=df)
    vanilla_obj.fit(controls=z, draws=1000, kfold=10, n_cpu=32, seed=192735)

    # 5. Retrieve and plot results
    vanilla_res = vanilla_obj.get_results()
    vanilla_res.plot(specs=[z[1:3]], figsize=(16, 16), ci=0.95,
                     ic="bic", ext="svg", figpath='../figures',
                     project_name=project_name)
    vanilla_res.summary()


if __name__ == "__main__":
    sim1('sim1_example')