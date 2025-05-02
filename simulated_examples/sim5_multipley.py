import numpy as np
import pandas as pd
from robustipy.models import OLSRobust


def sim5(project_name):

    # 1. Simulation parameters
    n, p = 1000, 5
    np.random.seed(192735)
    Σ_X = np.eye(p)
    rng = np.random.default_rng()

    # 2. Draw covariates X and error ε
    X = rng.multivariate_normal(mean=np.zeros(p), cov=Σ_X, size=n)
    ε = rng.standard_normal(n)

    # 3. Stack β‐vectors into a (4×p) matrix
    B = np.array([
        [0.20, 0.50, -0.40, -0.10, 0.20],
        [0.30, 0.40, -0.35, -0.10, 0.20],
        [0.15, 0.60, -0.45, -0.10, 0.20],
        [0.40, 0.30, -0.50, -0.10, 0.20],
    ])  # shape = (4, p)

    # 4. Build DataFrame and generate outcomes y₁…y₄ and controls z₁…z₄
    df = pd.DataFrame(X, columns=[f'x{i}' for i in range(1, p + 1)])
    df[[f'y{j}' for j in range(1, 5)]] = X @ B.T + ε[:, None]
    df[[f'z{j}' for j in range(1, 5)]] = rng.standard_normal((n, 4))

    # 5. Specify outcome and control names
    Y = [f'y{j}' for j in range(1, 5)]
    Z = [f'z{j}' for j in range(1, 5)]

    # 6. Fit robust OLS
    model = OLSRobust(y=Y, x=['x1'], data=df)
    model.fit(controls=Z, draws=1000, kfold=10)

    # 7. Retrieve and plot results
    res = model.get_results()
    res.plot(
        loess=False,
        specs=[['y1', 'y2', 'z1', 'z2'],
               ['y3', 'y4', 'z3', 'z4']],
        figsize=(16, 8),
        project_name=project_name
    )
    res.summary()

if __name__ == "__main__":
    sim5('sim5_example')
