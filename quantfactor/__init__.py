"""quantfactor: an honest-statistics equity factor research library.

Modules:
    data_acquisition   Batch price/volume download with retry & checkpoints.
    factor_mining      Vectorized factor computations.
    ic_calculator      IC testing, Newey-West, multiple-testing control,
                       orthogonalization, train/test workflow.
    back_testing       Point-in-time quantile backtest, turnover costs,
                       sanity tests.
    factor_attribution Carhart four-factor attribution.
    visualization      Plotly figure builders for the analytics dashboard.

Run research entry points from the repo root as modules, e.g.:
    python -m quantfactor.ic_calculator
    python -m quantfactor.back_testing
    python -m quantfactor.factor_attribution
"""

__version__ = "0.1.0"
