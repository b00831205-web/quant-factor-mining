# Test suite

Two categories of tests with different data requirements:

## 1. Unit tests (synthetic/dummy data, no real files needed)
- test_market_data.py
- test_factor_registry.py
- test_calculate_all_factors.py
- test_constituents_source.py
- test_quantile_backtest.py
- test_turnover_cost.py
- test_config.py
- test_forward_returns_golden.py (mathematical contract, self-contained)

These run in seconds, should be run after every change, and are CI-friendly.

## 2. Golden-value regression tests (require real legacy data files)
- test_factors_golden.py
- test_cs_ic_golden.py

These compare the new architecture against archived legacy parquet outputs.
When the files are absent (e.g. in CI), the `skipif` markers in `conftest.py`
report them as SKIPPED instead of failing.

## Running

```bash
# from the repo root
python -m pytest test -v

# a single file
python -m pytest test/test_factors_golden.py -v
```

## Maintenance rules

**Any change to factor computation (factor_mining.py):**
run test_factors_golden.py to confirm the verified numerical equivalence
still holds.

**Any change to call_single_factors / calculate_all_factors / try_loop:**
run test_factor_registry.py and test_calculate_all_factors.py -- they pin
down specific bugs found during development (default-priority branching,
KeyError vs ValueError exception contracts, infinite-loop termination).

**When adding a factor:**
add its legacy column prefix to NAME_MAPPING in test_factors_golden.py if
legacy data exists for comparison; otherwise skip the golden test and cover
registration/invocation through the unit tests.
