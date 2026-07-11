"""Tests for the factor registry and call_single_factors parameter resolution.

This file pins down pitfalls actually hit during development:
1. call_single_factors must prefer values from param_pool; the function's own
   defaults are only a fallback when the pool provides nothing (an earlier
   non-exclusive if/if branch let defaults override explicitly passed values)
2. a missing required parameter (no default, not in the pool) must raise the
   exception type that calculate_all_factors catches (KeyError)
"""
import pytest

# import the real implementation: an earlier version kept a local copy of the
# "intended semantics" here -- tests were green while the real code in
# factor_register.py still lacked the fix. The function under test must be imported.
from quantmine.factor_register import call_single_factors as call_single_factor


# a fully self-contained dummy registry, independent of the real
# factor_mining.py content, keeps these unit tests isolated and fast
@pytest.fixture
def dummy_registry():
    registry = {}

    def register(name):
        def decorator(func):
            registry[name] = func
            return func
        return decorator

    def factor_a(base_value: int) -> int:
        return base_value * 2
    register("A")(factor_a)

    def factor_b(A: int) -> int:  # depends on factor A's result
        return A + 10
    register("B")(factor_b)

    def factor_with_default(base_value: int, day: int = 5) -> int:
        return base_value + day
    register("with_default")(factor_with_default)

    return registry


def test_param_pool_value_takes_priority_over_default(dummy_registry):
    """A value explicitly provided in param_pool must override the function default."""
    func = dummy_registry["with_default"]
    result = call_single_factor(func, {"base_value": 1, "day": 20})
    assert result == 21  # 1 + 20: uses the pool's 20, not the default 5


def test_falls_back_to_function_default_when_not_in_pool(dummy_registry):
    """With day absent from the pool, the function's declared default (5) applies without error."""
    func = dummy_registry["with_default"]
    result = call_single_factor(func, {"base_value": 1})
    assert result == 6  # 1 + 5 (default)


def test_missing_required_param_raises_keyerror(dummy_registry):
    """A missing required parameter (no default) must raise KeyError so that
    calculate_all_factors / try_loop's `except KeyError` catches it correctly.
    """
    func = dummy_registry["B"]  # needs A, which is neither in the pool nor defaulted
    with pytest.raises(KeyError):
        call_single_factor(func, {})


def test_dependency_chain_resolves_across_retry_rounds(dummy_registry):
    """Simulates the real calculate_all_factors + try_loop retry logic:
    B depends on A's result; round one computes A and fails B, the retry
    round must then succeed for B.
    """
    param_pool = {"base_value": 5}
    completed = {}
    failures = {}

    for name, func in dummy_registry.items():
        if name == "with_default":
            continue  # not exercised by this case
        try:
            completed[name] = call_single_factor(func, param_pool)
        except KeyError:
            failures[name] = True

    assert "A" in completed
    assert completed["A"] == 10
    assert "B" in failures  # B must fail in round one: A's result is not in completed yet

    # simulate the retry: merge completed into the pool and try again
    for name in list(failures.keys()):
        try:
            current_params = {**param_pool, **completed}
            completed[name] = call_single_factor(dummy_registry[name], current_params)
            del failures[name]
        except KeyError:
            pass

    assert "B" in completed
    assert completed["B"] == 20  # A(10) + 10
    assert len(failures) == 0
