"""Core try_loop behavior, tested against the REAL implementation.

(Lesson learned: an earlier version of this file tested a local copy of
try_loop while the real function still had an infinite-loop bug -- the
function under test must be imported, and the registry is injected via
monkeypatch so the tests stay independent of the built-in factors.)

Coverage:
1. multi-level dependency chains resolve within the right number of rounds
2. permanently unsatisfiable dependencies are marked failed WITHOUT looping
   forever (the most important edge case)
3. circular dependencies both fail cleanly
"""
import pytest

#note: "from quantmine import factor_register" would fetch the re-exported
#decorator FUNCTION (it shadows the submodule name on the package); importing
#names directly from the submodule sidesteps the collision
from quantmine.factor_register import FACTOR_REGISTRY, try_loop


@pytest.fixture
def chained_registry(monkeypatch):
    """A -> B -> C three-level chain; D depends on a Z that never exists.

    Factors are injected into the real FACTOR_REGISTRY via monkeypatch, which
    restores the registry after each test.
    """
    def factor_a(base_value: int) -> int:
        return base_value * 2

    def factor_b(A: int) -> int:
        return A + 10

    def factor_c(B: int) -> int:
        return B * 100

    def factor_d(Z: int) -> int:  # Z never exists
        return Z + 1

    fakes = {"A": factor_a, "B": factor_b, "C": factor_c, "D": factor_d}
    for name, func in fakes.items():
        monkeypatch.setitem(FACTOR_REGISTRY, name, func)
    return fakes


def test_multi_level_dependency_chain_resolves(chained_registry):
    """The A->B->C chain must fully resolve regardless of initial iteration order."""
    param_pool = {"base_value": 5}
    pending = {"B": "...", "C": "...", "D": "..."}
    completed = {"A": 10}  # assume A succeeded in the first round

    remaining, completed = try_loop(pending, completed, param_pool)

    assert completed["A"] == 10
    assert completed["B"] == 20
    assert completed["C"] == 2000
    assert "D" in remaining  # D must fail permanently


def test_permanently_failing_dependency_does_not_infinite_loop(chained_registry):
    """Core edge case: unsatisfiable dependencies must terminate, not hang.

    If try_loop regressed to the old infinite-loop behavior this test would
    hang the process, which CI timeouts will surface.
    """
    param_pool = {"base_value": 5}
    remaining, completed = try_loop({"D": "..."}, {}, param_pool)

    assert "D" in remaining
    assert completed.get("D") is None  # marked as failed


def test_circular_dependency_both_permanently_fail(monkeypatch):
    """Circular dependency: X needs Y, Y needs X -- both must fail without hanging."""
    def factor_x(Y: int) -> int:
        return Y + 1

    def factor_y(X: int) -> int:
        return X + 1

    monkeypatch.setitem(FACTOR_REGISTRY, "X", factor_x)
    monkeypatch.setitem(FACTOR_REGISTRY, "Y", factor_y)

    remaining, completed = try_loop({"X": "...", "Y": "..."}, {}, {})

    assert "X" in remaining
    assert "Y" in remaining
    #both are recorded as None (failed), not as computed values
    assert completed.get("X") is None
    assert completed.get("Y") is None
