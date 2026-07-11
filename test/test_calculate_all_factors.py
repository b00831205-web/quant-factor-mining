"""
测试 try_loop 的核心行为：
1. 多层依赖链能在正确的轮次内被解决
2. 永久无法满足的依赖（缺少的东西永远不存在）会被正确识别为失败，
   且不会导致死循环（这是最重要的边界情况）
"""
import inspect
import pytest


def call_single_factor(func, param_pool: dict):
    sig = inspect.signature(func)
    kwargs = {}
    for name, param in sig.parameters.items():
        if name in param_pool:
            kwargs[name] = param_pool[name]
        elif param.default is not inspect.Parameter.empty:
            kwargs[name] = param.default
        else:
            raise KeyError(name)
    return func(**kwargs)


def try_loop(pending: dict, completed: dict, param_pool: dict, registry: dict):
    """被测函数：从 factor_register.py 复制/import 过来测试。"""
    pending = pending.copy()
    completed = completed.copy()
    while pending:
        length_before = len(pending)
        for factor_name in list(pending.keys()):
            current_params = {**param_pool, **completed}
            try:
                completed[factor_name] = call_single_factor(registry[factor_name], current_params)
                del pending[factor_name]
            except KeyError:
                continue
        if len(pending) == length_before:
            break
    return pending, completed


@pytest.fixture
def chained_registry():
    """A -> B -> C 三层依赖链，D依赖一个永远不存在的Z。"""
    def factor_a(base_value: int) -> int:
        return base_value * 2

    def factor_b(A: int) -> int:
        return A + 10

    def factor_c(B: int) -> int:
        return B * 100

    def factor_d(Z: int) -> int:  # Z永远不存在
        return Z + 1

    return {"A": factor_a, "B": factor_b, "C": factor_c, "D": factor_d}


def test_multi_level_dependency_chain_resolves(chained_registry):
    """三层依赖链 A->B->C 应该全部成功，不管初始遍历顺序如何。"""
    param_pool = {"base_value": 5}
    pending = {"B": "...", "C": "...", "D": "..."}
    completed = {"A": 10}  # 假设A已经在第一轮算出来了

    remaining, completed = try_loop(pending, completed, param_pool, chained_registry)

    assert completed["A"] == 10
    assert completed["B"] == 20
    assert completed["C"] == 2000
    assert "D" in remaining  # D应该永久失败


def test_permanently_failing_dependency_does_not_infinite_loop(chained_registry):
    """核心边界测试：依赖永远无法满足时，必须能正确终止，不能死循环。
    用pytest-timeout或者简单的迭代次数断言来防止真的挂起测试进程。
    """
    param_pool = {"base_value": 5}
    pending = {"D": "..."}
    completed = {}

    # 如果try_loop有bug导致死循环，这一行会直接把测试挂起，
    # pytest默认没有超时保护，但至少能明确看到测试卡住不动，
    # 结合CI的超时设置能捕获这种情况
    remaining, completed = try_loop(pending, completed, param_pool, chained_registry)

    assert "D" in remaining
    assert "D" not in completed


def test_circular_dependency_both_permanently_fail():
    """循环依赖：X需要Y，Y需要X，两者都应该永久失败，不应该死循环。"""
    def factor_x(Y: int) -> int:
        return Y + 1

    def factor_y(X: int) -> int:
        return X + 1

    registry = {"X": factor_x, "Y": factor_y}
    pending = {"X": "...", "Y": "..."}
    completed = {}

    remaining, completed = try_loop(pending, completed, {}, registry)

    assert "X" in remaining
    assert "Y" in remaining
    assert len(completed) == 0
