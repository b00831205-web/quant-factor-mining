"""
测试因子注册机制和 call_single_factors 的参数解析逻辑。

这个文件专门覆盖对话中实际踩过的坑：
1. call_single_factors 应该优先用 param_pool 里的值，
   函数自身默认值只在 param_pool 没提供时才作为兜底
   （之前 if/if 写错成非互斥分支，导致默认值覆盖了显式传入的值）
2. 缺少必需参数（无默认值、param_pool也没提供）应该抛出
   能被 calculate_all_factors 正确捕获的异常类型
"""
import inspect
import pytest


# 用完全独立的、和真实项目无关的 dummy 注册表做测试，
# 避免这些单元测试依赖真实的 factor_mining.py 内容，
# 保持测试的独立性和运行速度
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

    def factor_b(A: int) -> int:  # 依赖因子A的结果
        return A + 10
    register("B")(factor_b)

    def factor_with_default(base_value: int, day: int = 5) -> int:
        return base_value + day
    register("with_default")(factor_with_default)

    return registry


# 直接import真实实现: 之前这里放的是"期望语义"的本地复制版,
# 测试绿了但factor_register.py里的真代码并没有这个修复——教训是被测函数必须import
from quantfactor.factor_register import call_single_factors as call_single_factor


def test_param_pool_value_takes_priority_over_default(dummy_registry):
    """param_pool显式提供的值，应该覆盖函数自身的默认值。"""
    func = dummy_registry["with_default"]
    result = call_single_factor(func, {"base_value": 1, "day": 20})
    assert result == 21  # 1 + 20，用的是param_pool的20，不是默认值5


def test_falls_back_to_function_default_when_not_in_pool(dummy_registry):
    """param_pool没提供day，应该用函数自身声明的默认值5，不应该报错。"""
    func = dummy_registry["with_default"]
    result = call_single_factor(func, {"base_value": 1})
    assert result == 6  # 1 + 5(默认值)


def test_missing_required_param_raises_keyerror(dummy_registry):
    """必需参数(无默认值)缺失时，应该抛出KeyError，
    这样才能被calculate_all_factors/try_loop的except KeyError正确捕获。
    """
    func = dummy_registry["B"]  # 需要参数A，既不在param_pool也没有默认值
    with pytest.raises(KeyError):
        call_single_factor(func, {})


def test_dependency_chain_resolves_across_retry_rounds(dummy_registry):
    """模拟真实的 calculate_all_factors + try_loop 依赖重试逻辑：
    B依赖A的结果，第一轮A成功、B失败，重试后B应该成功。
    """
    param_pool = {"base_value": 5}
    completed = {}
    failures = {}

    for name, func in dummy_registry.items():
        if name == "with_default":
            continue  # 这个用例不测这个因子
        try:
            completed[name] = call_single_factor(func, param_pool)
        except KeyError:
            failures[name] = True

    assert "A" in completed
    assert completed["A"] == 10
    assert "B" in failures  # 第一轮B应该失败，因为A的结果还没进completed

    # 模拟重试：把completed合并进param_pool再试一次
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
