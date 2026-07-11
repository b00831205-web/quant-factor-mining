"""YAML 配置层测试(由 quantmine/test.py 的验证脚本迁移而来)。

覆盖:
1. YAML 提到的字段正确覆盖默认值
2. YAML 没提到的 key/字段用 dataclass 默认值
3. __post_init__ 校验拦截非法值
4. periods 传 int 时归一化为 list(类型声明接受 int|list)
5. 未知 top-level key 的行为(当前语义: 静默忽略——已知限制, 拼错key不会报错)
"""
import pytest

from quantmine.config import CONFIG_REGISTRY, ForwardReturnConfig, OrthogonalizeConfig
from quantmine.load_config import load_configs


@pytest.fixture
def yaml_file(tmp_path):
    def _write(content: str):
        p = tmp_path / "config.yaml"
        p.write_text(content, encoding="utf-8")
        return str(p)
    return _write


def test_yaml_values_override_defaults(yaml_file):
    path = yaml_file("""
orthogonalize:
  threshold: 0.05
backtest:
  part: 10
""")
    configs = load_configs(path)
    assert configs["orthogonalize"].threshold == 0.05
    assert configs["backtest"].part == 10


def test_missing_fields_fall_back_to_defaults(yaml_file):
    path = yaml_file("orthogonalize:\n  threshold: 0.05\n")
    configs = load_configs(path)
    assert configs["orthogonalize"].min_period == 60   # 字段级默认
    assert configs["momentum"].day == 5                # 整个key缺失, 全默认


def test_empty_yaml_returns_all_defaults(yaml_file):
    configs = load_configs(yaml_file(""))
    assert set(configs.keys()) == set(CONFIG_REGISTRY.keys())
    assert configs["newey_west"].lag_multiplier == 2


def test_validation_rejects_illegal_values():
    with pytest.raises(ValueError):
        OrthogonalizeConfig(threshold=1.5)      # 超出(0,1)
    with pytest.raises(ValueError):
        ForwardReturnConfig(periods=[0])        # period必须>=1


def test_validation_triggered_through_yaml(yaml_file):
    path = yaml_file("transaction_cost:\n  cost_per_trade: 2.0\n")
    with pytest.raises(ValueError):
        load_configs(path)


def test_int_periods_normalized_to_list():
    cfg = ForwardReturnConfig(periods=20)
    assert cfg.periods == [20]


def test_unknown_top_level_key_silently_ignored(yaml_file):
    """当前语义: YAML里拼错的key不报错(如 newey_west_ 多个下划线),
    对应配置静默使用默认值。这是已知限制——若改成报错, 请更新本测试。"""
    path = yaml_file("newey_west_typo:\n  lag_multiplier: 99\n")
    configs = load_configs(path)
    assert configs["newey_west"].lag_multiplier == 2


def test_unknown_field_raises_type_error(yaml_file):
    """已知key下的未知字段会raise(dataclass拒绝多余kwargs), 拼错字段名能被发现。"""
    path = yaml_file("orthogonalize:\n  threshhold: 0.05\n")
    with pytest.raises(TypeError):
        load_configs(path)
