# 测试说明

这套测试分两类，写法和运行方式不同：

## 第一类：单元测试（用合成/dummy数据，不依赖真实文件）
- test_market_data.py
- test_factor_registry.py
- test_calculate_all_factors.py

这些跑得快（几秒内完成），每次改代码后都应该跑，适合放进CI。

## 第二类：黄金值回归测试（依赖真实的旧数据文件做基准对比）
- test_factors_golden.py
- test_forward_returns_golden.py
- test_cs_ic_golden.py

这些需要真实的旧parquet文件路径，通过 conftest.py 里的
skipif 标记自动跳过（如果本地没有这些文件，比如在CI环境里，
不会导致测试失败，而是显示SKIPPED）。

## 运行方式

```bash
# 装依赖（如果还没装）
pip install pytest --break-system-packages

# 跑全部测试
cd quantmine
pytest tests/ -v

# 只跑快速的单元测试，跳过黄金值对比
pytest tests/ -v -m "not golden"

# 只跑某一个文件
pytest tests/test_factors_golden.py -v
```

## 后续维护规则

**每次改动因子计算逻辑（factor_mining.py）**：
跑一次 test_factors_golden.py，确认没有破坏已验证过的数值一致性。

**每次改动 call_single_factors / calculate_all_factors / try_loop**：
跑 test_factor_registry.py 和 test_calculate_all_factors.py，
这两个测试专门覆盖了这次调试中发现的具体bug
（比如 elif vs if 的顺序问题、ValueError vs KeyError 的异常类型问题）。

**新增因子时**：
在 test_factors_golden.py 里的 NAME_MAPPING 补充新因子对应的旧列名前缀
（如果有旧数据可对比的话），没有旧数据可对比就跳过golden测试，
只需要保证它能被正常注册和调用（走单元测试覆盖）。
