from .config import CONFIG_REGISTRY
import yaml

def load_configs(yaml_path: str) -> dict:
    """
    读取YAML文件，返回 {config_name: Config实例} 的字典。
    YAML里没有出现的顶层key，对应的Config类会用全部默认值创建。
    """
    with open(yaml_path, 'r') as f:
        raw = yaml.safe_load(f) or {}  # 如果文件是空的，safe_load返回None，用 or {} 兜底

    result = {}
    for config_name, config_class in CONFIG_REGISTRY.items():
        kwargs = raw.get(config_name, {})  # YAML里没有这个key，就用空字典（意味着全部用默认值）
        result[config_name] = config_class(**kwargs)  # 你来写这一行，回忆之前已经确认过的模式
    return result