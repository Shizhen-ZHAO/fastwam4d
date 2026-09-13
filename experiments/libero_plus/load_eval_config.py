#!/usr/bin/env python
"""把 LIBERO-plus eval 的 YAML 配置转成 export 语句,供 run_eval.sh eval。

用法:
    load_eval_config.py <默认配置.yaml> [用户配置.yaml]

规则:
- 环境变量优先: 已存在于环境中的 key 不输出(不覆盖);
- 用户配置覆盖默认配置;用户配置里出现默认配置没有的 key 直接报错(防笔误);
- 值为 null 的 key 跳过(交给 shell 侧的派生默认值);
- bool 统一输出小写 true/false,其余值 str() 后 shlex.quote。
"""
import os
import re
import shlex
import sys

import yaml

KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def load(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise SystemExit(f"[libero-plus] 配置文件格式错误(应为 key: value 映射): {path}")
    return data


def main() -> None:
    default_path = sys.argv[1]
    user_path = sys.argv[2] if len(sys.argv) > 2 else None

    merged = load(default_path)
    if user_path:
        if not os.path.isfile(user_path):
            raise SystemExit(f"[libero-plus] 配置文件不存在: {user_path}")
        user = load(user_path)
        unknown = sorted(set(user) - set(merged))
        if unknown:
            raise SystemExit(
                f"[libero-plus] {user_path} 含未知配置项: {', '.join(map(str, unknown))}\n"
                f"[libero-plus] 可用配置项见默认配置: {default_path}"
            )
        merged.update(user)

    for key, value in merged.items():
        if value is None or key in os.environ:
            continue
        key = str(key)
        if not KEY_RE.match(key):
            raise SystemExit(f"[libero-plus] 非法配置项名称: {key!r}")
        if isinstance(value, bool):
            value = "true" if value else "false"
        print(f"export {key}={shlex.quote(str(value))}")


if __name__ == "__main__":
    main()
