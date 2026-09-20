from __future__ import annotations

import os
import sys


def main() -> None:
    try:
        import uvicorn
    except ImportError:
        print("缺少依赖：请先执行 `pip install -r requirements.txt`")
        sys.exit(1)

    if not (os.getenv("LLM_API_KEY") or os.getenv("DASHSCOPE_API_KEY")):
        print("未检测到 API Key。")
        print("请先设置环境变量：")
        print('  PowerShell: $env:LLM_API_KEY="你的通义API Key"')
        sys.exit(1)

    print("启动中: http://127.0.0.1:8080")
    uvicorn.run("backend.main:app", host="127.0.0.1", port=8080, reload=False)


if __name__ == "__main__":
    main()
