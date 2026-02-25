# LAW MASTER v2.0

法律咨询智能体（法条库 + 案例库 + 网页搜索 + 文书生成 + 思维导图）。

## 这版（V2.0）已完成的改动
- Logo 替换为 `Master Aniya.jpg`（前端与 PDF 页眉统一使用）。
- 回答里的 `【网页1】`、`【网页2】` 支持点击跳转到右侧“网页搜索结果”并高亮。
- 案例 AI 分析与思维导图并行执行，避免只在点击案例预览时才开始分析。

## 目录说明
- `app_fastapi.py`：后端主服务（FastAPI）
- `web/index.html`：前端页面
- `web/assets/app.js`：前端逻辑
- `web/assets/styles.css`：样式
- `corpus_enriched_fast.jsonl`：法条知识库
- `case_kb.jsonl`：案例知识库
- `requirements.txt`：Python 依赖

## 本地启动
```powershell
cd "E:\LAW MASTER\LAW_MASTER V2.0"
python -m pip install -r requirements.txt
python -m uvicorn app_fastapi:app --host 127.0.0.1 --port 9998 --reload
```

浏览器打开：
- `http://127.0.0.1:9998`

## 环境变量
建议在系统或终端中设置：
- `DEEPSEEK_API_KEY`
- （可选）`DEEPSEEK_BASE_URL`
- （可选）`DEEPSEEK_MODEL`

`.env.example` 只保留模板，不要提交真实 key。

## GitHub 需要上传哪些文件
建议上传（最小可运行）：
- `app_fastapi.py`
- `requirements.txt`
- `web/`
- `corpus_enriched_fast.jsonl`
- `case_kb.jsonl`
- `.env.example`
- `README.md`
- `AGENT_ARCHITECTURE.md`（可选）
- 其他技术文档（可选）

不要上传：
- `.env`（真实密钥）
- `cache/`、`logs/`、`exports/`、`__pycache__/`
- 本地虚拟环境 `.venv/`
- 任何 API key、证书、个人隐私数据

## Cloudflare Tunnel（可选临时公网）
```powershell
cloudflared tunnel --protocol http2 --url http://127.0.0.1:9998
```

## 备注
- 若知识库文件过大（>100MB），请使用 Git LFS。
- 如需上线正式环境，建议增加 SQL 存储、鉴权与审计日志。
