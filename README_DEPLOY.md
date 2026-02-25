# LAW MASTER v1.0 快速部署

## 1. 环境准备
- Python 3.10+（建议 3.10/3.11）
- Windows PowerShell

## 2. 安装依赖
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 3. 设置环境变量
```powershell
$env:DEEPSEEK_API_KEY="你的KEY"
```
可选：
```powershell
$env:DEEPSEEK_BASE_URL="https://api.deepseek.com/v1"
$env:DEEPSEEK_MODEL="deepseek-chat"
```

## 4. 启动服务
```powershell
python -m uvicorn app_fastapi:app --host 127.0.0.1 --port 7860 --reload
```
浏览器访问：`http://127.0.0.1:7860`

## 5. 关键文件
- app_fastapi.py：主后端
- web/：前端 UI
- corpus_enriched_fast.jsonl：法条库
- case_kb.jsonl：案例库
- corpus_extension.jsonl（如存在）：扩展知识库

## 6. 常见问题
- 首次加载较慢：需建立索引
- API 401：检查 Key 是否有效
- 网页搜索无结果：可切换站点区域
