<p align="center">
  <img src="./Master%20Aniya.jpg" alt="LAW MASTER Logo" width="180" />
</p>
<p align="center">
  <img src="./law%20aniya.png" alt="LAW ANIYA" width="220" />
</p>

<h1 align="center">LAW MASTER v2.0</h1>

<p align="center">
  法律智能体平台：法条检索 + 案例检索 + 网页搜索 + 计划推理 + 文书生成 + 思维导图
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB" alt="Python 3.10+" />
  <img src="https://img.shields.io/badge/FastAPI-Backend-009688" alt="FastAPI" />
  <img src="https://img.shields.io/badge/Frontend-HTML%2FCSS%2FJS-1E3A8A" alt="Frontend" />
  <img src="https://img.shields.io/badge/Model-DeepSeek-0F172A" alt="DeepSeek" />
</p>

## 项目简介
LAW MASTER v2.0 是一个面向法律场景的智能体应用。  
核心目标是把「用户问题 -> 信息澄清 -> 法条/案例/网页证据 -> 可执行建议 -> 文书输出」打成一条完整链路。

支持两种使用形态：
- Web 端（当前主版本）：`web/` 前端 + `FastAPI` 后端。
- 微信小程序（进行中）：`miniapp_wechat/`，已完成双模式登录与会话系统首版。

## 核心能力
- 智能体流程：规划、澄清、检索、总结、输出。
- 双知识库检索：法条库 + 案例库（本地语料索引）。
- 网页搜索补充：按问题动态决定是否联网补证。
- 流式交互：实时状态、进度、过程追踪、增量回复。
- 文书助手：申诉书/投诉信/律师函/起诉状/通用文书生成与微调。
- 思维导图：结合检索与结论自动生成结构化导图。
- 会话能力：新建会话、历史记录、重命名、收藏、删除、自动标题。

## V2.0 已完成更新
- Logo 统一：前端展示与 PDF 页眉统一使用 `Master Aniya.jpg`。
- 右侧面板改造：连接设置/检索参数/错误信息收纳到“更多...”折叠区。
- 会话交互增强：新聊天、我的聊天、当前会话标题面板。
- 自动命名优化：会话首轮回复后并行生成标题，替换默认“新聊天”。
- 链接可跳转：回复中的 `【网页N】`、`【案例N】` 可点击定位明细。
- 性能优化：案例分析与思维导图并行执行。

## 技术架构
```text
User
  -> Web UI / Mini Program
      -> FastAPI (app_fastapi.py)
          -> Law Corpus Index
          -> Case Corpus Index
          -> Web Search
          -> DeepSeek API
          -> PDF Export
          -> Local Cache & Session State
```

## 目录结构
```text
LAW_MASTER V2.0/
├─ app_fastapi.py                 # FastAPI 主服务
├─ web/
│  ├─ index.html                  # Web 端页面
│  └─ assets/
│     ├─ app.js                   # 前端逻辑
│     └─ styles.css               # 样式
├─ miniapp_wechat/                # 微信小程序（首版骨架）
├─ corpus_enriched_fast.jsonl     # 法条知识库
├─ case_kb.jsonl                  # 案例知识库
├─ Master Aniya.jpg               # 项目 Logo
├─ requirements.txt               # Python 依赖
└─ README_DEPLOY.md               # 部署补充说明
```

## 本地快速启动
### 1. 安装依赖
```powershell
cd "E:\LAW MASTER\LAW_MASTER V2.0"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 2. 配置环境变量
```powershell
$env:DEEPSEEK_API_KEY="你的Key"
```
可选：
```powershell
$env:DEEPSEEK_BASE_URL="https://api.deepseek.com/v1"
$env:DEEPSEEK_MODEL="deepseek-chat"
```

### 3. 启动服务
```powershell
python -m uvicorn app_fastapi:app --host 127.0.0.1 --port 9998 --reload
```

访问：
- `http://127.0.0.1:9998`

## API 入口（核心）
- `POST /api/agent_stream`：智能体主流程（推荐）。
- `POST /api/chat_stream`：基础流式聊天。
- `POST /api/chat`：非流式聊天兜底。
- `POST /api/session_title`：会话标题生成。
- `POST /api/generate_document`：文书生成/PDF导出。
- `POST /api/doc_refine`：文书润色。
- `POST /api/parse_file`：文件解析。
- `POST /api/transcribe`：音频转写。

小程序用户会话接口：
- `POST /api/mp/auth/guest`
- `POST /api/mp/auth/wx_login`
- `GET /api/mp/auth/me`
- `GET/POST/PATCH/DELETE /api/mp/sessions...`

## Render 部署提示
如果 Render 启动命令使用 `gunicorn ...`，请确保已安装 `gunicorn`；  
或者直接使用 `uvicorn` 启动命令：

```bash
python -m uvicorn app_fastapi:app --host 0.0.0.0 --port $PORT
```

另外需确认：
- 仓库包含 `web/` 目录（否则会出现 `Missing web/index.html`）。
- 服务已绑定 `$PORT`（否则会出现 `No open ports detected`）。

## 微信小程序（首版）
路径：[`miniapp_wechat/`](./miniapp_wechat)

已实现：
- 微信登录 + 游客登录双模式
- 用户独立会话存储
- 新建/切换/重命名/收藏/删除会话
- 流式聊天与非流式兜底

小程序后端需配置：
- `WECHAT_APPID`
- `WECHAT_APPSECRET`
- `DEEPSEEK_API_KEY`

## 安全与合规建议
- 不要提交真实 `.env` 或 API Key。
- 生产环境建议加：鉴权、限流、审计日志、成本告警。
- 涉及用户资料时，建议增加隐私协议与数据保留策略。

## 免责声明
本项目用于技术研究与产品原型验证，不构成正式法律意见。  
涉及重大权益争议时，请咨询持证律师。
