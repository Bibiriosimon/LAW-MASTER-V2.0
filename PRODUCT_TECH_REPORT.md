# LAW MASTER v1.0 — 产品技术报告

> 生成日期：2026-02-09  
> 覆盖范围：`Law-Master-main/` 主工程  
> 核心目标：本地可调试的法律咨询 Agent（RAG + Web Search + 文书生成 + 思维导图）

---

## 1. 产品概述

LAW MASTER v1.0 是一个 **本地可交互的法律咨询智能体系统**，强调：
1. 法条与案例双知识库检索
2. 规划型 Agent（Reasoner 规划 + Chat 执行）
3. 支持网页搜索补充证据
4. 支持文书生成与 PDF 下载
5. 思维导图可视化与过程透明

主要运行方式为本地 FastAPI + Web UI，便于调试与快速迭代。

---

## 2. 技术架构总览

系统由四层构成：
1. **UI 层**：Web 单页应用，负责输入/输出/状态展示
2. **Agent 层**：Reasoner 规划，Chat 执行
3. **检索层**：双库 RAG + 可选网页搜索
4. **生成层**：答案生成 + 文书生成 + 思维导图生成

核心入口：`app_fastapi.py`  
核心前端：`web/index.html` + `web/assets/app.js` + `web/assets/styles.css`

---

## 3. 运行与部署

启动：
```
python -m uvicorn app_fastapi:app --host 127.0.0.1 --port 7860 --reload
```

默认模型：
- Chat：`deepseek-chat`
- Reasoner：`deepseek-reasoner`
- API Base：`https://api.deepseek.com/v1`

环境变量：
- `DEEPSEEK_API_KEY`
- `DEEPSEEK_BASE_URL`（可选）
- `DEEPSEEK_MODEL`（可选）

---

## 4. 数据与知识库

知识库文件：
- 法条库：`corpus_enriched_fast.jsonl`
- 案例库：`case_kb.jsonl`
- 扩展知识库：`corpus_extension.jsonl`（运行时追加）

案例字段（常见）：
- `title` / `case_title`
- `content` / `text`
- `source`
- `page_start` / `page_end`

索引策略：
1. TF-IDF 语义检索（`scikit-learn`）
2. 关键词命中打分（支持 title boost）
3. 权重融合：`keyword_weight` + `vector_weight`
4. 索引缓存：`cache/tfidf_*.joblib`

---

## 5. Agent 流程

### 5.1 模式切换
UI 提供“自我意识模式”开关：
1. 关闭：`/api/chat_stream`（简化 RAG）
2. 开启：`/api/agent_stream`（完整 Agent 规划流程）

### 5.2 规划与澄清
Reasoner 负责：
- 是否需要澄清问题
- 产出探索计划
- 指定 `action`（rag/web_search/generate_document）
- 生成网页搜索关键词

澄清策略：
1. 快速澄清（进入规划前）
2. 深度澄清（Reasoner 生成）

### 5.3 非法律问题兜底
若命中非法律问题且未命中法律关键词：
1. 直接用 Chat 模型输出生活建议
2. 不进入规划/检索流程

---

## 6. RAG 检索

双库结构：
1. 法条库检索（law_keywords）
2. 案例库检索（case_keywords）

关键词生成（单次调用）：
- `law_keywords`：法条检索
- `case_keywords`：案例检索
- `web_queries`：网页搜索
- `vector_query`：向量检索

输出：
1. 生成答案
2. 思维导图摘要
3. 文书生成依据
4. 右侧“检索详情”展示

---

## 7. 网页搜索

搜索组件：
- `ddgs`（DuckDuckGo search）
- `fetch_page_text` + `readability` + `BeautifulSoup` 解析正文

流程：
1. 生成关键词
2. 搜索
3. 抓取正文（可二次深度抓取）
4. 评估不足则改写关键词重试

诊断信息：
`/api/search_diagnostics` 返回：
- 引擎类型
- 命中数量
- 正文数量
- 查询列表

---

## 8. 最终回答生成

模型：DeepSeek Chat  
输入包括：
- 法条上下文
- 案例上下文
- 网页搜索摘要
- 文件/OCR内容
- 稀疏记忆

回答结构：
1. 结论
2. 法条依据（含通俗解释）
3. 分析（事实 → 条文要件 → 结论）
4. 建议
5. 风险提示

---

## 9. 文书生成

触发条件：
1. 用户明确要求
2. Reasoner 判定 `action=generate_document`

生成特点：
1. 模板骨架 + RAG 依据
2. 必须说明事实与法条要件的对应关系
3. 支持插入图片证据 `[[IMAGE:asset_id]]`
4. 输出 PDF（ReportLab）

文书预览：
1. 顶部提示补充个人信息
2. 强调隐私保护
3. 支持 AI 微调

---

## 10. 思维导图

生成时机：
1. 最终答案生成之后

数据来源：
1. 目标分析
2. 计划步骤
3. 法条摘要
4. 案例摘要
5. 网页摘要
6. 用户事实陈述

特点：
1. SVG 可拖拽/缩放
2. 节点点击弹窗查看详情
3. 支持关键词高亮

---

## 11. 案例预览增强

特性：
1. 右侧“案例搜索”编号：案例1/案例2/案例3
2. 输出中的 `【案例N】` 可点击定位高亮
3. 预览页顶部显示来源/页码 + Logo
4. 末尾 AI 对比分析（并行生成）

新增接口：
`POST /api/case_analysis`  
功能：并行生成案例与用户事件的相似点与可借鉴点

---

## 12. OCR 与文件解析

文件解析：
- `.txt/.md/.json/.jsonl`：直接读取
- `.pdf`：`pypdf`
- `.docx`：`python-docx`

OCR：
1. 主 UI 使用浏览器端 `Tesseract.js`
2. 后端保留 Paddle/Tesseract 兼容接口（当前默认浏览器）

---

## 13. 进度与状态机制

流式消息类型：
- `status`：阶段状态
- `trace`：过程日志
- `progress`：进度条
- `delta`：流式文本
- `meta`：最终汇总

进度分配：
1. 10%：规划完成
2. 30%：知识库检索
3. 35%~70%：网页搜索抓取
4. 70%~90%：答案生成
5. 90%~100%：思维导图生成

完成提示：
状态栏显示绿色 “任务完成！请查收”

---

## 14. API 列表

1. `GET /`：UI 入口
2. `POST /api/agent_stream`：完整 Agent 流式
3. `POST /api/chat_stream`：简化 RAG 流式
4. `POST /api/chat`：非流式
5. `POST /api/parse_file`：文件解析/OCR
6. `POST /api/transcribe`：语音转写
7. `POST /api/generate_document`：文书生成
8. `POST /api/doc_refine`：文书微调
9. `POST /api/case_analysis`：案例相似分析
10. `GET /api/ocr_info`：OCR 环境信息
11. `GET /api/search_diagnostics`：网页搜索诊断

---

## 15. 依赖与关键库

运行依赖（节选）：
- `fastapi`, `uvicorn`
- `requests`, `beautifulsoup4`, `readability-lxml`
- `ddgs`
- `jieba`, `scikit-learn`
- `pypdf`, `python-docx`
- `reportlab`
- `faster-whisper`（可选）
- `paddleocr`, `paddlepaddle`（可选）

详见：`requirements.txt`

---

## 16. 日志与诊断

日志输出：
- `logs/app_fastapi.log`

搜索诊断：
```json
{
  "engine": "ddgs",
  "queries": [],
  "total_results": 0,
  "content_hits": 0
}
```

---

## 17. 已知边界与改进方向

1. Web 搜索稳定性依赖站点反爬策略
2. 案例库内容质量影响摘要精度
3. OCR 更适合配合结构化文本
4. 思维导图布局可进一步智能避让

---

## 18. 关键文件索引

1. `app_fastapi.py`：主后端与 Agent 逻辑  
2. `web/index.html`：UI 布局  
3. `web/assets/app.js`：交互逻辑  
4. `web/assets/styles.css`：样式  
5. `corpus_enriched_fast.jsonl`：法条库  
6. `case_kb.jsonl`：案例库  
7. `exports/`：文书 PDF 输出目录  
8. `cache/`：索引、会话、OCR 资产缓存  

---

## 19. 总结

LAW MASTER v1.0 已形成完整的“法律咨询 Agent 栈”：
1. 规划  
2. 双库检索  
3. 网页搜索补充  
4. 结构化回答  
5. 文书生成  
6. 可视化导图  

系统具备持续扩展能力，可继续引入更多知识库、向量数据库以及更强的审校模型。

