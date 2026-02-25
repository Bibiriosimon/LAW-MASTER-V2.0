# LAW MASTER V1.0 ? Cost Evaluation Report (USD)

> Scope: `E:\LAW MASTER\LAW_MASTER V1.0`
> 
> Goal: Provide a clear framework for **LLM usage cost + server cost + maintenance cost**, with example calculations based on your inputs.
> 
> Date: 2026-02-12

---

## 1. Business Inputs (Provided)

- **Model combo**: 1 call to `deepseek-chat` + 1 call to `deepseek-reasoner` per request
- **MAU**: 10,000
- **Calls per user / month**: 5
- **Agent mode share**: 60%
- **Web search rate**: 75% (agent mode always on)
- **OCR usage**: high (almost every request)
- **Speech usage**: ~40%
- **Server**: ~2 RMB/hour
- **Team**: 6 devs, 3 maintainers

> Note: If speech uses a paid STT/TTS API, add that cost. OCR is browser-side (Tesseract.js), so **no server OCR cost**.

---

## 2. DeepSeek Pricing (Provided)

**deepseek-chat**
- Input cache hit: $0.07 / 1M tokens
- Input cache miss: $0.27 / 1M tokens
- Output: $1.10 / 1M tokens

**deepseek-reasoner**
- Input cache hit: $0.14 / 1M tokens
- Input cache miss: $0.55 / 1M tokens
- Output: $2.19 / 1M tokens

---

## 3. Token Assumptions (Replaceable)

No measured token stats were provided, so we use a **conservative default**:

- **deepseek-chat**: input 3,000 tokens / output 1,200 tokens
- **deepseek-reasoner**: input 1,500 tokens / output 600 tokens

> Web search + OCR typically increases input tokens, hence the higher input side.

---

## 4. Per-Call Cost Formula

Let cache hit rate be `h`:

```
chat_in_cost = (h*0.07 + (1-h)*0.27) * chat_in_tokens / 1e6
chat_out_cost = 1.10 * chat_out_tokens / 1e6
reason_in_cost = (h*0.14 + (1-h)*0.55) * reason_in_tokens / 1e6
reason_out_cost = 2.19 * reason_out_tokens / 1e6

per_call_cost = chat_in_cost + chat_out_cost + reason_in_cost + reason_out_cost
```

---

## 5. LLM Cost (10k MAU / 5 calls per user)

**Monthly calls** = 10,000 ? 5 = **50,000 calls/month**

| Cache hit rate | Cost per call (USD) | Monthly LLM cost (USD) |
|---------------|----------------------|-------------------------|
| 0%            | 0.004269             | 213.45                  |
| 30%           | 0.003904             | 195.22                  |
| 60%           | 0.003540             | 177.00                  |

> **Suggested baseline**: cache hit 30% ? **~$195/month**

---

## 6. LLM Cost at 100k Users

**Monthly calls** = 100,000 ? 5 = **500,000 calls/month**

| Cache hit rate | Cost per call (USD) | Monthly LLM cost (USD) |
|---------------|----------------------|-------------------------|
| 0%            | 0.004269             | 2,134.50                |
| 30%           | 0.003904             | 1,952.25                |
| 60%           | 0.003540             | 1,770.00                |

---

## 7. Server Cost (Cloud)

Given **2 RMB/hour**, using an example rate 1 USD = 6.9087 CNY:

- Per hour: **$0.2895**
- Per month (730 hours): **~$211.33**

> Provide the exact exchange rate you want, and I can update this.

---

## 8. Total Monthly Cost (Example)

Using **30% cache hit**:

### 10k MAU
- LLM: $195.23
- Server: $211.33
- **Total: ~$406.56 / month**

### 100k MAU
- LLM: $1,952.25
- Server: $211.33
- **Total: ~$2,163.58 / month**

> Server costs will rise with capacity and HA requirements.

---

## 9. Development & Maintenance Labor (Pending)

You provided team size but not salary. Use:

```
Dev cost = 6 ? monthly fully-loaded cost ? dev months
Ops cost = 3 ? monthly fully-loaded cost ? ops months
```

If you provide **monthly fully-loaded cost per person**, I can fill this in.

---

## 10. Inputs Still Needed for Precision

1. Real average **input/output tokens per request**
2. Real **cache hit rate**
3. **Speech API cost** (if any)
4. **Fully-loaded labor cost** per person
5. **Server scaling plan** for 100k users

---

## 11. Summary

- At 10k MAU & 5 calls/user: **LLM ~$195/month**, server ~$211/month.
- At 100k MAU: **LLM ~$1,952/month** plus higher infra.
- Labor & speech costs remain the biggest unknowns.

If you share real token stats and labor cost, I will deliver a final precise report.
