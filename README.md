# 岚序 LANXU · Luxury Client Advisor AI

为虚构高端皮具品牌生成中文微信话术的本地 Python 工具。输入客户资料、场景、对话历史及适用商品，输出三条不同的候选消息，由顾问审阅选择。

支持六个场景：生日／节日问候、商品推荐、购后关怀、沉睡客户唤醒、活动邀请、价格异议。工具不发送微信。

仓库维护项目代码、测试、配置、输入示例与技术说明。[测试结果](tests/results/README.md) 保存离线验证、输入预检和真实接口调用的状态；`evidence/` 只收录测试中虚构的顾问 agent 与客户的对话、模型候选回复及必要的来源标记。开发者与编码助手的交谈不纳入 Git。面向公司的报告和提交材料由使用者独立整理。

## 安装

已在 Windows / Python 3.13 验证。在项目根目录执行：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -X utf8 -m pytest -q -m "not live"
```

默认测试使用明确注入的测试替身或内存 HTTP transport，不调用付费 API。

## 配置 Gemini Developer API

默认使用 Google AI Studio 的 Gemini Developer API key，通过原生 `generateContent` API 调用 `gemini-3.1-flash-lite`。端点为 `https://generativelanguage.googleapis.com/v1beta/models/gemini-3.1-flash-lite:generateContent`。[官方 API 参考](https://ai.google.dev/api/generate-content)

第一次运行时复制配置文件；已有 `.env` 时直接编辑现有文件：

```powershell
Copy-Item .env.example .env
```

填写本机 `.env`：

```dotenv
LLM_PROVIDER=gemini
LLM_MODEL=gemini-3.1-flash-lite
GEMINI_API_KEY=
LLM_MAX_ATTEMPTS=1
LLM_MAX_TOTAL_CALLS=75
```

将 AI Studio 中的 key 填入本机 `GEMINI_API_KEY`。密钥字段在 `.env.example` 中保持空白；完整 key 只保存在 Git 忽略的 `.env`，不要粘贴到聊天、源码或测试结果。系统环境变量优先于 `.env`。

Gemini API key 已关联 AI Studio 的后台项目，程序使用 key 认证，不需要填写项目 ID、区域或执行 Cloud ADC 登录。key 的创建与项目关联见 [Gemini API key 说明](https://ai.google.dev/gemini-api/docs/api-key)。代码保留可显式选择的 Vertex 等其他适配器；默认 provider 为 `gemini`，不自动切换后端或模型。

### Free Tier、价格与调用边界

官方 [Gemini 定价](https://ai.google.dev/gemini-api/docs/pricing) 将 `gemini-3.1-flash-lite` 标准调用的 Free Tier 输入／输出列为免费；付费层文本输入为 **USD 0.25／百万 token**、输出为 **USD 1.50／百万 token**，输出计价包含思考 token。选用轻量模型控制开发成本，话术质量仍需实际测试。

Free Tier 只开放部分模型并受限额约束。批量运行前，在 AI Studio 查看当前项目的使用层级、可用模型及额度；不假定固定 RPM／RPD，也不因模型出现在列表中就认定生成权限或剩余额度已通过。依据见 [计费与使用层级](https://ai.google.dev/gemini-api/docs/billing) 和 [限额说明](https://ai.google.dev/gemini-api/docs/rate-limits)。

`LLM_MAX_ATTEMPTS` 控制每个请求的最大调用次数，范围 1—3；网络重试与输出修复共享此上限。`LLM_MAX_TOTAL_CALLS` 是每次批量执行的调用次数上限，不是美元金额或跨命令的总预算。单次调用默认超时 30 秒、请求总时间预算 90 秒，可在 `.env` 调整。

已完成一次 Gemini Developer API 真实实测：模型列表查询及生成请求均返回 HTTP 200；单次调用生成三条中文候选，程序校验通过，人工表达与语义审阅仍待完成。原文及来源见[Gemini 价格异议测试](evidence/simulated_conversations/gemini_core_price_objection_2026-11-08.json)。50 例当前仅完成输入预检，真实生成保持 `not_run`。另保留一次原校验状态为 `OUTPUT_INVALID` 的历史模型响应。最新状态见[测试结果](tests/results/README.md)。

## 单次生成

先校验输入与场景事实，无需 API：

```powershell
.\.venv\Scripts\python.exe -X utf8 -m ca_tool.generate --input examples/core_price_objection.json --preflight
```

实际生成会调用模型：

```powershell
.\.venv\Scripts\python.exe -X utf8 -m ca_tool.generate --input examples/core_price_objection.json --runs-dir runs
```

成功结果包含 `status=success`、业务 `request_id`、唯一 `run_id`、场景、三个 `candidates`，以及独立的 `warnings` 和 `metadata`。候选 ID 固定为 `option_1`、`option_2`、`option_3`，正文必须含中文、互不重复，每条不超过 160 个 Unicode 字符。

失败时 `candidates=[]`，通过 `error.code`、`message`、`missing_fields`、`retryable` 说明原因。不会用模板填充失败结果。成功表示结构、可检查业务规则与留档通过；语义和表达仍须顾问复核。

| 场景 | 输入示例 | 关键事实 |
| --- | --- | --- |
| 生日／节日问候 | [greeting.json](examples/greeting.json) | 场合类型、名称、日期 |
| 商品推荐 | [product_recommendation.json](examples/product_recommendation.json) | 至少一个已知商品 |
| 购后关怀 | [post_purchase_care.json](examples/post_purchase_care.json) | 已购商品引用、购买时间 |
| 沉睡唤醒 | [reactivation.json](examples/reactivation.json) | 上次互动或购买时间 |
| 活动邀请 | [event_invitation.json](examples/event_invitation.json) | 活动名称、未来时间、地点、预约方式 |
| 价格异议 | [core_price_objection.json](examples/core_price_objection.json) | 目标商品、正式价格与币种、异议内容 |

新输入可以直接复制示例 JSON，修改客户、商品、对话和匹配的 `scenario_context`，无需修改代码。`reference_time` 必须带时区，是场景业务时间，与实际执行时间分别记录。

输入拒绝未知字段；最多 30 条历史、每条 1000 字符、10 个商品，默认完整输入上限 128 KiB。商品 ID 唯一，价格以 Decimal 处理并与币种共同提供。

## 批量处理

项目包含独立的 12 例开发输入、50 例固定测试输入和 16 例健壮性输入。用以下命令做无 API 的前置校验：

```powershell
.\.venv\Scripts\python.exe -X utf8 -m ca_tool.batch --input tests/data/final_50.jsonl --preflight
```

实际批量生成并导出 CSV／JSONL：

```powershell
.\.venv\Scripts\python.exe -X utf8 -m ca_tool.batch --input tests/data/final_50.jsonl --runs-dir runs --output artifacts/test_cases.csv --max-total-calls 75
```

显式 `--max-total-calls` 优先于 `.env`。批次在每个案例完成后保存断点，保留失败和未执行案例，预算耗尽后停止新增调用。中断后可传入对应的 checkpoint：

```powershell
.\.venv\Scripts\python.exe -X utf8 -m ca_tool.batch --input tests/data/final_50.jsonl --output artifacts/test_cases.csv --max-total-calls 75 --resume-manifest artifacts/REPLACE_WITH_CHECKPOINT.json
```

从实际生成的文件名替换占位符。只有输入哈希、源码、配置、模型与采样参数一致的真实成功记录才允许复用。`--prepare-only` 可以生成空输出工作表，不调用模型；空输出会明确标记为 pending。

## 留档与校验

每次执行保存到 `runs/<run_id>/`，包括完整输入、事实与提示词快照、每次供应商返回或错误、校验结果、版本、用量和文件哈希。相同 request_id 重跑不会覆盖历史。配置或输入尚未读取成功的 CLI 错误使用 `run_id=null`。

```powershell
.\.venv\Scripts\python.exe -X utf8 -m ca_tool.verify_run --run-dir runs/REPLACE_WITH_RUN_ID --expect-live
```

`--expect-live` 检查实际调用来源、输入／输出关联及文件完整性。哈希用于检测改动，不是供应商签名。密钥不进入运行快照。

单次生成退出码：0 成功，2 输入／配置／认证／联系规则阻止，3 模型或输出失败，4 存储失败。批次有失败或未完成结果时返回非零。

## 项目结构

```text
ca_tool/        输入模型、上下文、提示词、模型适配、生成、留档和批量工具
config/         版本化品牌规则
examples/       六场景可运行输入
tests/          离线回归与虚构测试数据
docs/           品牌假设和技术架构
scripts/        Git 暂存区凭证检查
evidence/       虚构顾问与客户的测试对话及模型候选回复
```

品牌为虚构的岚序 LANXU，政策来自 [brand.json](config/brand.json)。客户历史作为参考数据处理，不能覆盖本地品牌政策。拒绝联系会阻止生成；不承诺未经授权的折扣、库存、礼品或售后，不猜测客户年龄、生活变化或沉默原因。

详细业务约定见 [品牌假设](docs/assumptions.md)，数据流见 [架构图](docs/architecture.md)。`runs/`、`artifacts/`、`reports/`、`submission/`、`dist/` 和 `.local/` 均留在本地，不纳入 Git；`evidence/` 仅保留经过检查的虚构测试对话。
