# 项目测试结果

这里保存程序测试的技术结果；不包含开发者与编码助手的交谈，也不生成公司提交报告。

- [离线回归结果](offline_regression.json)：由实际 pytest 执行产生，记录每项测试的状态、测试文件哈希和软件版本。使用测试替身及内存 HTTP transport，不消耗模型调用。
- [50 例输入预检](final_50_preflight.json)：逐例记录输入可用状态；`provider_calls=0`。`ready` 只说明输入可以送入模型，全部真实输出目前为 `not_run`。
- [保留的历史模型响应](live_api_results.json)：从本地实际 run 提取模型响应 ID、用量、哈希和完整性校验结果。该次模型返回了三条候选，原应用校验状态为 `OUTPUT_INVALID`；不将原始状态改记为成功。
- [Gemini Developer API 状态](gemini_api_status.json)：分别记录模型列表可发现状态、真实生成与候选校验；模型查询不计为成功生成，未执行的步骤保持 `not_run`。

本次三条真实模型原文见[Gemini 价格异议测试](../../evidence/simulated_conversations/gemini_core_price_objection_2026-11-08.json)：一次真实调用、HTTP 200、程序校验通过，人工审阅待完成。另保留[历史价格异议测试](../../evidence/simulated_conversations/core_price_objection_2026-11-03.json)的原始失败状态。候选是备选回复，没有实际发送给客户。

当前默认后端为 AI Studio key 对应的 Gemini Developer API，模型为 `gemini-3.1-flash-lite`，原生根地址为 `https://generativelanguage.googleapis.com/v1beta`。项目与 API key 已关联，程序不需要 Cloud 项目／区域／ADC 配置。本次模型发现与单例真实生成均已通过；最终 50 例仍未生成，以 Gemini 状态 JSON 和逐例结果为准。

离线测试通过和输入 `ready` 均不能解释为已取得 150 条真实输出。官方 [Gemini 定价](https://ai.google.dev/gemini-api/docs/pricing) 将选定模型的标准 Free Tier 输入／输出列为免费；项目实际可用模型及额度仍需在 AI Studio 核对，见 [计费与层级说明](https://ai.google.dev/gemini-api/docs/billing)。

复核命令（前两条不调用 API）：

```powershell
.\.venv\Scripts\python.exe -X utf8 -m pytest -q -m "not live"
.\.venv\Scripts\python.exe -X utf8 -m ca_tool.batch --input tests/data/final_50.jsonl --preflight
.\.venv\Scripts\python.exe -X utf8 -m ca_tool.verify_run --run-dir runs/REPLACE_WITH_RUN_ID --expect-live
```

完整原始 run 与 JUnit 输出只保存在本机；提交的 JSON 是技术结果的有限字段，不包含本机主目录、密钥、开发聊天或调试正文。哈希用于核对本地来源，不是供应商签名。继续真实生成前，按主 README 在本机 `.env` 配置 `GEMINI_API_KEY` 并显式设置调用上限，完整 key 不进入聊天或 Git。
