# Lab 16 Benchmark Report

## Metadata
- Dataset: hotpot_qa.json
- Mode: openai_api:react=gpt-3.5-turbo;reflexion=gpt-4o-mini
- Records: 210
- Agents: react, reflexion

## Summary
| Metric | ReAct | Reflexion | Delta |
|---|---:|---:|---:|
| EM | 0.7333 | 0.9905 | 0.2572 |
| Avg attempts | 1 | 1.0286 | 0.0286 |
| Avg token estimate | 327.4 | 349.04 | 21.64 |
| Avg latency (ms) | 1814.39 | 3146.5 | 1332.11 |

## Failure modes
```json
{
  "react": {
    "none": 77,
    "wrong_final_answer": 28
  },
  "reflexion": {
    "none": 104,
    "wrong_final_answer": 1
  },
  "overall": {
    "none": 181,
    "wrong_final_answer": 29
  }
}
```

## Extensions implemented
- structured_evaluator
- reflection_memory
- benchmark_report_json
- adaptive_max_attempts
- memory_compression

## Discussion
Reflexion helps when the first attempt stops after the first hop or drifts to a wrong second-hop entity. The tradeoff is higher attempts, token cost, and latency. In this run, we also enabled adaptive max attempts and reflection memory compression to reduce redundant retries while preserving useful strategy history.
