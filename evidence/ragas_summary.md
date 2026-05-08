**RAGAS Evaluation Summary (copied)

- See `ragas_report.json` in this folder for full report.

Prompt V1 (summary):
- faithfulness: 0.78754
- answer_relevancy: 0.87189
- context_recall: 0.63
- context_precision: 0.97333

Prompt V2 (summary):
- faithfulness: 0.47367
- answer_relevancy: 0.84674
- context_recall: 0.63
- context_precision: 0.97

Target: faithfulness >= 0.8 — NOT MET

Screenshots to include (place images into this folder):
- `langsmith_traces.png` — LangSmith Traces view showing many rag-query entries
- `langsmith_prompts.png` — Prompt Hub showing `day22-rag-prompt-v1` and `day22-rag-prompt-v2`

Command to regenerate report:
```powershell
python pseudocode/03_ragas_evaluation.py
```
