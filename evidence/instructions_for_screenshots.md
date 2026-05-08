Hướng dẫn thu thập ảnh chụp màn hình để nộp:

1) LangSmith Traces (Runs/Traces)
   - Mở https://smith.langchain.com → project `day22-langsmith-lab` → tab `Runs` hoặc `Traces`.
   - Chụp màn hình danh sách traces (nhiều entry `rag-query`) và lưu tên `langsmith_traces.png` trong thư mục `evidence/`.

2) LangSmith Prompt Hub
   - Mở `Prompts` trong LangSmith, đảm bảo `day22-rag-prompt-v1` và `day22-rag-prompt-v2` hiển thị.
   - Chụp màn hình và lưu là `langsmith_prompts.png` trong `evidence/`.

3) RAGAS report
   - File `evidence/ragas_report.json` đã có sẵn.

4) Console output
   - Nếu muốn, bạn có thể lưu đầu ra terminal khi chạy `pseudocode/03_ragas_evaluation.py` sang `evidence/console_output.txt`.

Khi đã có ảnh, tôi sẽ đóng gói `evidence/` thành file nộp (zip) nếu bạn muốn.
