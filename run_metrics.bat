@echo off
cd /d E:\paper_proposal\week_10\secure_codgen

python src\core\metrics.py ^
    outputs\code\securityeval_direct\securityeval_direct.jsonl ^
    outputs\code\sallm_direct\sallm_direct.jsonl ^
    outputs\code\securityeval_planning\securityeval_planning.jsonl ^
    outputs\code\sallm_planning\sallm_planning.jsonl ^
    --csv results.csv

echo.
echo ============================
echo Metrics calculation finished
echo Results saved to results.csv
echo ============================
pause
