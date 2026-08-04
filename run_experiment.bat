@echo off
REM Launch 3 sharded workers, detached from any console session.
REM Safe to run repeatedly: each worker skips (series, detector, tier) combinations
REM already present in ANY results\full_results*.csv file, so nothing is recomputed.
REM Thread caps avoid CPU oversubscription across workers (24 cores / 3 workers).
cd /d "%~dp0"
set HF_HUB_DISABLE_XET=1
set KILLTEST_N_JOBS=7
set KILLTEST_TORCH_THREADS=7
set OMP_NUM_THREADS=7
set MKL_NUM_THREADS=7
start "" /B .venv\Scripts\python.exe -u -m killtest.run_full --datasets nasa,psm,smd,ucr --n-ucr 250 --tiers default,tuned --shard 0/3 --tag s0 >> results\worker_s0.log 2>&1
timeout /t 6 /nobreak >nul
start "" /B .venv\Scripts\python.exe -u -m killtest.run_full --datasets nasa,psm,smd,ucr --n-ucr 250 --tiers default,tuned --shard 1/3 --tag s1 >> results\worker_s1.log 2>&1
timeout /t 6 /nobreak >nul
start "" /B .venv\Scripts\python.exe -u -m killtest.run_full --datasets nasa,psm,smd,ucr --n-ucr 250 --tiers default,tuned --shard 2/3 --tag s2 >> results\worker_s2.log 2>&1
echo Launched 3 workers. Check with:  .venv\Scripts\python.exe check_progress.py
