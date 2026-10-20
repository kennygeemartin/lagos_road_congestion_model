"""Monitor the authorized full run, update status, finalize only after completion.

Run concurrently with run_reconstructed_experiments.py. Never starts another
trainer or reopens final test. All progress is read from actual saved artifacts.
"""
import subprocess
import sys
import time
import psutil
import pandas as pd
from reconstructed_study import ROOT,STUDY
from reconstructed_progress import refresh

def trainer_alive():
    for process in psutil.process_iter():
        try:
            if process.name().lower()=='python.exe' and any(a.endswith('run_reconstructed_experiments.py') for a in process.cmdline()):return True
        except (psutil.NoSuchProcess,psutil.AccessDenied):continue
    return False

def main():
    while True:
        refresh()
        final=STUDY/'results/final_eval/completed.json'
        benchmark=STUDY/'results/benchmark/benchmark_by_seed.csv'
        if final.exists() and benchmark.exists() and len(pd.read_csv(benchmark))==5:
            subprocess.run([sys.executable,'summarize_reconstructed_study.py'],cwd=ROOT,check=True)
            print('Full study and result compilation completed.',flush=True)
            return
        if not trainer_alive():
            print('Trainer exited before study completion. Inspect reconstructed_run.log; no result compilation attempted.',flush=True)
            raise SystemExit(1)
        time.sleep(30)

if __name__=='__main__':main()
