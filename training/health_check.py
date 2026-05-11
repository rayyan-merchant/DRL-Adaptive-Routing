import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.hyperparams import (
    LOGS_DIR, N_EPISODES, EPS_MAX, EPS_MIN, STEPS_PER_EP
)

def check_training_log(csv_path, algo_name):
    """Run 5 health checks on a training log CSV. Returns number of checks passed."""
    df = pd.read_csv(csv_path)
    n_passed = 0
    n_eps = len(df)
    
    print(f"\n{'='*60}")
    print(f"HEALTH CHECK: {algo_name} ({n_eps} episodes)")
    print(f"{'='*60}")
    
    # CHECK 1: Epsilon decays correctly
    first_eps = df['epsilon'].iloc[0]
    last_eps  = df['epsilon'].iloc[-1]
    check1 = first_eps >= 0.9 and last_eps <= EPS_MIN + 0.15
    status = "PASS" if check1 else "FAIL"
    print(f"  [{status}] CHECK 1: Epsilon decays correctly")
    print(f"         First eps: {first_eps:.4f}, Last eps: {last_eps:.4f}")
    if not check1:
        print(f"         Expected: first >= 0.9, last <= {EPS_MIN + 0.15:.2f}")
    if check1:
        n_passed += 1
    
    # CHECK 2: Training loss is non-zero
    nonzero_loss = (df['avg_loss'] > 0).sum()
    pct_nonzero = nonzero_loss / n_eps
    check2 = pct_nonzero >= 0.3  # Relaxed: at least 30% should have loss
    status = "PASS" if check2 else "FAIL"
    print(f"  [{status}] CHECK 2: Training loss is non-zero")
    print(f"         {nonzero_loss}/{n_eps} episodes ({pct_nonzero:.0%}) have loss > 0")
    if not check2:
        print(f"         Loss is zero in most episodes — buffer may not be filling")
        print(f"         (REPLAY_START={200} steps, only {n_eps * STEPS_PER_EP} total steps taken)")
    if check2:
        n_passed += 1
    
    # CHECK 3: Episode cost shows downward trend
    split = max(n_eps // 4, 1)  # Use quarters for short training runs
    first_mean = df['reward'].head(split).mean()
    last_mean  = df['reward'].tail(split).mean()
    check3 = last_mean <= first_mean * 1.1  # Allow 10% tolerance
    status = "PASS" if check3 else "FAIL"
    print(f"  [{status}] CHECK 3: Episode cost trend")
    print(f"         First {split} eps avg: {first_mean:.4f}")
    print(f"         Last  {split} eps avg: {last_mean:.4f}")
    if not check3:
        print(f"         Cost not decreasing — check reward formula and argmin")
    if check3:
        n_passed += 1
    
    # CHECK 4: Agent explores all paths
    check4 = True
    if 'action0_frac' in df.columns:
        for i in range(3):
            col = f'action{i}_frac'
            if col in df.columns:
                nonzero = (df[col] > 0).sum()
                pct = nonzero / n_eps
                if pct < 0.05:
                    check4 = False
    status = "PASS" if check4 else "FAIL"
    print(f"  [{status}] CHECK 4: Agent explores all paths")
    if 'action0_frac' in df.columns:
        for i in range(3):
            col = f'action{i}_frac'
            if col in df.columns:
                mean_frac = df[col].mean()
                print(f"         Path {i} avg fraction: {mean_frac:.4f}")
    if check4:
        n_passed += 1
    
    # CHECK 5: No NaN values in log
    has_nan = df.isnull().any().any()
    check5 = not has_nan
    status = "PASS" if check5 else "FAIL"
    print(f"  [{status}] CHECK 5: No NaN values in log")
    if has_nan:
        nan_cols = df.columns[df.isnull().any()].tolist()
        print(f"         NaN found in columns: {nan_cols}")
    if check5:
        n_passed += 1
    
    print(f"\n  RESULT: {n_passed}/5 checks PASSED for {algo_name}")
    return n_passed

if __name__ == "__main__":
    overall = 0
    total = 0
    
    for algo in ['dqn', 'ddqn']:
        csv_path = os.path.join(LOGS_DIR, f"{algo}_training.csv")
        if os.path.exists(csv_path):
            n_passed = check_training_log(csv_path, algo.upper())
            overall += n_passed
            total += 5
            if n_passed < 4:
                print(f"  ⚠️  WARNING: {algo.upper()} training may have issues.")
            else:
                print(f"  ✅ {algo.upper()} training looks healthy!")
        else:
            print(f"\nINFO: {csv_path} not found. Train {algo.upper()} first.")
    
    if total > 0:
        print(f"\n{'='*60}")
        print(f"OVERALL: {overall}/{total} checks passed")
        print(f"{'='*60}")