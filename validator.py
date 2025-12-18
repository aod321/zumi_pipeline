import sys
import os
import json
from datetime import datetime

def validate(run_id):
    motor_file = f"data/{run_id}_motor.json"
    
    if not os.path.exists(motor_file):
        print(f"[FAIL] {run_id}: Missing Motor Data")
        return

    try:
        with open(motor_file, 'r') as f:
            data = json.load(f)
            
        if not data:
            print(f"[FAIL] {run_id}: Empty Data")
            return
            
        # Frequency Check
        count = len(data)
        t_start = datetime.fromisoformat(data[0]['date'].replace('Z', '+00:00'))
        t_end = datetime.fromisoformat(data[-1]['date'].replace('Z', '+00:00'))
        duration = (t_end - t_start).total_seconds()
        
        freq = count / duration if duration > 0 else 0
        
        # Iteration Continuity Check
        iters = [d['iter'] for d in data]
        dropped = 0
        if len(iters) > 1:
            diffs = [j-i for i, j in zip(iters[:-1], iters[1:])]
            # Ideally all diffs should be 1
            dropped = sum(d - 1 for d in diffs if d > 1)
        
        status = "VALID" if freq > 100 else "WARNING"
        print(f"[{status}] {run_id}: {freq:.2f}Hz, Frames={count}, Duration={duration:.2f}s, Dropped_Idx={dropped}")
        
    except Exception as e:
        print(f"[ERROR] {run_id}: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        validate(sys.argv[1])