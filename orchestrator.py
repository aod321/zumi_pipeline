import zmq
import time
import os
import sys
import click
import json
import subprocess
import re

# Setup Data Directory
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

def get_next_run_id():
    # Robust auto-increment: Find max existing ID instead of counting files
    max_id = 0
    if os.path.exists(DATA_DIR):
        for filename in os.listdir(DATA_DIR):
            # Match run_001, run_123, etc.
            match = re.match(r"run_(\d+)", filename)
            if match:
                try:
                    num = int(match.group(1))
                    if num > max_id:
                        max_id = num
                except ValueError:
                    continue
    
    return f"run_{max_id + 1:03d}"

def send_start(socket, delay):
    run_id = get_next_run_id()
    start_ts = time.time() + delay
    
    click.secho(f"\n>>> MISSION START: {run_id}", fg="green", bold=True)
    click.echo(f"    Sync Delay: {delay*1000:.1f}ms")
    
    socket.send_json({
        "cmd": "START_SYNC",
        "payload": {
            "run_id": run_id,
            "start_time": start_ts
        }
    })
    return run_id

def send_stop(socket, run_id, delay):
    stop_ts = time.time() + delay
    
    click.secho(f"\n>>> MISSION STOP: {run_id}", fg="yellow", bold=True)
    click.echo(f"    Sync Delay: {delay*1000:.1f}ms")
    
    socket.send_json({
        "cmd": "STOP_SYNC",
        "payload": {
            "stop_time": stop_ts
        }
    })
    return stop_ts

def run_validation(run_id):
    click.secho("    Validating...", dim=True)
    subprocess.Popen([sys.executable, "validator.py", run_id])

@click.command()
@click.option('--delay', default=0.05, help='Sync delay latency in seconds (default 0.05s).')
@click.option('--port', default=5555, help='ZMQ PUB port.')
def main(delay, port):
    """
    Zumi Pipeline Orchestrator
    
    Controls Start/Stop signals with precise timestamp synchronization.
    Supports Data Counting and Discarding.
    """
    context = zmq.Context()
    socket = context.socket(zmq.PUB)
    socket.bind(f"tcp://*:{port}")
    
    click.clear()
    click.secho("=== Zumi Pipeline Orchestrator ===", fg="cyan", bold=True)
    click.echo(f"[*] Sync Delay: {delay*1000:.1f} ms")
    click.echo(f"[*] Publishing on port {port}")
    click.echo("\nCommands:")
    click.echo("  [ENTER]  Start/Stop Recording")
    click.echo("  [d]      Discard Last Run")
    click.echo("  [v]      Re-Validate Last Run")
    click.echo("  [q]      Quit")
    
    is_recording = False
    current_run = None
    last_run_id = None
    valid_count = 0
    
    # Init valid count from existing files
    try:
        existing = [d for d in os.listdir(DATA_DIR) if d.startswith("run_") and (d.endswith("_motor.json") or d.endswith("_motor.npz"))]
        valid_count = len(existing)
    except Exception:
        pass
        
    click.echo(f"\nTotal Valid Runs: {valid_count}")

    # Allow ZMQ connections to establish
    time.sleep(0.5)

    try:
        while True:
            c = click.getchar()
            
            if c == '\r' or c == '\n': # Enter
                if not is_recording:
                    # START
                    current_run = send_start(socket, delay)
                    is_recording = True
                    click.echo("(Recording... Press Enter to STOP)")
                else:
                    # STOP
                    send_stop(socket, current_run, delay)
                    is_recording = False
                    last_run_id = current_run
                    valid_count += 1
                    click.echo(f"(Stopped. Run {last_run_id} Saved.)")
                    click.secho(f"Total Valid Runs: {valid_count}", fg="blue", bold=True)
                    
                    # Auto-validate (async)
                    run_validation(last_run_id)

            elif c == 'd':
                if is_recording:
                     click.secho("Cannot discard while recording!", fg="red")
                elif last_run_id:
                     click.echo(f"\nDiscarding {last_run_id}? [y/N] ", nl=False)
                     if click.getchar() in ['y', 'Y']:
                         click.echo("Yes")
                         socket.send_json({"cmd": "DISCARD_RUN", "payload": {"run_id": last_run_id}})
                         valid_count = max(0, valid_count - 1)
                         click.secho(f"Run {last_run_id} DISCARDED.", fg="red")
                         click.secho(f"Total Valid Runs: {valid_count}", fg="blue", bold=True)
                         last_run_id = None
                     else:
                         click.echo("No")
                else:
                     click.echo("\nNo run to discard.")
            
            elif c == 'v':
                if last_run_id:
                    run_validation(last_run_id)
                else:
                    click.echo("\nNo run to validate.")

            elif c == 'q' or c == '\x03': # Quit or Ctrl+C
                click.secho("\nExiting...", fg="red")
                socket.send_json({"cmd": "EXIT"})
                break
                
    except KeyboardInterrupt:
        click.secho("\nExiting...", fg="red")
        socket.send_json({"cmd": "EXIT"})

if __name__ == "__main__":
    main()
