import zmq
import time
import os
import sys
import click
import json

# Setup Data Directory
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

def get_next_run_id():
    # Simple auto-increment
    existing = [d for d in os.listdir(DATA_DIR) if d.startswith("run_") and d.endswith("_motor.json")]
    return f"run_{len(existing) + 1:03d}"

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

    # Trigger async validation
    click.secho("    Validating...", dim=True)
    import subprocess
    subprocess.Popen([sys.executable, "validator.py", run_id])

@click.command()
@click.option('--delay', default=0.05, help='Sync delay latency in seconds (default 0.05s).')
@click.option('--port', default=5555, help='ZMQ PUB port.')
def main(delay, port):
    """
    Zumi Pipeline Orchestrator
    
    Controls Start/Stop signals with precise timestamp synchronization.
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
    click.echo("  [Ctrl+C] Exit")
    
    is_recording = False
    current_run = None
    
    # Allow ZMQ connections to establish
    time.sleep(0.5)

    try:
        while True:
            click.pause(info="") # Waits for any key (or Enter)
            
            if not is_recording:
                current_run = send_start(socket, delay)
                is_recording = True
                click.echo("(Recording... Press Enter to STOP)")
            else:
                send_stop(socket, current_run, delay)
                is_recording = False
                click.echo("(Stopped. Press Enter to START)")
                
    except KeyboardInterrupt:
        click.secho("\nExiting...", fg="red")
        socket.send_json({"cmd": "EXIT"})

if __name__ == "__main__":
    main()