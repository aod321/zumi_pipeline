import time

def precise_sleep(dt: float, slack_time: float=0.002, time_func=time.monotonic):
    """
    Hybrid sleep: Sleeps first, then busy-waits (spins) for the last few milliseconds.
    Reduces CPU usage while maintaining sub-millisecond precision.
    """
    t_start = time_func()
    if dt > slack_time:
        time.sleep(dt - slack_time)
    t_end = t_start + dt
    while time_func() < t_end:
        pass
    return

def precise_wait(t_end: float, slack_time: float=0.002, time_func=time.monotonic):
    """
    Waits until the absolute time `t_end` is reached.
    """
    t_start = time_func()
    t_wait = t_end - t_start
    if t_wait > 0:
        t_sleep = t_wait - slack_time
        if t_sleep > 0:
            time.sleep(t_sleep)
        while time_func() < t_end:
            pass
    return