import time


def precise_wait(t_end: float, slack_time: float = 0.002, time_func=time.monotonic):
    """
    Wait until absolute time `t_end` using a hybrid sleep/spin.
    """
    t_start = time_func()
    t_wait = t_end - t_start
    if t_wait <= 0:
        return

    if t_wait > slack_time:
        time.sleep(t_wait - slack_time)

    while time_func() < t_end:
        pass


class RateLimiter:
    """
    Precise loop frequency controller with drift correction.
    """

    def __init__(self, frequency: float, slack: float = 0.002):
        self.dt = 1.0 / frequency
        self.next_wake = time.monotonic()
        self.slack = slack

    def sleep(self):
        self.next_wake += self.dt
        now = time.monotonic()

        if now > self.next_wake:
            if now - self.next_wake > self.dt:
                self.next_wake = now
            return

        precise_wait(self.next_wake, self.slack)
