import time

class Timer:
    def __init__(self, timeout:float, start_now=False):
        self.timeout = timeout
        self.start_ts = 0.0
        if start_now:
            self.start()

    def start(self):
        self.start_ts = time.perf_counter()

    def reached_timeout(self) -> bool:
        if self.start_ts == 0:
            return False

        if time.perf_counter() - self.start_ts > self.timeout:
            return True

        return False

    def restart(self):
        self.start()

class ConditionalTimer():
    def __init__(self, timeout:float):
        self.timeout = timeout
        self.start_ts = 0.0
        self.condition = False
        self.condition_lock = False

    def set_conditional_flag(self):
        self.condition = True

    def start(self):
        if self.condition and not self.condition_lock:
            self.start_ts = time.perf_counter()
            self.condition_lock = True

    def reached_timeout(self) -> bool:
        if self.start_ts == 0:
            return False

        if time.perf_counter() - self.start_ts > self.timeout:
            return True

        return False

    def restart(self):
        self.start_ts = time.perf_counter()


