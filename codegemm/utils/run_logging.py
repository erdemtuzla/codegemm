import atexit
import datetime
import os
import re
import sys


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
        self.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()


def _safe_name(value):
    value = str(value or "run")
    value = os.path.basename(os.path.normpath(value)) or "run"
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def setup_run_log(script_name, log_dir="history", run_name=None):
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    original_stdout = sys.stdout
    original_stderr = sys.stderr

    parts = [timestamp, _safe_name(script_name)]
    if run_name:
        parts.append(_safe_name(run_name))

    rank = os.environ.get("RANK") or os.environ.get("LOCAL_RANK")
    if rank is not None:
        parts.append(f"rank{rank}")
    parts.append(f"pid{os.getpid()}")

    filename_base = "_".join(parts)
    for attempt in range(100):
        suffix = "" if attempt == 0 else f"_{attempt}"
        log_path = os.path.join(log_dir, f"{filename_base}{suffix}.log")
        try:
            log_file = open(log_path, "x", buffering=1)
            break
        except FileExistsError:
            continue
    else:
        raise RuntimeError(f"Could not create a unique log file in {log_dir}")

    sys.stdout = Tee(original_stdout, log_file)
    sys.stderr = Tee(original_stderr, log_file)

    def close_run_log():
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        log_file.close()

    atexit.register(close_run_log)
    return log_path
