import sys
import timeit

import pandas as pd


class CodeTimer:
    def __init__(self, activity_name=None):
        self.activity_name = activity_name
        self.has_already_exited = False
        self.start = timeit.default_timer()

    def __enter__(self):
        self.start = timeit.default_timer()
        if self.activity_name:
            print(f"⚡ {self.activity_name}...")
            sys.stdout.flush()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if not self.has_already_exited:
            ms_taken = (timeit.default_timer() - self.start) * 1000.0
            pd_time_delta = pd.to_timedelta(ms_taken, unit="ms")
            if self.activity_name:
                print(f"Time taken for '{self.activity_name}': {pd_time_delta}.")
            else:
                print(f"Time taken: {pd_time_delta}.")
            print("")
            sys.stdout.flush()

    def exit_with_infos(self):
        ms_taken = (timeit.default_timer() - self.start) * 1000.0
        pd_time_delta = pd.to_timedelta(ms_taken, unit="ms")
        pd_time_delta_str = f"{pd_time_delta}"
        if self.activity_name:
            print(f"Time taken for '{self.activity_name}': {pd_time_delta}.")
        else:
            print(f"Time taken: {pd_time_delta}.")
        print("")
        sys.stdout.flush()
        self.has_already_exited = True
        return {"ms_taken": ms_taken, "time_delta_str": pd_time_delta_str, "pd_time_delta": pd_time_delta}
