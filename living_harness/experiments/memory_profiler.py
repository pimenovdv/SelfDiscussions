import sys
import os
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from living_harness.memory.harness_context import HarnessContext

def run_profiling():
    # Simulate a Harness Context with N=2048
    context = HarnessContext(N=2048, initial_t=50)

    # Simulate continuous generation
    for i in range(1, 100):
        # Simulate tokens generated (approx 20 per step)
        context.add_generation_timestamp(time.time(), 20)

        # Simulate adding a memory periodically
        if i % 10 == 0:
             context.memories[f"mem_{i}"] = {"timestamp": time.time(), "data": f"Important event {i}"}
             context.update_dynamic_t(25) # assumed tokens used for the memory

        # Add to sliding window
        context.sliding_window.append(f"Step {i} text. ")

        # Calculate current window size
        current_tokens = len("".join(context.sliding_window).split())

        new_tokens, warning = context.check_and_truncate_window(current_tokens)
        if new_tokens < current_tokens:
            print(f"Truncated at step {i} from {current_tokens} to {new_tokens}")
            # Keep only the last `new_tokens` words
            words = "".join(context.sliding_window).split()
            context.sliding_window = [" ".join(words[-new_tokens:])]

        if warning:
            print(warning.strip())

        if i % 20 == 0:
            print(f"Step {i}: available memory cells: {context.get_memory_space()}")
            print(f"Time header: {context.get_time_header()}")

if __name__ == "__main__":
    run_profiling()
