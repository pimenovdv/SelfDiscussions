import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import torch
import random
from living_harness.memory.context_manager import ContextManager

def run_test():
    cm = ContextManager(system_prompt="You are a continuous AI.", max_window_tokens=100, enable_hybrid_compression=True)

    print(f"Initial reasoning window size: {len(cm.reasoning_window)}")

    cm.add_reasoning("Block 1. We are discussing AI memory. It is important.", 50, [random.random() for _ in range(128)])
    print(f"After block 1: {len(cm.reasoning_window)} items, {sum(i['tokens'] for i in cm.reasoning_window)} tokens.")

    cm.add_reasoning("Block 2. We should use compression. This saves space.", 40, [random.random() for _ in range(128)])
    print(f"After block 2: {len(cm.reasoning_window)} items, {sum(i['tokens'] for i in cm.reasoning_window)} tokens.")

    # This block should trigger compression (90 tokens + 20 = 110 > 95 threshold)
    cm.add_reasoning("Block 3. Let's see if compression triggers. Need more text.", 20, [random.random() for _ in range(128)])

    print(f"After block 3: {len(cm.reasoning_window)} items, {sum(i['tokens'] for i in cm.reasoning_window)} tokens.")

    print("\nReasoning Window Contents:")
    for i, item in enumerate(cm.reasoning_window):
        print(f"Item {i}: {item['text']}")
        print(f"Tokens: {item['tokens']}, Vector type: {type(item['vector'])}")

if __name__ == "__main__":
    run_test()
