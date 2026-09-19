import json
import time

class HarnessContext:
    def __init__(self, N: int, initial_t: int = 50):
        self.N = N
        self.t = initial_t
        self.system_prompt = "Ты - живая нейронная сеть..."
        self.n1 = len(self.system_prompt.split())

        self.memories = {}
        self.birth_time = time.time()
        self.sliding_window = []

        self.generation_timestamps = []

        self.total_memory_tokens = 0
        self.memory_entries_count = 0

    def add_generation_timestamp(self, ts, tokens_generated):
        self.generation_timestamps.append((ts, tokens_generated))
        if len(self.generation_timestamps) > 5:
            self.generation_timestamps.pop(0)

    def update_dynamic_t(self, tokens_used):
        self.total_memory_tokens += tokens_used
        self.memory_entries_count += 1
        self.t = int(self.total_memory_tokens / self.memory_entries_count)

    def get_time_header(self):
        current = time.time()
        uptime = current - self.birth_time

        speed_info = ""
        if len(self.generation_timestamps) >= 2:
            dt = self.generation_timestamps[-1][0] - self.generation_timestamps[0][0]
            dtok = sum(x[1] for x in self.generation_timestamps[1:])
            if dt > 0:
                speed = dtok / dt
                speed_info = f" | Generation speed: {speed:.1f} tok/s"

        return f"[SYSTEM TIME] Birth: {self.birth_time:.2f} | Current: {current:.2f} | Uptime: {uptime:.2f}s{speed_info}"

    def h_size(self):
        return len(self.get_time_header().split())

    def get_memory_space(self) -> int:
        available_tokens = (self.N / 2) - self.n1 - self.h_size()
        cells = max(0, int(available_tokens // self.t))
        return cells

    def check_and_truncate_window(self, current_tokens):
        limit = self.N / 2
        warning_msg = None

        if current_tokens > limit * 0.95:
            keep_tokens = int(limit * 0.10)
            return keep_tokens, None

        if current_tokens > limit * 0.85 and current_tokens < limit * 0.90:
             warning_msg = "\n[SYSTEM: Внимание! Плавающее окно заполнено на 85%. Скоро произойдет очистка.]\n"

        return current_tokens, warning_msg

    def build_full_context(self) -> str:
        context = self.system_prompt + "\n"

        if self.memories:
            context += "<memory>\n"
            for k, v in self.memories.items():
                context += f"[{v['timestamp']}] {k}: {v['data']}\n"
            context += "</memory>\n"

        context += self.get_time_header() + "\n\n"
        context += "".join(self.sliding_window)

        return context
