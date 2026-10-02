import time
from living_harness.core.self_prompting import SelfPrompter
from living_harness.core.local_llm_connector import LocalLLMConnector
from living_harness.memory.consolidation import MemoryConsolidator
from living_harness.core.entropy import EntropyManager
from living_harness.core.fatigue_manager import FatigueManager

class BackgroundLoop:
    def __init__(self):
        self.llm = LocalLLMConnector()
        self.prompter = SelfPrompter(llm_connector=self.llm, idle_threshold=10.0)
        self.consolidator = MemoryConsolidator()
        self.entropy_manager = EntropyManager(boredom_threshold=5.0)
        self.fatigue_manager = FatigueManager()
        self.deep_sleep_threshold = 60.0 # Порог для сна

    def run(self):
        print("Starting background loop...")
        while True:
            idle_time = time.time() - self.prompter.last_active_time

            if self.fatigue_manager.is_exhausted() or idle_time > self.deep_sleep_threshold:
                print("Entering deep sleep. Consolidating memories and recovering fatigue...")
                self.consolidator.run_sleep_cycle()
                self.prompter.update_activity()
                self.entropy_manager.reset_boredom()
                self.fatigue_manager.recover(self.fatigue_manager.fatigue_threshold) # Full recovery
            elif self.prompter.should_self_prompt():
                print("Idle time exceeded threshold. Self-prompting...")
                self.entropy_manager.increase_boredom(1.0)
                if self.entropy_manager.should_inject_entropy():
                    perturbation = self.entropy_manager.generate_perturbation()
                    print(f"Injecting entropy: {perturbation}")
                    goal = self.prompter.generate_goal(f"User is inactive. {perturbation}")
                else:
                    goal = self.prompter.generate_goal("User is inactive.")
                print(f"Generated internal goal: {goal}")
                self.prompter.update_activity()
                self.fatigue_manager.add_fatigue()
            time.sleep(2)

if __name__ == "__main__":
    loop = BackgroundLoop()
    # loop.run() # Uncomment to run infinitely
