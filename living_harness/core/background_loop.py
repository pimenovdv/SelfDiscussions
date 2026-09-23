import time
from living_harness.core.self_prompting import SelfPrompter
from living_harness.core.local_llm_connector import LocalLLMConnector

class BackgroundLoop:
    def __init__(self):
        self.llm = LocalLLMConnector()
        self.prompter = SelfPrompter(llm_connector=self.llm, idle_threshold=10.0)

    def run(self):
        print("Starting background loop...")
        while True:
            if self.prompter.should_self_prompt():
                print("Idle time exceeded threshold. Self-prompting...")
                goal = self.prompter.generate_goal("User is inactive.")
                print(f"Generated internal goal: {goal}")
                self.prompter.update_activity()
            time.sleep(2)

if __name__ == "__main__":
    loop = BackgroundLoop()
    # loop.run() # Uncomment to run infinitely
