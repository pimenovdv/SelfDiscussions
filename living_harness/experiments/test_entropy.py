import unittest
import time
from living_harness.core.entropy import EntropyManager
from living_harness.core.background_loop import BackgroundLoop
from living_harness.core.self_prompting import SelfPrompter

class TestEntropyManager(unittest.TestCase):
    def test_increase_boredom(self):
        manager = EntropyManager(boredom_threshold=5.0)
        manager.increase_boredom(2.0)
        self.assertEqual(manager.boredom_index, 2.0)
        manager.increase_boredom(3.0)
        self.assertEqual(manager.boredom_index, 5.0)

    def test_should_inject_entropy(self):
        manager = EntropyManager(boredom_threshold=5.0)
        manager.increase_boredom(4.9)
        self.assertFalse(manager.should_inject_entropy())
        manager.increase_boredom(0.1)
        self.assertTrue(manager.should_inject_entropy())

    def test_generate_perturbation(self):
        manager = EntropyManager(boredom_threshold=5.0)
        manager.increase_boredom(6.0)
        perturbation = manager.generate_perturbation()
        self.assertIsInstance(perturbation, str)
        self.assertTrue(len(perturbation) > 0)
        self.assertEqual(manager.boredom_index, 0.0) # Should reset

class TestBackgroundLoopEntropy(unittest.TestCase):
    def test_entropy_integration(self):
        # We need a mock llm to avoid slow generations
        class MockLLM:
            def generate(self, prompt):
                return "Mock generated goal"

        loop = BackgroundLoop()
        loop.llm = MockLLM()
        loop.prompter = SelfPrompter(llm_connector=loop.llm, idle_threshold=0.1)

        # Override threshold to test injection quickly
        loop.entropy_manager.boredom_threshold = 2.0

        # 1st self-prompt (idle time exceeded)
        time.sleep(0.15)
        self.assertTrue(loop.prompter.should_self_prompt())
        loop.entropy_manager.increase_boredom(1.0)
        self.assertFalse(loop.entropy_manager.should_inject_entropy())
        loop.prompter.update_activity()

        # 2nd self-prompt
        time.sleep(0.15)
        self.assertTrue(loop.prompter.should_self_prompt())
        loop.entropy_manager.increase_boredom(1.0)
        self.assertTrue(loop.entropy_manager.should_inject_entropy())

        # Inject entropy
        perturbation = loop.entropy_manager.generate_perturbation()
        self.assertTrue(len(perturbation) > 0)
        self.assertEqual(loop.entropy_manager.boredom_index, 0.0)

    def test_background_loop_context_dynamics(self):
        # Test long-term context dynamics with perturbation injection
        class MockLLM:
            def __init__(self):
                self.calls = 0
            def generate(self, prompt):
                self.calls += 1
                return f"Mock goal {self.calls}"

        loop = BackgroundLoop()
        loop.llm = MockLLM()
        loop.prompter = SelfPrompter(llm_connector=loop.llm, idle_threshold=0.1)
        loop.entropy_manager.boredom_threshold = 1.0 # Set low to trigger often

        # We'll simulate 3 cycles where the loop generates goals
        goals = []
        for _ in range(3):
            time.sleep(0.15)
            if loop.prompter.should_self_prompt():
                loop.entropy_manager.increase_boredom(1.0)
                if loop.entropy_manager.should_inject_entropy():
                    perturbation = loop.entropy_manager.generate_perturbation()
                    goal = loop.prompter.generate_goal(f"User is inactive. {perturbation}")
                else:
                    goal = loop.prompter.generate_goal("User is inactive.")
                goals.append(goal)
                loop.prompter.update_activity()

        self.assertEqual(len(goals), 3)
        self.assertEqual(loop.llm.calls, 3)

if __name__ == '__main__':
    unittest.main()
