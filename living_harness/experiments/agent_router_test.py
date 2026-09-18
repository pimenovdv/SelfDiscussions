from living_harness.core.agent import Agent
from living_harness.core.agent_router import AgentRouter
import sys
import os

def main():
    # Создаем тестовых агентов с разными специализациями
    coder_agent = Agent(name="CodeBot", system_prompt="I am a highly skilled programmer. I write code, debug issues, and architect software systems.")
    theory_agent = Agent(name="TheoryBot", system_prompt="I am a theoretical physicist and philosopher. I discuss abstract concepts, quantum mechanics, and metaphysics.")
    design_agent = Agent(name="DesignBot", system_prompt="I am a UI/UX designer. I create beautiful interfaces, focus on typography, color theory, and user experience.")

    router = AgentRouter(agents=[coder_agent, theory_agent, design_agent])

    queries = [
        "Can you help me center a div using CSS Flexbox?",
        "What are the implications of string theory on our understanding of multiple universes?",
        "Which color palette would best suit a modern meditation app?"
    ]

    results = []
    for q in queries:
        selected_agent = router.route(q)
        results.append(f"Query: '{q}' -> Routed to: {selected_agent.name}")
        print(f"Query: '{q}'\nRouted to: {selected_agent.name}\n")

    os.makedirs('living_harness/data', exist_ok=True)
    with open('living_harness/data/agent_router_results.txt', 'w') as f:
        f.write("\n".join(results))

if __name__ == "__main__":
    main()
