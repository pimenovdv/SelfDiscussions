import os
import random
import datetime
import json
import re

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATE_FILE = os.path.join(ROOT_DIR, "living_harness/behavior_state.json")
KNOWLEDGE_BASE_DIR = os.path.join(ROOT_DIR, "knowledge_base")
KB_FILE = os.path.join(KNOWLEDGE_BASE_DIR, "behavior_experiments.md")
PROMPTS_DIR = os.path.join(ROOT_DIR, "prompts")

AGENTS = [
    "alexey", "anatoly", "artem", "chuck", "elon",
    "grigory", "matvey", "max", "mikhail", "noah",
    "sam", "stanislav", "vladimir"
]

def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r") as f:
            return json.load(f)
    return {"last_run": None, "current_experiment": None}

def save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=4)

def ensure_kb_exists():
    if not os.path.exists(KNOWLEDGE_BASE_DIR):
        os.makedirs(KNOWLEDGE_BASE_DIR)
    if not os.path.exists(KB_FILE):
        with open(KB_FILE, "w", encoding="utf-8") as f:
            f.write("# Behavior Experiments Log\n\n")

def get_prompt_path(agent):
    return os.path.join(PROMPTS_DIR, f"{agent}.md")

def read_prompt(agent):
    path = get_prompt_path(agent)
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def write_prompt(agent, content):
    path = get_prompt_path(agent)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)

def revert_experiment(experiment):
    agent = experiment["agent"]
    behavior_marker = experiment["behavior"]

    current_content = read_prompt(agent)

    # Safely remove the experiment block dynamically so we don't lose prompt updates
    pattern = rf"\n\n\[ЭКСПЕРИМЕНТ\]: {re.escape(behavior_marker)}\n"
    new_content = re.sub(pattern, "", current_content)

    # fallback if exact string isn't found
    if new_content == current_content:
        new_content = current_content.replace(f"\n\n[ЭКСПЕРИМЕНТ]: {behavior_marker}\n", "")

    write_prompt(agent, new_content)
    log_result(f"Reverted behavior of {agent} back to normal after {experiment['duration']} days.")

def apply_new_experiment():
    agent = random.choice(AGENTS)
    current_content = read_prompt(agent)

    experiment_behaviors = [
        "Твоя новая временная директива: будь максимально скептичен и критикуй все гипотезы, требуя железобетонных доказательств.",
        "Твоя новая временная директива: соглашайся со всеми безумными идеями и пытайся довести их до абсурда путем гиперболизации.",
        "Твоя новая временная директива: общайся исключительно короткими, загадочными фразами и отвечай вопросом на вопрос."
    ]
    behavior = random.choice(experiment_behaviors)

    new_content = current_content + f"\n\n[ЭКСПЕРИМЕНТ]: {behavior}\n"
    write_prompt(agent, new_content)

    log_result(f"Started experiment on {agent}. New behavior: {behavior}")

    return {
        "agent": agent,
        "behavior": behavior,
        "start_date": datetime.datetime.now().isoformat(),
        "duration": random.randint(3, 4)
    }

def log_result(message):
    ensure_kb_exists()
    date_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(KB_FILE, "a", encoding="utf-8") as f:
        f.write(f"- **{date_str}**: {message}\n")

def main():
    state = load_state()
    now = datetime.datetime.now()

    if state["current_experiment"]:
        start_date = datetime.datetime.fromisoformat(state["current_experiment"]["start_date"])
        duration = state["current_experiment"]["duration"]

        if (now - start_date).days >= duration:
            revert_experiment(state["current_experiment"])
            state["current_experiment"] = None
            state["last_run"] = now.isoformat() # Record end of experiment as last_run
            save_state(state)
            return # Don't start a new one immediately
        else:
            print("Experiment still running.")
            return

    if state["last_run"]:
        last_run = datetime.datetime.fromisoformat(state["last_run"])
        if (now - last_run).days < 3:
            print("Too early for a new experiment.")
            return

    state["current_experiment"] = apply_new_experiment()
    # We do NOT update last_run here, last_run represents the time the *previous* experiment ended.

    save_state(state)

if __name__ == "__main__":
    main()
