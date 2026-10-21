import os
import json
import datetime
import random
from unittest import mock
import living_harness.auto_behavior_change as auto_behavior_change

@mock.patch('living_harness.auto_behavior_change.STATE_FILE', 'test_behavior_state.json')
@mock.patch('living_harness.auto_behavior_change.KB_FILE', 'test_behavior_experiments.md')
@mock.patch('living_harness.auto_behavior_change.PROMPTS_DIR', 'test_prompts')
def test_auto_behavior_change_lifecycle():
    # Setup
    os.makedirs('test_prompts', exist_ok=True)
    with open('test_prompts/alexey.md', 'w') as f:
        f.write("Original content.")

    # Force the choice to be alexey
    with mock.patch('living_harness.auto_behavior_change.random.choice', return_value='alexey'), \
         mock.patch('living_harness.auto_behavior_change.AGENTS', ['alexey']), \
         mock.patch('living_harness.auto_behavior_change.random.randint', return_value=3):

        # Test 1: Start new experiment
        auto_behavior_change.main()

        with open('test_behavior_state.json', 'r') as f:
            state = json.load(f)
            assert state['current_experiment'] is not None
            assert state['current_experiment']['agent'] == 'alexey'
            assert state['current_experiment']['duration'] == 3
            start_date_str = state['current_experiment']['start_date']

        with open('test_prompts/alexey.md', 'r') as f:
            content = f.read()
            assert "[ЭКСПЕРИМЕНТ]" in content

        # Test 2: Run again immediately (experiment should still be running)
        auto_behavior_change.main()

        with open('test_behavior_state.json', 'r') as f:
            state2 = json.load(f)
            assert state2['current_experiment'] is not None

        # Test 3: Fast forward 4 days to end experiment
        future_date = datetime.datetime.fromisoformat(start_date_str) + datetime.timedelta(days=4)

        with mock.patch('living_harness.auto_behavior_change.datetime') as mock_datetime:
            mock_datetime.datetime.now.return_value = future_date
            mock_datetime.datetime.fromisoformat = datetime.datetime.fromisoformat

            auto_behavior_change.main()

        with open('test_behavior_state.json', 'r') as f:
            state3 = json.load(f)
            assert state3['current_experiment'] is None
            assert state3['last_run'] is not None

        with open('test_prompts/alexey.md', 'r') as f:
            content3 = f.read()
            assert "[ЭКСПЕРИМЕНТ]" not in content3

        # Test 4: Fast forward 1 day after end (too early)
        future_date_2 = future_date + datetime.timedelta(days=1)
        with mock.patch('living_harness.auto_behavior_change.datetime') as mock_datetime:
            mock_datetime.datetime.now.return_value = future_date_2
            mock_datetime.datetime.fromisoformat = datetime.datetime.fromisoformat

            auto_behavior_change.main()

        with open('test_behavior_state.json', 'r') as f:
            state4 = json.load(f)
            assert state4['current_experiment'] is None

        # Test 5: Fast forward 4 days after end (new experiment)
        future_date_3 = future_date + datetime.timedelta(days=4)
        with mock.patch('living_harness.auto_behavior_change.datetime') as mock_datetime:
            mock_datetime.datetime.now.return_value = future_date_3
            mock_datetime.datetime.fromisoformat = datetime.datetime.fromisoformat

            auto_behavior_change.main()

        with open('test_behavior_state.json', 'r') as f:
            state5 = json.load(f)
            assert state5['current_experiment'] is not None

    # Cleanup
    if os.path.exists('test_behavior_state.json'):
        os.remove('test_behavior_state.json')
    if os.path.exists('test_behavior_experiments.md'):
        os.remove('test_behavior_experiments.md')
    if os.path.exists('test_prompts/alexey.md'):
        os.remove('test_prompts/alexey.md')
    if os.path.exists('test_prompts'):
        os.rmdir('test_prompts')
