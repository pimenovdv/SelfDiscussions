import pytest
from unittest.mock import patch, MagicMock
from living_harness.core.background_loop import BackgroundLoop

def test_fatigue_cycles():
    loop = BackgroundLoop()

    loop.fatigue_manager.fatigue_threshold = 100.0
    loop.fatigue_manager.add_fatigue(150.0)

    loop.prompter.last_active_time = float('inf')

    with patch.object(loop.consolidator, 'run_sleep_cycle', MagicMock()) as mock_sleep, \
         patch('time.sleep', side_effect=StopIteration):
            try:
                loop.run()
            except StopIteration:
                pass

            mock_sleep.assert_called_once()
            assert loop.fatigue_manager.is_exhausted() == False
            assert loop.fatigue_manager.fatigue == 0.0

if __name__ == "__main__":
    pytest.main(["-v", "test_fatigue_cycles.py"])
