class FatigueManager:
    def __init__(self, fatigue_threshold=100.0, recovery_rate=10.0, action_cost=5.0):
        self.fatigue = 0.0
        self.fatigue_threshold = fatigue_threshold
        self.recovery_rate = recovery_rate
        self.action_cost = action_cost

    def add_fatigue(self, cost=None):
        if cost is None:
            cost = self.action_cost
        self.fatigue = min(self.fatigue + cost, self.fatigue_threshold)

    def recover(self, amount=None):
        if amount is None:
            amount = self.recovery_rate
        self.fatigue = max(self.fatigue - amount, 0.0)

    def is_exhausted(self):
        return self.fatigue >= self.fatigue_threshold
