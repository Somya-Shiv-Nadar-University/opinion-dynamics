class HKModel:
    def __init__(self, n_agents):
        self.n_agents = n_agents

    def run(self):
        print(f"Running HK model with {self.n_agents} agents")