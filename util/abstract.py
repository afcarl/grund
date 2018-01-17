class EnvironmentBase:

    def step(self, action):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    @property
    def neurons_required(self):
        raise NotImplementedError
