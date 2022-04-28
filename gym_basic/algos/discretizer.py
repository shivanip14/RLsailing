discr_vector = (60, 60, 180, 30,)  # Resolution degrees: How many discrete states per variable. Play with this parameter!


class Discretizer():
    """ mins: vector with minimim values allowed for each variable
        maxs: vector with maximum values allowed for each variable
    """

    def __init__(self, vector_discr, mins, maxs):
        self.mins = mins
        self.maxs = maxs

    def Discretize(self, obs):
        ratios = [(obs[i] + abs(self.mins[i])) / (self.maxs[i] - self.mins[i]) for i in range(len(obs))]
        new_obs = [int(round((discr_vector[i] - 1) * ratios[i])) for i in range(len(obs))]
        new_obs = [min(discr_vector[i] - 1, max(0, new_obs[i])) for i in range(len(obs))]
        return tuple(new_obs)

