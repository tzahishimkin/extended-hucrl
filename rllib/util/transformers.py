import numpy as np
import torch



class StateTransform(object):
    def __init__(self, type, factor=0.1):
        self.type = type
        self.factor = factor
        if type == None:
            self.func = lambda x: x[...,-1] * 0.
        elif type == 'half-sum':
            self.func = self.neg_abs_noise

    def neg_abs_noise(self, x):
        # std = np.zeros(x.shape[:-1])
        std = x.mean(-1) >= 0.
        if torch.is_tensor(std):
           std.to(int)
        else:
            try:
                std = np.int(std)
            except TypeError:
                std.astype(int)

        return std * self.factor

    def __call__(self, x, *args, **kwargs):
        std = self.func(x)

        n = np.random.randn()

        # if torch.is_tensor(x):
        #     n = torch.normal(mean=0., std=torch.tensor(std, dtype=torch.float))
        # else:
        #     n = np.random.randn()

        x_noise = n * std
        return x + x_noise


