import numpy as np


class RotaryFloat():
    def __init__(self, init_value=None):
        if init_value is None:
            init_value = np.random.randn()
        self.center = np.random.randn()
        self.val = init_value
        self.dv = 0.001 * np.random.randn()
        self.deltas = [[0, 1], [0, 1], [0, 1]]

    def _ddv(self):
        return (-0.0003 * self.deltas[2][1] + self.deltas[2][0]) * np.sign(self.val - self.center)

    def __float__(self):
        return self.val * self.deltas[0][1] + self.deltas[0][0]

    def step(self):
        self.dv += self._ddv()
        cur_dv = self.dv * self.deltas[1][1] + self.deltas[1][0]
        self.val += cur_dv

class PositiveRotaryFloat(RotaryFloat):
    def __float__(self):
        return np.abs(super().__float__())

class AnimatedVariation():
    def __init__(self, variation, lifespan=None):
        self.var = variation
        self.anim_args = [RotaryFloat() for p in variation.all_params]
        self.var.weight = PositiveRotaryFloat(variation.weight)
        #self.age = 0
        #self.lifespan = lifespan

    def step(self):
        if not self.var.unchanging:
            for rf in self.anim_args:
                rf.step()
            self.var.weight.step()
            self.var.all_params = np.array([float(rf) for rf in self.anim_args])
            self.var.all_params[np.isnan(self.var.all_params)] = 0
            self.var.make_pre()

        #self.age += 1
        #if self.lifespan is not None:
        #    phase = 0 if self.age >= self.lifespan else (self.age / self.lifespan)
        #    phase *= 2 * np.pi
        #    mag = (np.cos(phase) + 1) * -0.5
        #    self.var.weight = mag

    def __getattr__(self, key):
        return self.var.__getattribute__(key)

    def __str__(self):
        return "animated_" + str(self.var)

    def __call__(self, x, y):
        self.var(x, y)

    def __len__(self):
        return len(self.var)

    def __iter__(self):
        return iter(self.var)

    def __hash__(self):
        return hash(self.var)

