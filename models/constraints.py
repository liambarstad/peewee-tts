from torch import nn

class GreaterThan0Constraint:
    def __init__(self, param_names):
        self.param_names = param_names

    def __call__(self, module, grad_input, grad_output):
        for param in self.param_names:
            clamped = nn.Parameter(getattr(module, param).clamp(-1, 1))
            setattr(module, param, clamped)

class GradientScaleParams:
    def __init__(self, param_names, factor):
        self.param_names = param_names
        self.factor = factor

    def __call__(self, module, grad_input, grad_output):
        for param in self.param_names:
            if getattr(module, param).grad is not None:
                getattr(module, param).grad *= self.factor