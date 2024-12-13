import numpy as np


### Todo replace if else to match-case (not recognized in python 3.7)
class FuncClass():
    def __init__(self, function_name):
        self.function_name = function_name

    def run(self, input, noise_variance):
        if self.function_name == 'test':
            result = -input * np.sin(input / 3)
        elif self.function_name == 'amp-sin':
            result = np.sin(input) * input / 2
        elif self.function_name == 'amp-cos':
            result = np.cos(input) * input / 2
        elif self.function_name == 'exp-sin':
            result = np.exp(-0.2 * input) * (np.sin(input) + input)
        elif self.function_name == 'tan-sin':
            result = np.tan(0.15 * input) * np.sin(input)
        elif self.function_name == 'tanh-cos':
            result = 0.2 * np.square(input) * np.tanh(np.cos(input))
        elif self.function_name == 'log-sin':
            result = 0.5 * np.log(np.square(input) * (np.sin(2 * input) + 2) + 1)
        elif self.function_name == 'sin-sin':
            result = -7 * np.sin(input / 3) + 2 * np.sin(input * 10 / 9)

        if noise_variance != False:
            result = result + np.random.normal(0, noise_variance, result.shape[0])

        return result
