import numpy as np
### Todo replace if else to match-case (not recognized in python 3.7)
class FuncClass():
    def __init__(self, function_name):
        self.function_name = function_name

    def run(self,input):
        if self.function_name=='ampsin':
            return np.sin(input)*input/2
        elif self.function_name=='ampcos':
            return np.cos(input)*input/2

