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
        elif self.function_name=='exp-sin':
            return np.exp(-0.2*input)*np.sin(input)
        elif self.function_name=='tan-sin':
            return np.tan(0.15*input)*np.sin(input)
        elif self.function_name=='tanh-cos':
            return 0.2*np.square(input)*np.tanh(np.cos(input)) 
        elif self.function_name=='log-sin':
            return 0.5*np.log(np.square(input)*(np.sin(2*input)+2)+1)

