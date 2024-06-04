import os
import numpy as np

DEBUGGER = os.getenv("DEBUGGER")

class normalization:
    
    def __init__(self, data) -> None:
        if DEBUGGER=="True":
            print("exit create_dataset")

        self.data = data
        
        max = np.max(data)
        self.min = np.min(data)
        
        self.range_original = max - self.min
        # range_desired = 1

    def normalize(self, data):
        new_data = (data-self.min)/self.range_original
        return new_data

    def unnormalize(self, X):
        raw_data = X*self.range_original + self.min
        return raw_data
