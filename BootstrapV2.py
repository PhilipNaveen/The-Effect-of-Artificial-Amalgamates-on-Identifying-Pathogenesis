import random

class Bootstrap:
    
    def __init__(self, x, y):
        self.x = x; self.y = y
    
    def SRSM2D(self, number_of_samples=10, sensitivity=10):
        #random samples list for x and y
        x_samples = []; y_samples = []
        #for each sample
        for _ in range(number_of_samples):
            x_mean = 0; y_mean = 0
            #sensitivity for means
            for _ in range(sensitivity):
                x_mean += self.x[random.randint(0, len(self.x))-1]
                y_mean += self.y[0]
            x_samples.append(x_mean/sensitivity)
            y_samples.append(y_mean/sensitivity)
        return nm.array(x_samples).astype("float16"), nm.array(y_samples).astype("float16")
    
    def RSM2D(self, number_of_samples=10):
        #random samples list for x and y
        x_samples = []; y_samples = []
        #for each sample
        for _ in range(number_of_samples):
            x_mean = 0; y_mean = 0
            #sensitivity for means
            sensitivity = random.randint(1, len(self.x))
            for _ in range(sensitivity):
                random.randint(0, len(self.x))
                x_mean += self.x[random.randint(0, len(self.x))-1]
                y_mean += self.y[0]
            x_samples.append(x_mean/sensitivity)
            y_samples.append(y_mean/sensitivity)
        return nm.array(x_samples).astype("float16"), nm.array(y_samples).astype("float16")
    
    
    def SRSM1D(self, number_of_samples=10, sensitivity=10):
        #random samples list for x and y
        x_samples = []
        #for each sample
        for _ in range(number_of_samples):
            x_mean = 0
            #sensitivity for means
            for _ in range(sensitivity):
                x_mean += self.x[random.randint(0, len(self.x))-1]
            x_samples.append(x_mean/sensitivity)
        return nm.array(x_samples).astype("float16")
    
    def RSM1D(self, number_of_samples=10):
        #random samples list for x and y
        x_samples = []
        #for each sample
        for _ in range(number_of_samples):
            x_mean = 0
            #sensitivity for means
            sensitivity = random.randint(1, len(self.x))
            for _ in range(sensitivity):
                random.randint(0, len(self.x))
                x_mean += self.x[random.randint(0, len(self.x))-1]
            x_samples.append(x_mean/sensitivity)
        return nm.array(x_samples).astype("float16")
