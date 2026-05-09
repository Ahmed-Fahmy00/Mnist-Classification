import numpy as np

class NaiveBayes:
    def __init__(self, epsilon=1e-2):
        self.epsilon = epsilon
        self.classes = None
        self.parameters = []

    def fit(self, x, y):
        # Dynamically find classes
        self.classes = np.unique(y)
        self.parameters = []  
        
        for c in self.classes:
            xClass = x[y == c]
            
            # Feature extraction 
            mean = np.mean(xClass, axis=0)
            
            var = np.var(xClass, axis=0) + self.epsilon 
            
            # Prior probability
            prior = xClass.shape[0] / x.shape[0]
            
            self.parameters.append({
                "mean": mean,
                "var": var,
                "prior": prior
            })

    def likelihoodCalculate(self, x, classidx):

        mean = self.parameters[classidx]["mean"]
        var = self.parameters[classidx]["var"]
        
        # Gaussian formula
        numerator = np.exp(-(x - mean)**2 / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        
        return (numerator / denominator) + 1e-9

    def predict(self, X):

        y_pred = [self._predict_single(x) for x in X]
        return np.array(y_pred)

    def _predict_single(self, x):
        posteriors = []
        
        for i, c in enumerate(self.classes):
            # Prior: Log(P(Class))
            prior_log = np.log(self.parameters[i]["prior"])
            
            # We use Log-Sum for numerical stability
            probs = self.likelihoodCalculate(x, i)
            likelihood_log = np.sum(np.log(probs))
            
            #3amalna sum log probabilities instead of product to avoid underflow
            posteriors.append(prior_log + likelihood_log)
        
        # Return class with highest probability
        return self.classes[np.argmax(posteriors)]