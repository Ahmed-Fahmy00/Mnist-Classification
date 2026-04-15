import numpy as np

class NaiveBayes:
    def fit(self,x,y):
        self.classes=np.unique(y)
        self.parameters=[]  #list hatsheel el numerical values of each pixel (mean and variance)
        for c in self.classes:
            xClass=x[y==c]
            mean=np.mean(xClass,axis=0)
            var=np.var(xClass,axis=0) + 1e-2 #avoiding division by zero
            prior=xClass.shape[0]/x.shape[0]
            self.parameters.append({
                "mean":mean,
                "var":var,
                "prior":prior
            })
    def likelihoodCalculate(self,x,classidx):
        mean=self.parameters[classidx]["mean"]
        var=self.parameters[classidx]["var"]
        numerator=np.exp(-(x-mean)**2/(2*var))
        denominator=np.sqrt(2*np.pi*var)
        return numerator/denominator     #gaussian distribution formula
    def predict(self, X):
        y_pred = [self._predict_single(x) for x in X] # This creates a list of guesses for every image in the test set
        return np.array(y_pred)

    def _predict_single(self, x):
        posteriors = []
        for i, c in enumerate(self.classes):
            prior = np.log(self.parameters[i]["prior"])
            probs = self.likelihoodCalculate(x, i)
            likelihood = np.sum(np.log(probs + 1e-9))
            posteriors.append(prior + likelihood)
        return self.classes[np.argmax(posteriors)]