# Dependencies,
import numpy as np

class MinMaxScaler():
    """Class for min-max scaling."""

    def __init__(self):
        """Constructor method. The class attributes are created."""

        self.X = None
        X_max, X_min = None, None

    def fit_transform(self, X):
        """Performs min-max normalisation on (n, M) shaped data X stored in the class where n is the number of samples
        and M the number of features and stores the parameters of the normalisation."""

        # Fitting,
        self.X = X

        # Computing minimum and maximum values of each feature,
        X_max = np.max(self.X, axis=0)
        X_min = np.min(self.X, axis=0)
        self.X_max, self.X_min = X_max, X_min

        # Performing normalisation,
        X_normed = (self.X - X_min)/(X_max - X_min)

        return X_normed
    
    def transform(self, X):
        """Performs min-max normalisation on a (n, M) shaped data X."""
        X_normed = (X - self.X_min)/(self.X_max - self.X_min)
        return X_normed