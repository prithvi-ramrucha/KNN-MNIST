# Dependencies,
import numpy as np

class CrossValidation():
    """Class for the k-fold cross-validation algorithm."""

    def __init__(self, k, model_fn):
        """Constuctor method. This method assigns the class variables."""

        # Assigning hyperparametes as class variables,
        self.k = k

        # Storing model class,
        self.model_fn = model_fn

    def evaluate(self, X, y):
        """This method perfoms cross-validation on the dataset X."""

        # Assigning class variables,
        self.X, self.y = X, y

        # Generating indices,
        N = X.shape[0]
        X_idxs = np.arange(start=0, stop=N, step=1)

        # Randomly shuffle datapoints,
        np.random.shuffle(X_idxs)

        # Creating folds,
        folds_idxs = np.array_split(X_idxs, self.k) # <-- List of arrays

        # Evaluation loop,
        model_accuracies = []
        for i in range(self.k):

            # Creating model instance,
            model = self.model_fn()

            # Extracting the indices of the fold,
            fold_idxs = folds_idxs[i]

            # Extracting the complimentary indices to the fold,
            fold_compliment_idxs= np.concatenate([folds_idxs[j] for j in range(self.k) if j != i])

            # Creating data split,
            X_train, y_train = X[fold_compliment_idxs], y[fold_compliment_idxs]
            X_test, y_test = X[fold_idxs], y[fold_idxs]

            # Fitting the model,
            model.fit(X_train, y_train)

            # Scoring the model,
            accuracy = model.score(X_test, y_test)
            model_accuracies.append(accuracy)
        
        # Returning the mean model accuracy,
        return np.mean(model_accuracies)