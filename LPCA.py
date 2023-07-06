import numpy as np
import scipy

class LogisticPCA():
    def __init__(self, m, k):
        self.m = m
        self.k = k

    
    def fit(self, X, tol, maxiters=1000, verbose=False):
        """
        Fits the Logistic PCA model

        Parameters:
        - X (matrix): Data matrix containing only bernoulli RVs
        - m (float): Scale factor of matrix Q. As m->infinity, the model becomes more saturated
        - k (int): Number of principle components to keep
        - tol (float): Converge criteria. Minimum allowed difference between trained model and perfect fit
        """

        # Parameter Initializations
        # n: # of observations, p: # of features
        n, d = X.shape  

        # Natural parameters of the saturated model Theta_S
        Q = (2*X) - 1
        Theta_S = self.m * Q

        # Initialize U to the k right singular values of Q
        U = np.linalg.svd(Q)[2].T[:, :self.k]
        #U = self.generate_random_orthonormal_matrix(d, d)[:, :self.k]

        # Initialize mu to logit(X_bar)
        mu = np.mean(X, axis=0).reshape(-1, 1).T
        mu = self.logit(mu)
        Mu = np.ones((n, 1)) @ mu

        # Initialize Theta
        Theta = Mu + ((Theta_S - Mu) @ U @ U.T)

        # Initialize likelihood
        likelihood = self.likelihood(X, Theta)

        for iter in range(maxiters):
            # Update Z
            Z = Theta + 4 * (X - self.sigmoid(Theta))

            # Update mu
            mu = (1/n) * ((Z - (Theta_S @ U @ U.T)).T @ np.ones((n, 1))).T
            Mu = np.ones((n, 1)) @ mu

            # Compute E and update U
            Theta_centered = Theta_S - Mu
            Z_centered = Z - Mu
            temp = Theta_centered.T @ Z_centered
            argmax = temp + temp.T - (Theta_centered.T @ Theta_centered)
            eigenvectors = scipy.linalg.eigh(argmax)[1]  # eigh solves problem of complex eigenvectors/values
            U = eigenvectors[:, -self.k:]  # Returns eigenvalues in ascending order, NOT descending

            # Converge criteria
            Theta = Mu + ((Theta_S - Mu) @ U @ U.T)
            new_likelihood = self.likelihood(X, Theta)

            if likelihood > new_likelihood:
                print("Likelihood decreased, this should never happen. There is probably a bug.")
                break
            elif  abs(new_likelihood - likelihood) < tol:
                print("Reached Convergence on Iteration #" + str(iter + 1) + ", with a log likelihood of " + str(new_likelihood))
                break
            else:
                if verbose:
                    print("Iteration #" + str(iter) + ", Log Likelihood: " + str(new_likelihood))
                likelihood = new_likelihood

        self.mu = mu
        self.U = U

    
    def transform(self, X):
        """
        Transforms new data using the same model

        Parameters:
        - X (matrix): New binary data with the same number of features

        Returns:
        - Theta (matrix): Mean centered projection of the natural parameters
        """
        Q = (2*X) - 1
        Theta_S = (self.m * Q) - (np.ones((X.shape[0], 1)) @ self.mu)
        return Theta_S @ self.U

        
    def likelihood(self, X, Theta):
        """
        Compute the log likelihood of Theta as the Bernoulli natural parameters of X

        Parameters:
        - X (matrix): Original binary data
        - Theta (matrix): Estimated natural parameters
        """
        return np.sum(X * Theta - np.log(1 + np.exp(Theta)))
    

    def sigmoid(self, X):
        """
        Sigmoid of X matrix

        Parameters:
        - X (matrix): Matrix to apply sigmoid to

        Returns:
        - A (matrix): Matrix with sigmoid funciton applied elementwise
        """
        clipped_X = np.clip(X, -1 * self.m, self.m)
        t = np.exp(-clipped_X)
        return 1.0 / (1.0 + t)
    

    def logit(self, x):
        logit = np.log((x + 0.00001) / (1 - x)) # Add 0.00001 to avoid issues when no rows exhibit a behavior
        clipped = np.clip(logit, -1 * self.m, self.m)
        return clipped
    

    def is_symmetric(self, X):
        transpose = np.transpose(X)
        return np.array_equal(X, transpose)
