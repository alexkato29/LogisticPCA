import numpy as np
import scipy

class LogisticPCA():
    def __init__(self, m, k):
        self.m = m
        self.k = k

    
    def fit(self, X, tol, maxiters=100):
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
        U = np.linalg.svd(X)[2].T[:, :self.k]

        # Theta is the projected, then restored version of Theta
        Theta = self.project(Theta_S, U)

        # Initialize likelihood
        likelihood = -1e10

        for iter in range(maxiters):
            # Update Z
            Z = Theta + 4*(X - self.sigmoid(Theta))

            # Update mu
            mu = (1/n) * (Z - (Theta_S @ U @ U.T)).T @ np.ones((n, 1))

            # Compute E and update U
            Mu = np.ones((n, 1)) @ mu.T
            argmax = (Theta_S - Mu).T @ (Z - Mu) + (Z - Mu).T @ (Theta_S - Mu) - (Theta_S - Mu).T @ (Theta_S - Mu)
            eigenvectors = scipy.linalg.eig(argmax)[1]
            U = eigenvectors[:, :self.k]
            print(iter)

            # Update Theta
            Theta = Mu + ((Theta_S - Mu) @ U @ U.T)

            # Converge criteria
            if abs(likelihood - self.likelihood(X, Theta)) < tol:
                break
            else:
                likelihood = self.likelihood(X, Theta)

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
        Theta_S = (self.m * Q) - (np.ones((X.shape[0], 1)) @ self.mu.T)
        return Theta_S @ self.U


    def project(self, Theta_S, U):
        n = Theta_S.shape[0]
        
        # mean(Theta_S_j) where Theta_S_j is the jth column of the saturated model
        mu = np.mean(Theta_S, axis=0).reshape(-1, 1).T

        # Mu matrix is an NxD matrix where each column is the column mean of the data
        Mu = (np.ones((n, 1)) @ mu)

        return Mu + ((Theta_S - Mu) @ U @ U.T)

        
    def likelihood(self, X, Theta):
        """
        Compute the log likelihood of Theta as the Bernoulli natural parameters of X

        Parameters:
        - X (matrix): Original binary data
        - Theta (matrix): Estimated natural parameters
        """
        n, d = X.shape
        P = self.sigmoid(Theta)

        tr = np.trace(X.T @ Theta)
        
        s = 0
        for i in range(n):
            for j in range (d):
                s += np.log(1 - P[i][j])

        return tr + s
    

    def sigmoid(self, X):
        """
        Sigmoid of X matrix

        Parameters:
        - X (matrix): Matrix to apply sigmoid to

        Returns:
        - A (matrix): Matrix with sigmoid funciton applied elementwise
        """
        t = np.exp(-1 * X)
        return np.reciprocal(1 + t)
