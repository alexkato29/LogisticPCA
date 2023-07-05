import numpy as np

class LogisticPCA():
    def __init__(self):
        pass

    
    def fit(self, X, m, k, tol):
        """
        Fits the Logistic PCA model

        Parameters:
        - X (matrix): Data matrix containing only bernoulli RVs
        - m (float): Scale factor of matrix Q. As m->infinity, the model becomes more saturated
        - k (int): Number of principle components to keep
        - tol (float): Converge criteria. Minimum allowed difference between trained model and perfect fit
        """

        # n: # of observations, p: # of features
        n, d = X.shape

        # Create Q, which is simply X but -1 or 1
        Q = (2*X) - 1

        # Intialize the natural parameters of the saturated model Theta_S
        Theta_S = m * Q

        # Initialize mu to mean(Theta_S_j) where Theta_S_j is the jth column of the saturated model
        mu = np.mean(Theta_S, axis=0).reshape(-1, 1).T

        # Initialize U to right singular values of Q
        U = np.linalg.svd(X)[2].T

        # Intialize loss
        Theta = self.project(Theta_S, U)
        loss = self.likelihood(X, Theta)
        print(loss)


    def project(self, Theta_S, U):
        n = Theta_S.shape[0]
        
        # Initialize mu to mean(Theta_S_j) where Theta_S_j is the jth column of the saturated model
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
