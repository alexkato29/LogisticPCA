import numpy as np
import scipy
import time

class LogisticPCA():
    def __init__(self, m, k):
        """
        Initializes Logistic PCA object.

        Parameters:
        - m (float): Scale factor of matrix Q. As m->infinity, the model becomes more saturated
        - k (int): Number of principle components to keep
        """
        self.m = m
        self.k = k

    
    def fit(self, X, tol=0.0001, maxiters=1000, verbose=False):
        """
        Fits the Logistic PCA model.

        Parameters:
        - X (matrix): Data matrix containing only bernoulli RVs
        - tol (float): Converge criteria. Minimum allowed difference between previous loss and current loss
        - maxiters (int): Maximum number of iterations to run if converge criteria is never reached
        - verbose (boolean): If True, prints information on every 10th iteration
        """

        # Parameter Initializations
        # n: # of observations, p: # of features
        n, d = X.shape  

        # Natural parameters of the saturated model Theta_S
        Q = (2*X) - 1
        Q_sum = np.sum(Q)
        Theta_S = self.m * Q

        # Initialize U to the k right singular values of Q
        start_time = time.time()
        U = np.linalg.svd(Q)[2].T[:, :self.k]
        end_time = time.time()

        if verbose:
            print("SVD Initialization Time: " + str(end_time - start_time))

        # Initialize mu to logit(X_bar)
        mu = np.mean(X, axis=0).reshape(-1, 1).T
        mu = self.logit(mu)
        Mu = np.ones((n, 1)) @ mu
        mean_likelihood = self.likelihood(X, Mu)

        # Initialize Theta
        Theta = Mu + ((Theta_S - Mu) @ U @ U.T)

        # Initialize likelihood
        likelihood = self.likelihood(X, Theta)
        loss = (-likelihood)/Q_sum

        iter = 0
        while iter < maxiters:
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
            new_loss = (-new_likelihood)/Q_sum

            if likelihood > new_likelihood:
                print("Likelihood decreased, this should never happen. There is probably a bug.")
                break
            elif abs(new_loss - loss) < tol:
                print("Reached Convergence on Iteration #" + str(iter + 1))
                break
            else:
                if verbose and (iter % 10 == 0):
                    dev_explained = 1 - (likelihood / mean_likelihood)
                    formatted = "Iteration: {}\nPercent of Deviance Explained: {:.3f}%, Loss:  {:.3f}\n".format(iter, dev_explained*100, new_loss)
                    print(formatted)
                likelihood = new_likelihood
                loss = new_loss

            iter += 1

        # Save main effects and projection matrix
        self.mu = mu
        self.U = U

        # Calculate proportion of deviance explained
        dev_explained = 1 - (likelihood / mean_likelihood)
        self.dev = dev_explained
        
        print("Training Complete. Converged Reached: " + str(not (iter == maxiters)) +
              "\nPercent of Deviance Explained: " + str(dev_explained * 100) + "%")

    
    def transform(self, X):
        """
        Transforms new data using the same model.

        Parameters:
        - X (matrix): New binary data with the same number of features

        Returns:
        - Theta (matrix): Mean centered projection of the natural parameters
        """
        n, d = X.shape
        Q = (2*X) - 1

        Theta_S = (self.m * Q) - np.ones((n, 1)) @ self.mu
        return Theta_S @ self.U

        
    def likelihood(self, X, Theta):
        """
        Compute the log likelihood of Theta as the Bernoulli natural parameters of X

        Parameters:
        - X (matrix): Original binary data
        - Theta (matrix): Estimated natural parameters
        """
        return np.sum(X * Theta - np.log(1 + np.exp(Theta)))
    

    def show_info(self):
        """
        Displays the values of m, k, and % of deviance explained for the chosen model.
        """
        formatted = "Logistic PCA Model w/ m={} and k={}\nProjects the data onto {}-dimensional space, explaining {:.3f}% of the deviance".format(self.m, self.k, self.k, self.dev * 100)
        print(formatted)
    

    def sigmoid(self, X):
        """
        Computes the elementwise sigmoid of a matrix X.

        Parameters:
        - X (matrix): Matrix to apply sigmoid to, clipped to be between +/- m

        Returns:
        - A (matrix): Matrix with sigmoid funciton applied elementwise
        """
        clipped_X = np.clip(X, -1 * self.m, self.m)
        t = np.exp(-clipped_X)
        return 1.0 / (1.0 + t)
    

    def logit(self, X):
        """
        Computes the elementwise logit of a matrix X.

        Parameters:
        - X (matrix): Matrix to apply logit to

        Returns:
        - L (matrix): Matrix with logit applied, bound between +/- m
        """
        logit = np.log((X + 0.00001) / (1.00001 - X)) # Add 0.00001 to avoid issues when no/all rows have a particular feature
        clipped = np.clip(logit, -1 * self.m, self.m)
        return clipped
