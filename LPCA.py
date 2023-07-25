import time
import h5py
import numpy as np
from joblib import cpu_count
from joblib import Parallel, delayed

class LogisticPCA():
    def __init__(self, m=0, k=0, verbose=False, verbose_interval=100):
        """
        Initializes Logistic PCA object.

        Parameters:
        - m (float): Scale factor of matrix Q. As m->infinity, the model becomes more saturated
        - k (int): Number of principle components to keep
        """
        self.m = m
        self.k = k
        self.verbose = verbose
        self.verbose_interval = verbose_interval

        self.train_time = None
        self.converged = None
        self.dev = None
        self.mu = None
        self.U = None


    def set_verbose(self, verbose, verbose_interval):
        """
        Updates the verbose status for the current model

        Parameters:
        - verbose (boolean): If true, will print messages throughout training
        - verbose_interval (boolean): Specifies how many iterations should occur between verbose messages
        """
        self.verbose = verbose
        self.verbose_interval = verbose_interval

    
    def fit(self, X, tol=1e-4, maxiters=1000):
        """
        Fits the Logistic PCA model.

        Parameters:
        - X (matrix): Data matrix containing only bernoulli RVs
        - tol (float): Converge criteria. Minimum allowed difference between previous loss and current loss
        - maxiters (int): Maximum number of iterations to run if converge criteria is never reached
        """
        start_time = time.time()

        # Parameter Initializations
        self.converged = False

        # n: # of observations, p: # of features
        n, _ = X.shape  

        # Natural parameters of the saturated model Theta_S
        Q = (2*X) - 1
        Theta_S = self.m * Q
        frobenius = np.linalg.norm(Q)  # Used for loss calculation

        # Initialize U to the k right singular values of Q
        # Note this assumes there are more observations (rows) than features (columns)
        U = np.linalg.svd(Q, full_matrices=False)[2].T[:, :self.k]

        # Initialize mu to logit(X_bar)
        mu = np.mean(X, axis=0).reshape(-1, 1).T
        mu = self.logit(mu)
        Mu = np.ones((n, 1)) @ mu
        mean_likelihood = self.likelihood(X, Mu)

        # Initialize Theta
        Theta = Mu + ((Theta_S - Mu) @ U @ U.T)

        # Initialize likelihood and loss
        likelihood = self.likelihood(X, Theta)
        prev_loss = (-likelihood) / frobenius

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
            eigenvectors = np.linalg.eigh(argmax)[1]  # eigh solves problem of complex eigenvectors/values
            U = eigenvectors[:, -self.k:]  # Returns eigenvalues in ascending order, NOT descending

            # Converge criteria
            Theta = Mu + ((Theta_S - Mu) @ U @ U.T)
            new_likelihood = self.likelihood(X, Theta)
            new_loss = (-new_likelihood) / frobenius
            change = prev_loss - new_loss

            if likelihood > new_likelihood:
                self._verbose_local_minima(iter)
                likelihood = new_likelihood
                break
            
            elif change < tol:
                self.converged = True
                self._verbose_converged(iter)
                likelihood = new_likelihood
                break

            else:
                dev_explained = 1 - (likelihood / mean_likelihood)
                self._verbose_iter(iter, dev_explained, new_likelihood, change)

            # Update and increment
            likelihood = new_likelihood
            prev_loss = new_loss
            iter += 1

        end_time = time.time()
        self.train_time = end_time - start_time

        # Save main effects and projection matrix
        self.mu = mu
        self.U = U

        # Calculate proportion of deviance explained
        self.dev = 1 - (likelihood / mean_likelihood)

        self._verbose_train_complete()

    
    def _verbose_iter(self, iter, dev, lik, change):
        if self.verbose and iter % self.verbose_interval == 0:
            print(f"Iteration: {iter}\nPercent of Deviance Explained: {np.round(dev * 100, decimals = 3)}%\n" +
                  f"Log Likelihood: {np.round(lik, decimals=2)}, Loss Trace: {change}\n")
            

    def _verbose_local_minima(self, iter):
        if self.verbose:
            print(f"Likelihood decreased, local minima found on Iteration #{iter + 1}")
            
    
    def _verbose_converged(self, iter):
        if self.verbose:
            print(f"Reached Convergence on Iteration #{iter + 1}")
    

    def _verbose_train_complete(self):
        if self.verbose:
            print(f"Training Complete. Converged Reached: {self.converged}\n" +
                f"Percent of Deviance Explained: {np.round(self.dev * 100, decimals=3)} %\n" +
                f"Total Training Time: {np.round(self.train_time)}s")
    

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
    

    def reconstruct(self, Theta):
        """
        Reconstructs the data back to its original dimension

        Parameters:
        - Theta (matrix): Natural prameters of reduced data

        Returns:
        - X (matrix): Data reconstructed in its original dimension
        """
        n, _ = Theta.shape

        Mu = np.ones((n, 1)) @ self.mu
        X = Mu + Theta @ self.U.T

        return X

        
    def likelihood(self, X, Theta):
        """
        Compute the log likelihood of Theta as the Bernoulli natural parameters of X

        Parameters:
        - X (matrix): Original binary data
        - Theta (matrix): Estimated natural parameters
        """
        return np.sum(X * Theta - np.log(1 + np.exp(Theta)))
    

    def deviance(self, X):
        """
        Compute the proportion of deviance of X explained by the model. Designed to take

        Parameters:
        - X (matrix): New Data
        """
        n, d = X.shape

        mu = np.mean(X, axis=0).reshape(-1, 1).T
        mu = self.logit(mu)
        data_Mu = np.ones((n, 1)) @ mu
        mean_likelihood = self.likelihood(X, data_Mu)

        Theta = self.transform(X)
        X_reconstructed = self.reconstruct(Theta)

        likelihood = self.likelihood(X, X_reconstructed)

        return 1 - (likelihood / mean_likelihood)
    

    def crossval(self, X, target_dev, m_range=list(range(6,17,1)), k_range=None, nfolds=5, tol=1e-2, maxiters=100, verbose=False, n_jobs=1):
        """
        Use cross-validation to select the smallest model to achieve the desired deviance explained. 
        Automatically sets the hyperparameters to the best generalizing model and retrains on all of the data.

        Parameters:
        - X (matrix): The data
        - target_dev (float): Proportion of deviance looking to be explained by the model [0, 1]
        - m_range (list, type int): m values to check
        - k_range (list): Two element list. First is the bottom bound of k, second is the upper bound
        - tol (float): Minimum allowed difference between losses in the fitting method
        - nfolds (int): Number of folds to use in the cross validation process
        - verbose (boolean): When true, prints information on each cross validation fold
        - n_jobs (int): Number of CPU cores to train on. Can significantly speed training (defaults to 1, -1 for all available cores)
        """
        n, d = X.shape

        # Make sure the matrix is even sensical to reduce
        if d <= 3:
            print("Dimension is too small to reduce.")
            return
        
        if verbose:
            print("Searching for the best value of m and smallest value of k for deviance=" + str(target_dev) + " over "+ str(nfolds) + " folds...")

        # Initialize
        if n_jobs == -1:
            n_jobs = cpu_count()
        
        if k_range == None:
            low = 1
            high = d
        else:
            low = k_range[0]
            high = k_range[1]

        best_m = 0
        best_dev = 0

        # m values
        m_vals = m_range

        # Binary search for k
        while low < high:
            improved = False
            mid = (low + high) // 2
            self.k = mid

            # Check m values for a given k
            # The additional loop is a trick to break out of parallel compute when any solution is found
            m_batches = np.array_split(m_vals, np.ceil(len(m_vals)/n_jobs))

            for m_batch in m_batches:
                def fit_with_m(m_val):
                    # Create a temp class
                    tmp = self.__class__(m=m_val, k=mid)

                    start_time = time.time()

                    # New folds for each m value
                    folds = tmp.split_data(X, nfolds)
                    deviances = []

                    for i, fold in enumerate(folds):
                        # Create the training matrix
                        train = np.concatenate([f for j, f in enumerate(folds) if j != i])
                        tmp.fit(train, tol=tol, maxiters=maxiters)

                        # Apply it to the new fold to test generalization
                        deviances.append(tmp.deviance(fold))
                    
                    avg_dev = sum(deviances) / nfolds

                    end_time = time.time()
                    total_time = end_time - start_time

                    return m_val, avg_dev, total_time

                results = Parallel(n_jobs=n_jobs)(delayed(fit_with_m)(m) for m in m_batch)
                results.sort(key=lambda x: x[0])

                for m, avg_dev, total_time in results:
                    if verbose:
                        print(f"Checked m={m} and k={self.k}. Avg Deviance: {avg_dev}. Training Time: {total_time}")

                    if avg_dev >= target_dev:
                        best_m = m
                        high = mid
                        improved = True
                        best_dev = avg_dev
                        break  # This will break the inner loop over the batch
                else:
                    continue  # This will break the outer loop if the target_dev is found
                break  
            
            if not improved:
                low = mid + 1

        # Update k and m
        self.k = low
        self.m = best_m

        if verbose:
            print("Found m=" + str(best_m) + " and k=" + str(self.k) + " to be the best value. Deviance: " + str(best_dev))
            print("Retraining on all data with best hyperparameters")

        self.fit(X, tol=tol, maxiters=maxiters)
    

    def split_data(self, X, k):
        """
        Splits the data into k folds.

        Parameters:
        - X (matrix): Data
        - k (int): Number of folds to create
        """
        # Copy and shuffle
        A = X.copy()
        np.random.shuffle(A)  

        folds = np.array_split(A, k)
        return folds
    

    def show_info(self):
        """
        Displays the values of m, k, and % of deviance explained for the chosen model.
        """
        print(self)
    

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
    

    def save_model(self, file_path):
        """
        Stores the model at the specified path in h5 format

        Parameters:
        - file_path (string): Path to the file to create or write to
        """
        meta = [self.m, self.k, self.dev, self.train_time, self.converged, self.verbose, self.verbose_interval]
        with h5py.File(file_path, "w") as f:
            f.create_dataset("meta", data=meta)
            f.create_dataset("U", data=self.U)
            f.create_dataset("mu", data=self.mu)

    
    def load_model(self, file_path):
        """
        Loads an existing model from the specified h5 file

        Parameters:
        - file_path (string): Path to a model's h5 file
        """
        with h5py.File(file_path, "r") as f:
            meta = f["meta"][()]
            self.U = f["U"][()]
            self.mu = f["mu"][()]
        
        self.m = int(meta[0])
        self.k = int(meta[1])
        self.dev = meta[2]
        self.train_time = meta[3]
        self.converged = bool(meta[4])
        self.verbose = bool(meta[5])
        self.verbose_interval = int(meta[6])

    
    def __str__(self):
        formatted = f"Logistic PCA Model w/ m={self.m} and k={self.k}\nExplains {np.round(self.dev * 100, decimals=1)}% of the deviance"
        return formatted
