from joblib import Parallel, delayed
from joblib import cpu_count
from scipy import linalg
import numpy as np
import time
import h5py

class LogisticPCA():
    def __init__(self, m=0, k=0):
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

        start_time = time.time()

        # Parameter Initializations
        # n: # of observations, p: # of features
        n, d = X.shape  

        # Natural parameters of the saturated model Theta_S
        Q = (2*X) - 1
        Theta_S = self.m * Q

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

        # Initialize likelihood
        likelihood = self.likelihood(X, Theta)

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
            eigenvectors = linalg.eigh(argmax)[1]  # eigh solves problem of complex eigenvectors/values
            U = eigenvectors[:, -self.k:]  # Returns eigenvalues in ascending order, NOT descending

            # Converge criteria
            Theta = Mu + ((Theta_S - Mu) @ U @ U.T)
            new_likelihood = self.likelihood(X, Theta)

            if abs(new_likelihood - likelihood) < tol:
                if verbose:
                    print("Reached Convergence on Iteration #" + str(iter + 1))
                break
            elif likelihood > new_likelihood:
                if verbose:
                    print("Likelihood decreased, local minima found on Iteration #" + str(iter + 1))
                likelihood = new_likelihood
                break
            else:
                if verbose and (iter % 10 == 0):
                    dev_explained = 1 - (likelihood / mean_likelihood)
                    formatted = "Iteration: {}\nPercent of Deviance Explained: {:.3f}%, Log Likelihood: {:.2f}\n".format(iter, dev_explained*100, new_likelihood)
                    print(formatted)

            likelihood = new_likelihood
            iter += 1

        end_time = time.time()
        total_train_time = end_time - start_time

        # Save main effects and projection matrix
        self.mu = mu
        self.U = U

        # Calculate proportion of deviance explained
        dev_explained = 1 - (likelihood / mean_likelihood)
        self.dev = dev_explained
        
        if verbose:
            print("Training Complete. Converged Reached: " + str(not (iter == maxiters)) +
                "\nPercent of Deviance Explained: " + str(dev_explained * 100) + "%\n" +
                "Total Training Time: " + str(total_train_time))

    
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

        found_Mu = np.ones((n, 1)) @ self.mu
        Theta = found_Mu + self.transform(X) @ self.U.T

        likelihood = self.likelihood(X, Theta)

        return 1 - (likelihood / mean_likelihood)
    

    def crossval(self, X, target_dev, nfolds=5, tol=0.01, maxiters=100, verbose=False, n_jobs=1):
        """
        Use cross-validation to select the smallest model to achieve the desired deviance explained. 
        Automatically sets the hyperparameters to the best generalizing model and retrains on all of the data.

        Parameters:
        - X (matrix): The data
        - target_dev (float): Proportion of deviance looking to be explained by the model [0, 1]
        - tol (float): Minimum allowed difference between log likelihoods in the fitting method
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
        best_m = 0
        low = 1
        high = d
        best_dev = 0

        # m values
        m_vals = list(range(6, 17, 1))

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
    

    def speed_test(self, X, Ks, target_dev, nfolds=5, tol=0.01, maxiters=100, verbose=False, n_jobs=-1):
        """
        A test to see the total time taken on preset m and k values

        Parameters:
        - X (matrix): The data
        - Ks (list): k values to test
        - target_dev (float): Proportion of deviance looking to be explained by the model [0, 1]
        - tol (float): Minimum allowed difference between log likelihoods in the fitting method
        - nfolds (int): Number of folds to use in the cross validation process
        - verbose (boolean): When true, prints information on each cross validation fold
        """
        
        # m values
        m_vals = list(range(6, 17, 1))

        # Update CPUs
        if n_jobs == -1:
            n_jobs = cpu_count()

        # Testing all k vals
        for k in Ks:
            if verbose:
                print("Testing speed of k=" + str(k) + " and deviance=" + str(target_dev) + " over "+ str(nfolds) + " folds...")

            start_time = time.time()

            # Check m values for a given k
            # The additional loop is a trick to break out of parallel compute when any solution is found
            m_batches = np.array_split(m_vals, np.ceil(len(m_vals)/n_jobs))

            for m_batch in m_batches:
                def fit_with_m(m_val):
                    # Create a temp class
                    tmp = self.__class__(m=m_val, k=k)

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

                    return m_val, avg_dev

                results = Parallel(n_jobs=n_jobs)(delayed(fit_with_m)(m) for m in m_batch)
                results.sort(key=lambda x: x[0])

                for m, avg_dev in results:
                    if verbose:
                        print(f"Checked m={m} and k={self.k}. Avg Deviance: {avg_dev}.")

                    if avg_dev >= target_dev:
                        break  # This will break the inner loop over the batch
                else:
                    continue
                break
            
            end_time = time.time()
            total_time = end_time - start_time
            print(f"Speed of k={self.k}. Total Training Time: {total_time}")


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
    

    def save_model(self, file_path):
        with h5py.File(file_path, "w") as f:
            f.create_dataset("dev", data=self.dev)
            f.create_dataset("m", data=self.m)
            f.create_dataset("k", data=self.k)
            f.create_dataset("U", data=self.U)
            f.create_dataset("mu", data=self.mu)

    
    def load_model(self, file_path):
        with h5py.File(file_path, "r") as f:
            self.dev = f["dev"][()]
            self.m = f["m"][()]
            self.k = f["k"][()]
            self.U = f["U"][()]
            self.mu = f["mu"][()]
