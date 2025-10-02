from sklearn import base, ensemble, linear_model
from sklearn.metrics import pairwise
import numpy as np
from sklearn.preprocessing import KernelCenterer
import discotec
from .kernelkmeans import PUKernelKMeans

# For parameter validation
from numbers import Integral, Real
from sklearn.utils._param_validation import Interval
from sklearn.utils.validation import check_random_state



class PUMKLKmeans(base.BaseEstimator, base.ClusterMixin):
    _parameter_constraints = {
        "kkmeans_model":[PUKernelKMeans],
        "n_candidates":[Interval(Integral, 1, None, closed="left")],
        "n_initial_candidates":[Interval(Integral, 1, None, closed="left")],
        "max_kernels":[Interval(Integral, 2, None, closed="left")],
        "n_regressors":[Interval(Integral, 1, None, closed="left")],
        "exploration":[Interval(Real, 0, None, closed="left")],
        "discotec_target":[bool],
        "use_native_kernel":[bool],
        "verbose":[bool],
        "random_state":["random-state"]
    }
    def __init__(self, kkmeans_model, n_candidates=1000, n_initial_candidates=100, max_kernels=5, n_regressors=150, exploration = 0.1, kernel_params=[{"metric":"linear"}], discotec_target=True, use_native_kernel=True, verbose=False, random_state=None):
        """
        Creates a multi-kernel learning regressor to optimise the kernel weights used to train a positive-unlabelled kernel kmeans algorithm.

        The score used for regression is the regularised discotec. It is possible to discard the discotec score and just evaluate the percentage of satisfied constraints.

        Parameters
        ----------

        kkmeans_model: PUKernelKMeans
            A variant of kernel kmeans where the initialisation is done acccording to some positive-unlabelled targets.
        
        n_candidates: int, default=1000
            Number of sampled candidate weights for the multi kernel to generate and evaluate before accepting a new one
        
        n_initial_candidates: int, default=100
            Number of sampled candidate weights that were used for kernel kmeans before using the regression to accept further candidates.

        max_kernels: int, default=5
            Maximal number of non-zero weights for the kernel mixture
        
        n_regressors: int, default=150
            Number of estimators for the bagging regression when evaluating a candidate weight.
        
        exploration: float, default=0.1
            Coefficient for the standard deviation when adding to the predicted score by the bagged regressor
        
        kernel_params: list of dict, default=[{"metric":"linear"}]
            Kernel parameters. Each dictionary is a set of arguments that will be passed to `sklearn.metrics.pairwise_kernels`.
            The MKL will use as many kernels as the length of this list.

        discotec_target: bool, default=True
            For using the regularised discotec score (True) for regression or just the average constraint satisfaction (False)

        use_native_kernel: bool, default=True
            Whether to incorporate each kernel as a standalone among the initial candidates.

        verbose: bool, default=False
            For printing fit messages.

        random_state: RandomState, int, None, default=None
            For reproducibility purposes.

        """
        self.kkmeans_model = kkmeans_model
        self.n_candidates = n_candidates
        self.kernel_params = kernel_params
        self.max_kernels = max_kernels
        self.n_initial_candidates = n_initial_candidates
        self.n_regressors = n_regressors
        self.exploration = exploration
        self.discotec_target = discotec_target
        self.use_native_kernel = use_native_kernel
        self.verbose = verbose
        self.random_state = random_state


    def fit(self, X, y):
        """
        Description TODO

        Parameters
        ----------

        X: array-like (n_samples, n_features)
        y: array-like (n_samples)
            Positive [1] / Unlabelled [0] values for initialisation the kernel KMeans algorithm.
        """

        # Perform data validation (function call only valid for scikit learn <1.6)
        X, y = self._validate_data(X, y, accept_sparse=False, ensure_2d=True)
        random_state = check_random_state(self.random_state)
        
        self.n_kernels_ = len(self.kernel_params)
        if self.use_native_kernel:
            assert self.n_kernels_ <= self.n_initial_candidates, "It is impossible to have fewer initial candidates than kernel when standalone kernels are part of the initial candidates."
        assert self.max_kernels <= self.n_kernels_, "The number of maximal non-zero weights in the kernel mixture should be less than the number of kernels."

        # Compute kernels
        kernels = [self._compute_kernel(X, **params) for params in self.kernel_params]

        # Create initial candidates
        candidates = np.zeros((self.n_candidates, self.n_kernels_))

        if self.use_native_kernel:
            # We have initial deterministic candidates for evaluating the performance on each kernel alone
            candidates[:self.n_kernels_] = np.eye(self.n_kernels_)
            n_init=self.n_kernels_
        else:
            n_init=0
        if n_init<self.n_initial_candidates:
            random_candidates = self._generate_candidates(self.n_initial_candidates-n_init, random_state)
            candidates[n_init:self.n_initial_candidates] = random_candidates

        # Create a bagging regressor that will try to predict the PU constraint satisfaction
        # given kernel weights
        bagging_regressor = ensemble.BaggingRegressor(linear_model.LinearRegression(), n_estimators=self.n_regressors, random_state=random_state)

        clusterings = np.zeros((self.n_candidates, len(X)), dtype=int)
        scores_discotec = np.ones(self.n_candidates)
        scores_constraints = np.zeros(self.n_candidates)

        

        for i in range(self.n_candidates):
            # We start by fitting kernel kmeans using the current candidate
            
            kernel_matrix = sum([kernels[j]*candidates[i,j] for j in range(self.n_kernels_)])

            # We perform kernel kmeans
            clusterings[i] = self.kkmeans_model.fit_predict(kernel_matrix, y)

            scores_constraints[i] = self._constraint_satisfaction(y, clusterings[i])
            if self.verbose:
                print(f"Candidate {i}/{self.n_candidates}. Kernel weights are {candidates[i]}. Final cluster proportions are {np.bincount(clusterings[i])}. KTA: {scores_constraints[i]}")

            
            if self.discotec_target and i>=self.n_initial_candidates-1:
                # To evaluate potential candidates, we first compute the discotec score of our new set of solutions
                consensus_matrix = discotec.compute_consensus_matrix(clusterings[:i+1])
                binary_consensus = (consensus_matrix>consensus_matrix.mean()).astype(float)
                scores_discotec[:i+1] = discotec.compute_tv_ranking(clusterings[:i+1], binary_consensus)

            

            if i >= self.n_initial_candidates-1 and i<self.n_candidates-1:

                # We now have performed kernel kmeans for all initial candidates
                # So we must add new candidates
                potential_candidates = self._generate_candidates(1000, random_state=random_state)
                
                bagging_regressor.fit(candidates[:i+1], scores_constraints[:i+1]-scores_discotec[:i+1])
                score_predictions = np.concatenate([base_regressor.predict(potential_candidates).reshape((-1,1)) for base_regressor in bagging_regressor.estimators_], axis=1)

                selected_candidate = np.argmax(score_predictions.mean(1)+self.exploration * score_predictions.std(1))
                candidates[i+1] = potential_candidates[selected_candidate]

            

        # Finally, we will keep the model that had the best PU constraint satisfaction
        best_model_index = np.argmax(scores_constraints-scores_discotec)
        if self.verbose:
            print("Final scores are:", scores_constraints-scores_discotec)
            print("Selected model is:", best_model_index)
        self.labels_ = clusterings[best_model_index]
        self.kernel_weights_ = candidates[best_model_index]
        self.input_data_ = X

        return self

    def predict(self, X, y=None):
        training_kernel = np.zeros((len(self.labels_), len(self.labels_)))
        kernel = np.zeros((len(X), len(self.labels_)))
        for i, (w, params) in enumerate(zip(self.kernel_weights_, self.kernel_params)):
            original_kernel = pairwise.pairwise_kernels(self.input_data_, **params)
            centerer = KernelCenterer()
            original_kernel = centerer.fit_transform(original_kernel)
            new_kernel = pairwise.pairwise_kernels(X, self.input_data_, **params)
            new_kernel = centerer.transform(new_kernel)
            new_kernel = len(original_kernel)*new_kernel / np.trace(original_kernel)
            kernel += w * new_kernel
            training_kernel += w * len(original_kernel)*original_kernel/np.trace(original_kernel)

        self.kkmeans_model.fit(training_kernel, y)
        prediction = self.kkmeans_model.predict(training_kernel, kernel)

        return prediction
    
    def fit_predict(self, X, y = None):
        return self.fit(X, y).labels_
    
    def _compute_kernel(self, X, **params):
        # Compute the kernel
        kernel = pairwise.pairwise_kernels(X, **params)

        # Normalise it
        kernel_centerer = KernelCenterer()
        kernel = kernel_centerer.fit_transform(kernel)

        return len(kernel)*kernel/np.trace(kernel)

    def _constraint_satisfaction(self, y, y_pred):
        # The greater the better
        positive_clustering = y_pred[y==1]
        cluster_sizes = np.bincount(positive_clustering)
        return np.square(cluster_sizes).sum() / (len(positive_clustering)**2)
    
    def _generate_candidates(self, n_candidates, random_state):
        # We add along this random candidates the init_candidates
        # A random candidate is a vector in [0,1]^d (non-stochastic)
        # Some of its entries are cancelled such that [2, nonzero] entries are non-null
        random_candidates = random_state.uniform(size=(n_candidates, self.n_kernels_))
        number_of_preserved_kernels = random_state.randint(2, self.max_kernels+1, size=n_candidates)
        mask = 1-np.triu(np.ones((self.n_kernels_+1, self.n_kernels_)))
        mask = mask[number_of_preserved_kernels]
        # Shuffle each row
        mask = np.apply_along_axis(random_state.permutation, 1, mask)
        random_candidates = random_candidates * mask
        return random_candidates
