def cluster_features(self, features_to_include=None, number_of_features=2, feature=0,
                         min_cluster_size=5, min_samples=None, cluster_selection_epsilon=0.0,
                         cluster_selection_method='eom', allow_single_cluster=False, prediction_data=False):
        """
        A method for clustering of the extracted features  as returned by `SpectralExtraction.process_data`.

        Additional:
            https://hdbscan.readthedocs.io/en/latest/index.html

            https://towardsdatascience.com/a-gentle-introduction-to-hdbscan-and-density-based-clustering-5fd79329c1e8

            https://pberba.github.io/stats/2020/01/17/hdbscan/

        Args:
            feature (int): which feature to use for clustering. This only applies if `number_of_features=1`

            prediction_data (bool): Whether to generate extra cached data for predicting labels or
                membership vectors few new unseen points later. If you wish to
                persist the clustering object for later re-use you probably want
                to set this to True. Defaults to False

            number_of_features (int): How many features to use for the clustering. Goes from deepest to smallest e.g. if
                `number_of_features = 2` then it will use the 1st and 2nd deepest features.

            features_to_include (list): A list of length 9 consisting of 1's and 0's used to switch a given feature on or
                off

            min_cluster_size (int): The minimum size of clusters; single linkage splits that contain
                fewer points than this will be considered points "falling out" of a
                cluster rather than a cluster splitting into two new clusters.

            min_samples (int): The number of samples in a neighbourhood for a point to be considered a core point.

            cluster_selection_epsilon (float): A distance threshold. Clusters below this value will be merged.

            cluster_selection_method (str): The method used to select clusters from the condensed tree. The
                standard approach for HDBSCAN* is to use an Excess of Mass algorithm
                to find the most persistent clusters. Alternatively you can instead
                select the clusters at the leaves of the tree -- this provides the
                most fine grained and homogeneous clusters. Options are: ``eom`` or ``leaf``

            allow_single_cluster (bool): By default HDBSCAN* will not produce a single cluster, setting this
                to True will override this and allow single cluster results in
                the case that you feel this is a valid result for your dataset.

        Returns:
            -99: when the clustering can not be completed (a message is printed)

            class_labels (ndarry): An array of the same dimensions (non spectral) as the input data containing the
                resultant class labels as returned by `HDBSCAN`. The input data are the extracted spectral features.

            class_probabilities (ndarray): An array of the same dimensions (non spectral) as the input data containing
                the cluster probabilities as returned by `HDBSCAN`. The input data are the extracted spectral features.

        """
        # first check if any feature info exists
        if features_to_include is None:
            features_to_include = [1, 0, 0, 0, 0, 0, 0, 0, 0]
        if self.spectral_features is None:
            print("You need to do a feature extraction first using process_data()")
            self.extract_features()

        # write some information to the state variables
        self.features_to_include = features_to_include
        self.number_of_features = number_of_features
        self.feature = feature
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.cluster_selection_epsilon = cluster_selection_epsilon
        self.cluster_selection_method = cluster_selection_method
        self.allow_single_cluster = allow_single_cluster
        self.prediction_data = prediction_data

        # here we need to get what we are clustering. The self.feature_info parameter will contain a number of
        # results who's dimension is dependent on the dimensions of the original data and how many features were
        # requested by the user. If it is 2D then its for a single spectrum (so this wont actually work), if its 3D then
        # its for a collection of spectra (aka a TSG dataset), if its 4D then its for an image

        # get the data dimensions and shape
        dims = self.spectral_features.ndim
        original_shape = self.spectral_features.shape

        # see what features are being requested
        if np.where(np.array(features_to_include) > 0)[0].size > 0:
            what_features = np.where(np.array(features_to_include) > 0)[0]
        else:
            print("No features have been selected for clustering, so I am going to go back to sleep")
            return -99

        # if the number of features to cluster over is greater than the actual number of features extracted teh reset to
        # the maximum number of features extracted
        if number_of_features > self.max_features:
            number_of_features = self.max_features

        # okay so a couple of special cases now
        # If the number_of_features is equal to 1 then we need to see what the feature to use is
        if number_of_features == 0:
            print("Yeah that's not going to work, you need at least 1 feature to work with")
            return -99

        if number_of_features == 1:
            # special case!!
            if dims == 2:
                print("Yeah, so I cant do this. Its only a single spectrum. So nothing to cluster")
                return -99
            elif dims == 3:
                data = self.spectral_features[:, what_features, feature]
                data = np.reshape(data, [data.shape[0], data.shape[1]])
            elif dims == 4:
                data = self.spectral_features[:, :, what_features, feature]
                data = np.reshape(data, [data.shape[0] * data.shape[1], data.shape[2]])
            else:
                print("I have no idea what this is! It has to many dimensions")
                return -99
            if data.shape[1] == 1:
                # we are effectively making a second feature that is simply the index of the data point
                linear_array = np.arange(data.shape[0])
                data = np.transpose(np.stack((linear_array, data[:, 0])))
        else:
            if dims == 2:
                print("Yeah, so I cant do this. Its only a single spectrum. So nothing to cluster")
                return -99
            elif dims == 3:
                data = self.spectral_features[:, np.where(np.array(features_to_include) > 0)[0], :number_of_features]
                data = np.reshape(data, [data.shape[0], data.shape[1] * data.shape[2]])
            elif dims == 4:
                data = self.spectral_features[:, :, np.where(np.array(features_to_include) > 0)[0], :number_of_features]
                data = np.reshape(data, [data.shape[0] * data.shape[1], data.shape[2] * data.shape[3]])
            else:
                print("I have no idea what this is! It has to many dimensions")
                return -99

        #
        scalar = StandardScaler()
        scalar.fit(data)
        self.scalar = scalar
        scaled_data = scalar.transform(data)

        # set up the HDBSCAN Class & fit the data
        clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples, gen_min_span_tree=True,
                                    cluster_selection_epsilon=cluster_selection_epsilon,
                                    cluster_selection_method=cluster_selection_method,
                                    allow_single_cluster=allow_single_cluster,
                                    prediction_data=prediction_data, core_dist_n_jobs=1).fit(scaled_data)

        self.clusterer = clusterer

        # Fit the data
        class_labels = clusterer.labels_
        class_probabilities = clusterer.probabilities_
        if dims == 4:
            class_labels = np.reshape(class_labels, [original_shape[0], original_shape[1]])
            class_probabilities = np.reshape(class_probabilities, [original_shape[0], original_shape[1]])

        return class_labels, class_probabilities