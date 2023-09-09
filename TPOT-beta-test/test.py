from tpot import TPOTClassifier
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import numpy as np
digits = load_digits()
features = digits.data
targets = digits.target
print(type(features), features.shape)
print(type(targets), targets.shape)
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target,
                                                    train_size=0.75, test_size=0.25, random_state=42)
#默认搜索空间配置
# tpot = TPOTClassifier(generations=5, population_size=50, verbosity=2, random_state=42)
# t pot.fit(X_train, y_train)
# print(tpot.score(X_test, y_test))
# tpot.export('tpot_digits_pipeline.py')

#定制初始operators
tpot_pre_config_dict = {

    # Classifiers
    # 'sklearn.naive_bayes.GaussianNB': {
    # },

    'sklearn.naive_bayes.BernoulliNB': {
        #'alpha': [1e-4, 1e-3, 1e-2, 1e-1, 1., 10., 100.],
        'alpha': [1e-1, 1., 10., 100.],
        'fit_prior': [False]
    },

    #
    # # Preprocesssors
    # 'sklearn.preprocessing.Binarizer': {
    #     'threshold': np.arange(0.0, 1.01, 0.05)
    # },

    # 'sklearn.decomposition.FastICA': {
    #     'tol': np.arange(0.0, 1.01, 0.05)
    # },
    #
    # 'sklearn.cluster.FeatureAgglomeration': {
    #     'linkage': ['ward', 'complete', 'average'],
    #     'affinity': ['euclidean', 'l1', 'l2', 'manhattan', 'cosine']
    # },
    #
    # 'sklearn.preprocessing.MaxAbsScaler': {
    # },

    # 'sklearn.preprocessing.MinMaxScaler': {
    # },
    #
    # 'sklearn.preprocessing.Normalizer': {
    #     'norm': ['l1', 'l2', 'max']
    # },
    #
    # 'sklearn.kernel_approximation.Nystroem': {
    #     'kernel': ['rbf', 'cosine', 'chi2', 'laplacian', 'polynomial', 'poly', 'linear', 'additive_chi2', 'sigmoid'],
    #     'gamma': np.arange(0.0, 1.01, 0.05),
    #     'n_components': range(1, 11)
    # },
    #
    # 'sklearn.decomposition.PCA': {
    #     'svd_solver': ['randomized'],
    #     'iterated_power': range(1, 11)
    # },
    #
    # 'sklearn.preprocessing.PolynomialFeatures': {
    #     'degree': [2],
    #     'include_bias': [False],
    #     'interaction_only': [False]
    # },
    #
    # 'sklearn.kernel_approximation.RBFSampler': {
    #     'gamma': np.arange(0.0, 1.01, 0.05)
    # },

    # 'sklearn.preprocessing.RobustScaler': {
    # },
    #
    # 'sklearn.preprocessing.StandardScaler': {
    # },
    #
    #
    # # Selectors
    # 'sklearn.feature_selection.SelectFwe': {
    #     'alpha': np.arange(0, 0.05, 0.001),
    #     'score_func': {
    #         'sklearn.feature_selection.f_classif': None
    #     }
    # },
    #
    # 'sklearn.feature_selection.SelectPercentile': {
    #     'percentile': range(1, 100),
    #     'score_func': {
    #         'sklearn.feature_selection.f_classif': None
    #     }
    # },
    #
    # 'sklearn.feature_selection.VarianceThreshold': {
    #     'threshold': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.2]
    # },

}




#定制搜索空间配置
tpot_config = {
    # 'sklearn.naive_bayes.GaussianNB': {
    # },
    'sklearn.naive_bayes.BernoulliNB': {
        #'alpha': [1e-3, 1e-2, 1e-1, 1., 10., 100.],
        'alpha': [1e-1, 1., 10., 100.],
        'fit_prior': [True, False]
    },
    # 'sklearn.neural_network.MLPClassifier': {
    #     'alpha': [1e-4, 1e-3, 1e-2, 1e-1],
    #     'learning_rate_init': [1e-3, 1e-2, 1e-1, 0.5, 1.]
    # },
    'sklearn.naive_bayes.MultinomialNB': {
        'alpha': [1e-3, 1e-2, 1e-1, 1., 10., 100.],
        'fit_prior': [True, False]
    },
    # 'sklearn.feature_selection.SelectPercentile': {
    #     'percentile': range(1, 100),
    #     'score_func': {
    #         'sklearn.feature_selection.f_classif': None
    #     }
    # },
}
tpot = TPOTClassifier(generations=5, population_size=50, verbosity=2, config_dict=tpot_config,pre_config_dict=tpot_pre_config_dict)
tpot.fit(X_train, y_train)
print(tpot.score(X_test, y_test))
tpot.export('tpot_digits_pipeline_betatest.py')