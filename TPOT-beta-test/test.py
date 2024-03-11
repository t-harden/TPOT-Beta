from tpot import TPOTClassifier
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import numpy as np
import pickle
import time

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
# tpot_pre_config_dict = {
    #
    # Classifiers
    # 'sklearn.naive_bayes.GaussianNB': {
    # },
    #
    # 'sklearn.naive_bayes.BernoulliNB': {
    #     #'alpha': [1e-4, 1e-3, 1e-2, 1e-1, 1., 10., 100.],
    #     'alpha': [1e-1, 1., 10., 100.],
    #     'fit_prior': [False]
    # },
    #
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
#
# }

tpot_pre_config_dict = {'sklearn.ensemble.RandomForestClassifier': {'bootstrap': [False], 'max_features': [0.7000000000000001, 0.6500000000000001, 0.2, 0.25], 'min_samples_leaf': [17, 2, 7], 'min_samples_split': [8, 16, 6, 14], 'random_state': [42]}, 'sklearn.ensemble.GradientBoostingClassifier': {'learning_rate': [0.5], 'max_depth': [9, 10, 6], 'max_features': [0.9500000000000001, 0.9000000000000001, 0.1], 'min_samples_leaf': [4, 20, 14], 'min_samples_split': [3, 13, 6], 'random_state': [42], 'subsample': [0.8, 0.7000000000000001, 0.8500000000000001]}, 'sklearn.ensemble.ExtraTreesClassifier': {'bootstrap': [True], 'criterion': ['entropy'], 'max_features': [0.9000000000000001, 0.8], 'min_samples_leaf': [4], 'min_samples_split': [11, 5], 'random_state': [42]}, 'tpot.builtins.ZeroCount': {}, 'sklearn.preprocessing.StandardScaler': {}, 'sklearn.tree.DecisionTreeClassifier': {'max_depth': [1], 'min_samples_leaf': [4], 'min_samples_split': [5], 'random_state': [42]}, 'sklearn.neural_network.MLPClassifier': {'random_state': [42]}, 'sklearn.preprocessing.RobustScaler': {}, 'sklearn.preprocessing.MinMaxScaler': {}}




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
#读取进化的元数据dict_task_evoldata_20230130[task_id] = [[bootstrap1的进化结果],..., [bootstrap10的进化结果]]" \
#每个bootstrap的进化结果为列表[进化时间, 进化代数, UniformTPipeline列表[元组(sklearn pipeline, str pipeline, internal_cv_score)]]"
with open('/Users/yanggu/Documents/博士/科研/流程推荐/实验/机器学习工作流/AutoML-Systems/Surrogate4AutoML/1CollectMetadata/EvolResult/light_20230130/dict_task_evoldata_20230130_light.pkl', 'rb') as f:
    dict_task_evoldata_20230130_light = pickle.load(f)

print(list(np.arange(0.05, 1.01, 0.05)))
# exit

"构造初始种群"
i = 0
customized_pre_pop = []
for task_id in dict_task_evoldata_20230130_light:
    i += 1
    print("---", i, task_id, "---")
    UniformTPipeline = dict_task_evoldata_20230130_light[task_id][9][2]
    for tup in UniformTPipeline[0:1]:
        customized_pre_pop.append(tup[1])
    if i == 5: break

"数据集嵌入"
# 读取数据集信息和Bootstrap字典dict_task_data[task_id]=[task,dataset,metafe,[bootstrapped_line_numbers]]
with open('/Users/yanggu/Documents/博士/科研/流程推荐/实验/机器学习工作流/AutoML-Systems/Surrogate4AutoML/1CollectMetadata/dict_task_data.pkl', 'rb') as f:
    dict_task_data = pickle.load(f)
print(len(dict_task_data))  # 68个数据集

task_id = 3
X, y = dict_task_data[task_id][0].get_X_and_y()
dataset_embed = dict_task_data[task_id][2]  # (111,)的numpy数组
dataset_embed = np.nan_to_num(dataset_embed)  # 元特征里所有为nan的数被替换为0

#用初始种群
tpot = TPOTClassifier(generations=3, population_size=5, verbosity=2, n_jobs=1, random_state=42, config_dict='TPOT light', template='Selector-Transformer-Classifier',
                      pre_config_dict=None, customized_pre_population=None,
                      surrogate_model='SurrogateNet_model_20240214da与ops注意力+普通双向双层GRU.pt',
                      dataset_embed=dataset_embed)

st_time = time.time()
tpot.fit(X, y)
print("耗时：", time.time()-st_time)
print(tpot.score(X, y))
print("最优sklearn pipeline：", type(tpot.fitted_pipeline_), tpot.fitted_pipeline_)
print("最优deap pipeline：", type(tpot._optimized_pipeline), tpot._optimized_pipeline)
tpot.export('tpot_digits_pipeline_betatest.py')