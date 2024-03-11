# -*- coding: utf-8 -*-

"""This file is part of the TPOT library.

TPOT was primarily developed at the University of Pennsylvania by:
    - Randal S. Olson (rso@randalolson.com)
    - Weixuan Fu (weixuanf@upenn.edu)
    - Daniel Angell (dpa34@drexel.edu)
    - and many more generous open source contributors

TPOT is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as
published by the Free Software Foundation, either version 3 of
the License, or (at your option) any later version.

TPOT is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public
License along with TPOT. If not, see <http://www.gnu.org/licenses/>.

"""
import os

import numpy as np
from deap import tools, gp
from inspect import isclass
from .operator_utils import set_sample_weight
from sklearn.utils import indexable
from sklearn.metrics import check_scoring
from sklearn.model_selection._validation import _fit_and_score

from sklearn.base import clone
from collections import defaultdict
import warnings
from stopit import threading_timeoutable, TimeoutException

import pickle
import torch


def pick_two_individuals_eligible_for_crossover(population):
    """Pick two individuals from the population which can do crossover, that is, they share a primitive.

    Parameters
    ----------
    population: array of individuals

    Returns
    ----------
    tuple: (individual, individual)
        Two individuals which are not the same, but share at least one primitive.
        Alternatively, if no such pair exists in the population, (None, None) is returned instead.
    """
    primitives_by_ind = [set([node.name for node in ind if isinstance(node, gp.Primitive)])
                         for ind in population]
    pop_as_str = [str(ind) for ind in population]

    eligible_pairs = [(i, i+1+j) for i, ind1_prims in enumerate(primitives_by_ind)
                                 for j, ind2_prims in enumerate(primitives_by_ind[i+1:])
                                 if not ind1_prims.isdisjoint(ind2_prims) and
                                    pop_as_str[i] != pop_as_str[i+1+j]]

    # Pairs are eligible in both orders, this ensures that both orders are considered
    eligible_pairs += [(j, i) for (i, j) in eligible_pairs]

    if not eligible_pairs:
        # If there are no eligible pairs, the caller should decide what to do
        return None, None

    pair = np.random.randint(0, len(eligible_pairs))
    idx1, idx2 = eligible_pairs[pair]

    return population[idx1], population[idx2]


def mutate_random_individual(population, toolbox):
    """Picks a random individual from the population, and performs mutation on a copy of it.

    Parameters
    ----------
    population: array of individuals

    Returns
    ----------
    individual: individual
        An individual which is a mutated copy of one of the individuals in population,
        the returned individual does not have fitness.values
    """
    idx = np.random.randint(0,len(population))
    ind = population[idx]
    ind, = toolbox.mutate(ind)
    del ind.fitness.values
    return ind


def varOr(population, toolbox, lambda_, cxpb, mutpb):
    """Part of an evolutionary algorithm applying only the variation part
    (crossover, mutation **or** reproduction). The modified individuals have
    their fitness invalidated. The individuals are cloned so returned
    population is independent of the input population.
    :param population: A list of individuals to vary.
    :param toolbox: A :class:`~deap.base.Toolbox` that contains the evolution
                    operators.
    :param lambda\_: The number of children to produce
    :param cxpb: The probability of mating two individuals.
    :param mutpb: The probability of mutating an individual.
    :returns: The final population
    :returns: A class:`~deap.tools.Logbook` with the statistics of the
              evolution
    The variation goes as follow. On each of the *lambda_* iteration, it
    selects one of the three operations; crossover, mutation or reproduction.
    In the case of a crossover, two individuals are selected at random from
    the parental population :math:`P_\mathrm{p}`, those individuals are cloned
    using the :meth:`toolbox.clone` method and then mated using the
    :meth:`toolbox.mate` method. Only the first child is appended to the
    offspring population :math:`P_\mathrm{o}`, the second child is discarded.
    In the case of a mutation, one individual is selected at random from
    :math:`P_\mathrm{p}`, it is cloned and then mutated using using the
    :meth:`toolbox.mutate` method. The resulting mutant is appended to
    :math:`P_\mathrm{o}`. In the case of a reproduction, one individual is
    selected at random from :math:`P_\mathrm{p}`, cloned and appended to
    :math:`P_\mathrm{o}`.
    This variation is named *Or* beceause an offspring will never result from
    both operations crossover and mutation. The sum of both probabilities
    shall be in :math:`[0, 1]`, the reproduction probability is
    1 - *cxpb* - *mutpb*.
    """
    offspring = []

    for _ in range(lambda_):
        op_choice = np.random.random()
        if op_choice < cxpb:  # Apply crossover
            ind1, ind2 = pick_two_individuals_eligible_for_crossover(population)
            if ind1 is not None:
                ind1_cx, _, evaluated_individuals_= toolbox.mate(ind1, ind2)
                del ind1_cx.fitness.values

                if str(ind1_cx) in evaluated_individuals_:
                    ind1_cx = mutate_random_individual(population, toolbox)
                offspring.append(ind1_cx)
            else:
                # If there is no pair eligible for crossover, we still want to
                # create diversity in the population, and do so by mutation instead.
                ind_mu = mutate_random_individual(population, toolbox)
                offspring.append(ind_mu)
        elif op_choice < cxpb + mutpb:  # Apply mutation
            ind = mutate_random_individual(population, toolbox)
            offspring.append(ind)
        else:  # Apply reproduction
            idx = np.random.randint(0, len(population))
            offspring.append(toolbox.clone(population[idx]))

    return offspring

def initialize_stats_dict(individual):
    '''
    Initializes the stats dict for individual
    The statistics initialized are:
        'generation': generation in which the individual was evaluated. Initialized as: 0
        'mutation_count': number of mutation operations applied to the individual and its predecessor cumulatively. Initialized as: 0
        'crossover_count': number of crossover operations applied to the individual and its predecessor cumulatively. Initialized as: 0
        'predecessor': string representation of the individual. Initialized as: ('ROOT',)

    Parameters
    ----------
    individual: deap individual

    Returns
    -------
    object
    '''
    individual.statistics['generation'] = 0
    individual.statistics['mutation_count'] = 0
    individual.statistics['crossover_count'] = 0
    individual.statistics['predecessor'] = 'ROOT',


def eaMuPlusLambda(population, toolbox, mu, lambda_, cxpb, mutpb, ngen, pbar,
                   stats=None, halloffame=None, verbose=0,
                   per_generation_function=None, log_file=None):
    """This is the :math:`(\mu + \lambda)` evolutionary algorithm.
    :param population: A list of individuals.
    :param toolbox: A :class:`~deap.base.Toolbox` that contains the evolution
                    operators.
    :param mu: The number of individuals to select for the next generation.
    :param lambda\_: The number of children to produce at each generation.
    :param cxpb: The probability that an offspring is produced by crossover.
    :param mutpb: The probability that an offspring is produced by mutation.
    :param ngen: The number of generation.
    :param pbar: processing bar
    :param stats: A :class:`~deap.tools.Statistics` object that is updated
                  inplace, optional.
    :param halloffame: A :class:`~deap.tools.HallOfFame` object that will
                       contain the best individuals, optional.
    :param verbose: Whether or not to log the statistics.
    :param per_generation_function: if supplied, call this function before each generation
                            used by tpot to save best pipeline before each new generation
    :param log_file: io.TextIOWrapper or io.StringIO, optional (defaul: sys.stdout)
    :returns: The final population
    :returns: A class:`~deap.tools.Logbook` with the statistics of the
              evolution.
    The algorithm takes in a population and evolves it in place using the
    :func:`varOr` function. It returns the optimized population and a
    :class:`~deap.tools.Logbook` with the statistics of the evolution. The
    logbook will contain the generation number, the number of evalutions for
    each generation and the statistics if a :class:`~deap.tools.Statistics` is
    given as argument. The *cxpb* and *mutpb* arguments are passed to the
    :func:`varOr` function. The pseudocode goes as follow ::
        evaluate(population)
        for g in range(ngen):
            offspring = varOr(population, toolbox, lambda_, cxpb, mutpb)
            evaluate(offspring)
            population = select(population + offspring, mu)
    First, the individuals having an invalid fitness are evaluated. Second,
    the evolutionary loop begins by producing *lambda_* offspring from the
    population, the offspring are generated by the :func:`varOr` function. The
    offspring are then evaluated and the next generation population is
    selected from both the offspring **and** the population. Finally, when
    *ngen* generations are done, the algorithm returns a tuple with the final
    population and a :class:`~deap.tools.Logbook` of the evolution.
    This function expects :meth:`toolbox.mate`, :meth:`toolbox.mutate`,
    :meth:`toolbox.select` and :meth:`toolbox.evaluate` aliases to be
    registered in the toolbox. This algorithm uses the :func:`varOr`
    variation.
    """
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    # Initialize statistics dict for the individuals in the population, to keep track of mutation/crossover operations and predecessor relations
    for ind in population:
        initialize_stats_dict(ind)

    population[:] = toolbox.evaluate(population)

    record = stats.compile(population) if stats is not None else {}
    logbook.record(gen=0, nevals=len(population), **record)

    # Begin the generational process
    for gen in range(1, ngen + 1):
        # Vary the population
        offspring = varOr(population, toolbox, lambda_, cxpb, mutpb)


        # Update generation statistic for all individuals which have invalid 'generation' stats
        # This hold for individuals that have been altered in the varOr function
        for ind in offspring:
            if ind.statistics['generation'] == 'INVALID':
                ind.statistics['generation'] = gen

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]

        offspring = toolbox.evaluate(offspring)

        # Select the next generation population
        population[:] = toolbox.select(population + offspring, mu)

        # pbar process
        if not pbar.disable:
            # Print only the best individual fitness
            if verbose == 2:
                high_score = max(halloffame.keys[x].wvalues[1] \
                    for x in range(len(halloffame.keys)))
                pbar.write('\nGeneration {0} - Current '
                            'best internal CV score: {1}'.format(gen,
                                                        high_score),

                            file=log_file)

            # Print the entire Pareto front
            elif verbose == 3:
                pbar.write('\nGeneration {} - '
                            'Current Pareto front scores:'.format(gen),
                            file=log_file)
                for pipeline, pipeline_scores in zip(halloffame.items, reversed(halloffame.keys)):
                    pbar.write('\n{}\t{}\t{}'.format(
                            int(pipeline_scores.wvalues[0]),
                            pipeline_scores.wvalues[1],
                            pipeline
                        ),
                        file=log_file
                    )

        # after each population save a periodic pipeline
        if per_generation_function is not None:
            per_generation_function(gen)

        # Update the statistics with the new population
        record = stats.compile(population) if stats is not None else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)

    return population, logbook


def cxOnePoint(ind1, ind2):
    """Randomly select in each individual and exchange each subtree with the
    point as root between each individual.
    :param ind1: First tree participating in the crossover.
    :param ind2: Second tree participating in the crossover.
    :returns: A tuple of two trees.
    """
    # List all available primitive types in each individual
    types1 = defaultdict(list)
    types2 = defaultdict(list)

    for idx, node in enumerate(ind1[1:], 1):
        types1[node.ret].append(idx)
    common_types = []
    for idx, node in enumerate(ind2[1:], 1):
        if node.ret in types1 and node.ret not in types2:
            common_types.append(node.ret)
        types2[node.ret].append(idx)

    if len(common_types) > 0:
        type_ = np.random.choice(common_types)

        index1 = np.random.choice(types1[type_])
        index2 = np.random.choice(types2[type_])

        slice1 = ind1.searchSubtree(index1)
        slice2 = ind2.searchSubtree(index2)
        ind1[slice1], ind2[slice2] = ind2[slice2], ind1[slice1]

    return ind1, ind2


# point mutation function
def mutNodeReplacement(individual, pset):
    """Replaces a randomly chosen primitive from *individual* by a randomly
    chosen primitive no matter if it has the same number of arguments from the :attr:`pset`
    attribute of the individual.
    Parameters
    ----------
    individual: DEAP individual
        A list of pipeline operators and model parameters that can be
        compiled by DEAP into a callable function

    Returns
    -------
    individual: DEAP individual
        Returns the individual with one of point mutation applied to it

    """

    index = np.random.randint(0, len(individual))
    node = individual[index]
    slice_ = individual.searchSubtree(index)

    if node.arity == 0:  # Terminal
        term = np.random.choice(pset.terminals[node.ret])
        if isclass(term):
            term = term()
        individual[index] = term
    else:   # Primitive
        # find next primitive if any
        rindex = None
        if index + 1 < len(individual):
            for i, tmpnode in enumerate(individual[index + 1:], index + 1):
                if isinstance(tmpnode, gp.Primitive) and tmpnode.ret in node.args:
                    rindex = i
                    break

        # pset.primitives[node.ret] can get a list of the type of node
        # for example: if op.root is True then the node.ret is Output_DF object
        # based on the function _setup_pset. Then primitives is the list of classifor or regressor
        primitives = pset.primitives[node.ret]

        if len(primitives) != 0:
            new_node = np.random.choice(primitives)
            new_subtree = [None] * len(new_node.args)
            if rindex:
                rnode = individual[rindex]
                rslice = individual.searchSubtree(rindex)
                # find position for passing return values to next operator
                position = np.random.choice([i for i, a in enumerate(new_node.args) if a == rnode.ret])
            else:
                position = None
            for i, arg_type in enumerate(new_node.args):
                if i != position:
                    term = np.random.choice(pset.terminals[arg_type])
                    if isclass(term):
                        term = term()
                    new_subtree[i] = term
            # paste the subtree to new node
            if rindex:
                new_subtree[position:position + 1] = individual[rslice]
            # combine with primitives
            new_subtree.insert(0, new_node)
            individual[slice_] = new_subtree

    return individual,

#ToDO:看看在哪里读这两个字典更合适
print(os.getcwd())
" 读取light版的operator字典，key为没有前缀的operator名，key_value为长度为3的列表，第一位是operator的类型，第二位是超参数字典，第三位是组合好的字符串形式的sklearn operator列表（所有超参数都有值，包括默认值）"
with open('/Users/yanggu/Documents/博士/科研/流程推荐/实验/机器学习工作流/TPOT-Beta/tpot/surrogate_models/related_data/dict_operator_light.pkl', 'rb') as f:
    dict_operator_light = pickle.load(f)
print(len(dict_operator_light))
print(dict_operator_light['Binarizer'])

" 读取带超参数的operator预嵌入字典，key为带参数的operator字符串，key_value为其对应的numpy向量 "
with open('/Users/yanggu/Documents/博士/科研/流程推荐/实验/机器学习工作流/TPOT-Beta/tpot/surrogate_models/related_data/dict_operator_light_PreEmbedding.pkl', 'rb') as f:
    dict_operator_light_PreEmbedding = pickle.load(f)
print("共有多少种不同的operator超参数组合：", len(dict_operator_light_PreEmbedding))
print(dict_operator_light_PreEmbedding['Binarizer(threshold=0.30000000000000004)'].shape)

def ExtractOpEmbed(pipe):  #从sklearn pipeline里提取出三个operator的字符串，再转成嵌入表示
    # 遍历 Pipeline 的每个步骤
    op_str_list = []
    for step_name, step_obj in pipe.named_steps.items():
        # 获取当前步骤的operator名称
        step_type_str = type(step_obj).__name__
        # 获取当前步骤的operator参数字典
        step_params_dict = step_obj.get_params()
        light_params_dict = dict_operator_light[step_type_str][1]  # light字典里该operator的参数字典

        if len(light_params_dict) == 0:
            op_str_list.append(step_type_str + "()")
        else:
            op_str = step_type_str + "("
            temp_list = sorted(light_params_dict.keys())
            # print(temp_list)
            for hparam in temp_list[0:-1]:
                if hparam == 'score_func':  # 对SelectFwe和SelectPercentile的超参数'score_func'特殊处理
                    value = 'f_classif'
                else:
                    value = step_params_dict[hparam]
                value = "'" + value + "'" if isinstance(value, str) else str(value)  # 为了使本身为字符串的超参数值仍然带引号
                temp_str = hparam + '=' + value + ', '
                op_str += temp_str
            if temp_list[-1] == 'score_func':
                value = 'f_classif'
            else:
                value = step_params_dict[temp_list[-1]]
            value = "'" + value + "'" if isinstance(value, str) else str(value)
            op_str = op_str + temp_list[-1] + '=' + value + ")"
            op_str_list.append(op_str)
        # print(op_str_list)
    op1_str, op2_str, op3_str = op_str_list[0], op_str_list[1], op_str_list[2]
    op1_embed, op2_embed, op3_embed = dict_operator_light_PreEmbedding[op1_str], dict_operator_light_PreEmbedding[op2_str], dict_operator_light_PreEmbedding[op3_str]

    return op1_embed, op2_embed, op3_embed

@threading_timeoutable(default="Timeout")
def score_by_surrogate(sklearn_pipeline, dataset_embed, suModel, use_dask=False):

    if use_dask: #ToDO:后续完善use_dask的情况
        import dask  # noqa
    else:
        try:
            "dataset嵌入"
            da_embed = dataset_embed[None, ...]  # 很重要！增加一个维度，因为SurrogateNet的数据集输入有两个维度，第一维是batch-size。这里只有一个数据但也要加一个维度
            da_embed = torch.from_numpy(da_embed).float()  # numpy转torch
            "operators嵌入"
            op1_embed, op2_embed, op3_embed = ExtractOpEmbed(sklearn_pipeline)  # 从sklearn pipeline里提取出三个operator的字符串，再转成嵌入表示
            ops_embed = np.vstack([op1_embed, op2_embed, op3_embed])  # 把三个operator向量(768,)拼成(3,768)的数组
            ops_embed = ops_embed[None, ...]  # 同样增加一个维度
            ops_embed = torch.from_numpy(ops_embed).float()  # numpy转torch
            "代理网络预测"
            out = suModel(da_embed, ops_embed)
            out = out.detach().numpy().tolist()  # torch转numpy再转list，变成[[0.904032289981842]]
            predicted_cv_score = out[0][0]  # 取数值
            if not (0 <= predicted_cv_score <= 1):
                raise ValueError("Incorrect output format from Surrogate's prediction!")

            return predicted_cv_score
        except TimeoutException:
            return "Timeout"
        except Exception as e:
            return -float('inf')

@threading_timeoutable(default="Timeout")
def _wrapped_cross_val_score(sklearn_pipeline, features, target,
                             cv, scoring_function, sample_weight=None,
                             groups=None, use_dask=False):
    """Fit estimator and compute scores for a given dataset split.

    Parameters
    ----------
    sklearn_pipeline : pipeline object implementing 'fit'
        The object to use to fit the data.
    features : array-like of shape at least 2D
        The data to fit.
    target : array-like, optional, default: None
        The target variable to try to predict in the case of
        supervised learning.
    cv: cross-validation generator
        Object to be used as a cross-validation generator.
    scoring_function : callable
        A scorer callable object / function with signature
        ``scorer(estimator, X, y)``.
    sample_weight : array-like, optional
        List of sample weights to balance (or un-balanace) the dataset target as needed
    groups: array-like {n_samples, }, optional
        Group labels for the samples used while splitting the dataset into train/test set
    use_dask : bool, default False
        Whether to use dask
    """
    sample_weight_dict = set_sample_weight(sklearn_pipeline.steps, sample_weight)

    features, target, groups = indexable(features, target, groups)

    cv_iter = list(cv.split(features, target, groups))
    scorer = check_scoring(sklearn_pipeline, scoring=scoring_function)

    if use_dask:
        try:
            import dask_ml.model_selection  # noqa
            import dask  # noqa
            from dask.delayed import Delayed
        except Exception as e:
            msg = "'use_dask' requires the optional dask and dask-ml depedencies.\n{}".format(e)
            raise ImportError(msg)

        dsk, keys, n_splits = dask_ml.model_selection._search.build_graph(
            estimator=sklearn_pipeline,
            cv=cv,
            scorer=scorer,
            candidate_params=[{}],
            X=features,
            y=target,
            groups=groups,
            fit_params=sample_weight_dict,
            refit=False,
            error_score=float('-inf'),
        )

        cv_results = Delayed(keys[0], dsk)
        scores = [cv_results['split{}_test_score'.format(i)]
                  for i in range(n_splits)]
        CV_score = dask.delayed(np.array)(scores)[:, 0]
        return dask.delayed(np.nanmean)(CV_score)
    else:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                scores = [_fit_and_score(estimator=clone(sklearn_pipeline),
                                         X=features,
                                         y=target,
                                         scorer=scorer,
                                         train=train,
                                         test=test,
                                         verbose=0,
                                         parameters=None,
                                         error_score='raise',
                                         fit_params=sample_weight_dict)
                                    for train, test in cv_iter]
                if isinstance(scores[0], list): #scikit-learn <= 0.23.2
                    CV_score = np.array(scores)[:, 0]
                elif isinstance(scores[0], dict): # scikit-learn >= 0.24
                    from sklearn.model_selection._validation import _aggregate_score_dicts
                    CV_score = _aggregate_score_dicts(scores)["test_scores"]
                else:
                    raise ValueError("Incorrect output format from _fit_and_score!")
                CV_score_mean = np.nanmean(CV_score)
            return CV_score_mean
        except TimeoutException:
            return "Timeout"
        except Exception as e:
            return -float('inf')
