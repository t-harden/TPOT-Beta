#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 15:23:58 2023

@author: yuan
"""

import numpy as np

# 用于合并不同数据类型的config
'''
这里传入的第一个参数attr1是原有的config中的参数，而attr2是自己生成的
为了防止参数类型改变造成后续程序读取数据等方面出现问题，这里返回的参数类型与attr1保持一致

经前期统计，TPOT原始的operator参数一共有4种：dict;np.ndarray;list;range
与此同时，一些字典的最后一级的值会是None
'''


def _get_union(attr1, attr2):
    # 如果两个属性都是列表，则取并集即可
    if isinstance(attr1, list) and isinstance(attr2, list):
        result = list(set(attr1) | set(attr2))

    #     # 如果一个是列表，一个是np.ndarray，则将np.ndarray转换为列表，然后取并集
    #     # 但是考虑到我们自定义生成operators的方式，这样的情况应该是不存在的，先注释掉
    # elif isinstance(attr1, list) and isinstance(attr2, np.ndarray):
    #     result = list(set(attr1) | set(attr2.tolist()))

    # 如果一个是np.ndarray，一个是列表，则将np.ndarray转换为列表，然后取并集
    elif isinstance(attr1, np.ndarray) and isinstance(attr2, list):
        result = list(set(attr1.tolist()) | set(attr2))
        # 再把形式转换成attr1的，即ndarray类型
        result = np.array(result)

    elif isinstance(attr1, range) and isinstance(attr2, list):
        result = list(set(list(attr1)) | set(attr2))
        # 将list类型转换成range类型再返回
        min_value = min(result)
        max_value = max(result)
        result = range(min_value, max_value + 1)

    # 合并字典（我直接默认attr1和attr2的字典结构完全相同了），使用递归方法
    elif isinstance(attr1, dict) and isinstance(attr2, dict):
        result = {}
        for key in attr1.keys():
            # if isinstance(attr1[key],dict):
            result[key] = _get_union(attr1[key], attr2[key])


    # 虽说统计得到的类型中没有None，但是递归时可能遇到这种情况
    elif attr1 is None:
        result = attr2
    elif attr2 is None:
        result = attr1

    # 两个都是np.ndarray，考虑到我们自定义生成operators的方式，这样的情况应该是不存在的，先注释掉
    # 大哥我错了，不应该注释掉你的，合并字典会用上
    elif isinstance(attr1, np.ndarray) and isinstance(attr2, np.ndarray):
        result = np.unique(np.concatenate((attr1, attr2)))
        result = np.array(result)

    #同样的，合并字典会用上
    elif isinstance(attr1,range) and isinstance(attr2,range):
        result = list(set(list(attr1)) | set(list(attr2)))
        min_value = min(result)
        max_value = max(result)
        result = range(min_value, max_value + 1)

    return result


# 做一些测试
if __name__ == "__main__":
    # 字典测试1
    a1 = {
        'step': np.arange(0.10, 1.11, 0.05),
        'estimator': {
            'sklearn.ensemble.ExtraTreesClassifier': {
                'n_estimators': [100, 300],
                'criterion': ['gini', 'entropy'],
                'max_features': np.arange(0.05, 1.01, 0.05)
            }
        }
    }

    a2 = {
        'step': np.arange(0.05, 1.01, 0.05),
        'estimator': {
            'sklearn.ensemble.ExtraTreesClassifier': {
                'n_estimators': [100, 200],
                'criterion': ['gini', 'entropy'],
                'max_features': np.arange(0.05, 1.01, 0.05)
            }
        }
    }

    print(_get_union(a1, a2))

    print('赢麻了')
    # 赢！

    a3 = {
        'percentile': range(1, 110),
        'score_func': {
            'sklearn.feature_selection.f_regression': None
        }
    }

    a4 = {
        'percentile': range(1, 100),
        'score_func': {
            'sklearn.feature_selection.f_regression': [2]
        }
    }

    print(_get_union(a3, a4))

    print('大赢')