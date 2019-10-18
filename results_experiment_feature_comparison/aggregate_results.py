import os
import json
import pandas as pd
import numpy as np
import copy
from collections import defaultdict


def clustering_scores(filename):
    """Returns Adj Rand and Adj Mutual Info scores for each family and dataset aggregated by family.
        
    The main goal of this data structure is to facilitate the calculation of average performance
    accross the 6 families defined in families.json.

    Args:
        filename (str): xls clustering_stats file.
    Returns:
        {
            'Animals': {
                'mfcc': [<Adj Rnd score dataset 1>, <Adj Rnd score dataset 2>, ...]
            },
            ...
        }
    """
    families = json.load(open('families.json', 'rb'))
    features_names_idx_adj_rnd = [
        ('mfcc', 2), 
        ('audioset', 8), 
        ('openl3-env', 14), 
        ('openl3-music', 20), 
        ('soundnet', 26)
    ]
    features_names_idx_adj_mutual = [
        ('mfcc', 1), 
        ('audioset', 7), 
        ('openl3-env', 13), 
        ('openl3-music', 19), 
        ('soundnet', 25)
    ]
    
    df = pd.read_excel(filename)
    data = np.array(df.iloc[1:46,:])
    data_dict = {category_name:list(v) for category_name, v in zip(data[:,0], data[:,1:])}
    dataset_stats_ari = {}
    dataset_stats_ami = {}

    # ARI
    for family_name, categories_names in families.items():
        dataset_stats_ari[family_name] = defaultdict(list)
        for feature_name, idx_adj_rnd in features_names_idx_adj_rnd:
            for category_name in categories_names:
                dataset_stats_ari[family_name][feature_name].append(data_dict[category_name][idx_adj_rnd])

    # AMI
    for family_name, categories_names in families.items():
        dataset_stats_ami[family_name] = defaultdict(list)
        for feature_name, idx_adj_mutual in features_names_idx_adj_mutual:
            for category_name in categories_names:
                dataset_stats_ami[family_name][feature_name].append(data_dict[category_name][idx_adj_mutual])

    return dataset_stats_ari, dataset_stats_ami


def average_scores(dataset_stats):
    """Computes average (mean) of given score for each dataset family.
    Adds the mean scores to the given dict.
    """
    dataset_stats_p = copy.deepcopy(dataset_stats)
    for family_name, family_stats in dataset_stats.items():
        for feature_name, scores in family_stats.items():
            dataset_stats_p[family_name]['{}_mean'.format(feature_name)] = np.mean(scores)
        
    return dataset_stats_p


if __name__ == "__main__":
    dataset_stats_ari, dataset_stats_ami = clustering_scores('knn/clustering_stats.xlsx')
    dataset_stats_ari = average_scores(dataset_stats_ari)
    dataset_stats_ami = average_scores(dataset_stats_ami)
    json.dump(dataset_stats_ari, open('knn_average_family_adj_rnd.json', 'w'))
    json.dump(dataset_stats_ami, open('knn_average_family_adj_multual_info.json', 'w'))

    dataset_stats_ari, dataset_stats_ami = clustering_scores('kmeans/clustering_stats.xlsx')
    dataset_stats_ari = average_scores(dataset_stats_ari)
    dataset_stats_ami = average_scores(dataset_stats_ami)
    json.dump(dataset_stats_ari, open('kmeans_average_family_adj_rnd.json', 'w'))
    json.dump(dataset_stats_ami, open('kmeans_average_family_adj_multual_info.json', 'w'))
    
