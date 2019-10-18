import xlsxwriter

from settings import EMBEDDING_FOLDERS


def return_dataset_stats(dataset):
    datase_name = dataset['name']
    num_sounds = len(dataset['sound_ids'])
    num_labels = len(dataset['dataset'].keys())
    num_sounds_per_labels = ', '.join([str(len(v)) for _, v in dataset['dataset'].items()])
    return [datase_name, num_sounds, num_labels, num_sounds_per_labels]


def write_stats_datasets_xls(datasets, file_name='datasets_stats.xlsx'):
    header = ['Datasets', '# sounds', '# labels', '# sounds/labels']
    data = [header]
    for dataset in datasets:
        data.append(return_dataset_stats(dataset))
    write_2d_array_to_xls(data, file_name)


def write_stats_clusterings_xls(datasets, file_name='clustering_stats.xlsx'):
    header_1 = ['']
    header_2 = ['']
    stats_data = []
    for idx, dataset in enumerate(datasets):
        dataset_name = dataset['name']
        data_row = [dataset_name]
        for embedding_name, _ in EMBEDDING_FOLDERS.items():
            evaluation_metrics = dataset['evaluation_metrics_{}'.format(embedding_name)]
            purity = evaluation_metrics['purity']
            adj_mi = evaluation_metrics['adjusted_mutual_info']
            adj_rand = evaluation_metrics['adjusted_rand']
            avg_clust_cohesiveness = evaluation_metrics['average_cluster_cohesiveness']
            avg_semantic_cohesiveness = evaluation_metrics['average_semantic_cohesiveness']
            num_clusters = max(dataset['labels_{}'.format(embedding_name)]) + 1

            data_row += [purity, adj_mi, adj_rand, avg_clust_cohesiveness, avg_semantic_cohesiveness, num_clusters]

            if idx==0:  # add this headers only once for each feature
                header_1 += [embedding_name, '', '', '', '', '']
                header_2 += ['Purity', 'Adj. MI', 'Adj, Rand', 'Avg Clust Cohesiveness', 'Avg Semantic Cohesiveness', '# clusters']
        stats_data.append(data_row)   
    data = [header_1, header_2] + stats_data
    write_2d_array_to_xls(data, file_name)


def write_2d_array_to_xls(data, file_name):
    workbook = xlsxwriter.Workbook(file_name, {'strings_to_numbers': True})
    worksheet = workbook.add_worksheet()
    for row_idx, row in enumerate(data):
        for col_idx, cell in enumerate(row):
            worksheet.write(row_idx, col_idx, cell)
    workbook.close()