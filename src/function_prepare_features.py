# -*- coding: utf-8 -*-
# @Time    : 2020/8/29 0:13
# @Author  : scfan
# @FileName: function_prepare_features.py
# @Software: PyCharm

import os
import time
from subprocess import  call
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
import seaborn as sns
# import matplotlib.pyplot as plt
import scipy as sc
# from matplotlib.colors import ListedColormap
import scipy.sparse as sparse
import importlib
import copy
import gc
import datetime
from scipy.stats import zscore
import scipy.sparse as sparse
from scipy.sparse import coo_matrix
from joblib import Parallel, delayed

def get_CDD_vector( cell_name_list, acp_df, all_cdd_file):
    print("Start to get CDD")

    cis_contacts = acp_df[acp_df['chr1'] == acp_df['chr2']]
    band_paras = [8]
    band_thrs = [int(np.log2(21) * band_para) for band_para in band_paras]

    dis_vector_org = (((cis_contacts['pos1'] - cis_contacts['pos2']) / 1000).abs() + 1).apply(np.log2, 1)

    flag = 0
    dis_vector = pd.Series(dis_vector_org * band_paras[flag]).apply(int)

    temp_2 = pd.DataFrame(dis_vector, columns=['distance'])
    new_cis_contacts = pd.concat([cis_contacts, temp_2], axis=1)

    all_contacts = acp_df.groupby(['cell_name']).size()
    all_distance_count = new_cis_contacts.groupby(['cell_name', 'distance']).size()

    all_cdd = pd.DataFrame()

    for cell_name in cell_name_list:
        distri_vector = all_distance_count.loc[cell_name]

        all_cons = all_contacts[cell_name]
        distri_vector = distri_vector[distri_vector.index >= band_thrs[flag]]
        one_cdd_series_1 = distri_vector / all_cons * 100

        one_cdd_series_1_rename = one_cdd_series_1.rename(cell_name)
        one_cdd_series_1_pd = pd.DataFrame(one_cdd_series_1_rename)
        all_cdd = pd.concat([all_cdd, one_cdd_series_1_pd], axis=1)
    all_cdd = all_cdd.T
    all_cdd.index = pd.Series(all_cdd.index, name='cell_nm')
    all_cdd.columns = all_cdd.columns.rename('cell_nm')
    all_cdd.columns = pd.Series(all_cdd.columns, name='').apply(lambda x: "8+" + str(x))

    all_cdd.to_csv(all_cdd_file, sep='\t',header=True,index=True)
    return all_cdd


def get_PCC_vector_de_novo_MultiProcess_create_intermediate_file_sepcific_threshold_norm(sim_info,test_data_df_dic,cell_name_list,all_pcc_file,pcc_intermediate_file_dir,threshold,one_or_sum,kernel_number,max_value=50):
    '''
    norm: normalize the contacts by all_contacts in one cell
    :param sim_info:
    :param test_data_df_dic: make sure the index of test_data_df_dic has been reset
    :return:
    '''
    kernel_number = int(kernel_number)
    print("start to get PCC vector de novo")
    bin_dic = {'100kb': 100000, '500kb': 500000, '1Mb': 1000000}
    print("get bin")
    all_start=datetime.datetime.now()
    for bin_name, resolution in bin_dic.items():
        print(bin_name, resolution)
        for i, enum in enumerate(test_data_df_dic.items()):
            print(i, enum[0])
            one_df = enum[1]
            one_df[bin_name + '1'] = one_df['pos1'] // resolution
            one_df[bin_name + '2'] = one_df['pos2'] // resolution
            test_data_df_dic[enum[0]] = one_df
    all_cell_total_contacts_list=[]
    for cell_name in cell_name_list:
        all_cell_total_contacts_list.append(test_data_df_dic[cell_name]['count'].sum())
    all_cell_total_contacts_df=pd.DataFrame({'total_contacts':all_cell_total_contacts_list},index=cell_name_list)
    test_chr_data_df_dic = {}
    test_chr_combine_df_dic = {}
    print("start to get chr_combine_df_dic")
    pairs_sum_std_df_chr_bin_dic_file_path = os.path.join(pcc_intermediate_file_dir, 'pairs_sum_std_df_chr_bin_dic.pkl')
    start=datetime.datetime.now()

    if(not os.path.exists(pairs_sum_std_df_chr_bin_dic_file_path)):
        for chr_name in sim_info.chr_length_dic:
            if (chr_name == 'chrY'):
                continue
            print(chr_name)
            temp_chr_dic = {}
            temp_chr_list = []
            for cell_name, one_df in test_data_df_dic.items():
                chr_cis_df = one_df[(one_df['chr1'] == chr_name) & (one_df['chr2'] == chr_name)]
                temp_chr_dic[cell_name] = chr_cis_df
                temp_chr_list.append(chr_cis_df)
            test_chr_combine_df_dic[chr_name] = pd.concat(temp_chr_list, axis=0, sort=False)
            test_chr_data_df_dic[chr_name] = temp_chr_dic
    end=datetime.datetime.now()
    print("end, using ",end-start)
    pool_chr_bin_matrix = {}
    pool_chr_bin_matrix_file_path=os.path.join(pcc_intermediate_file_dir,'pool_chr_bin_matrix.pkl')

    print("start to get pool_chr_bin_matrix")
    start = datetime.datetime.now()
    if(not os.path.exists(pool_chr_bin_matrix_file_path)):
        for chr_name, chr_length in sim_info.chr_length_dic.items():
            if (chr_name == 'chrY'):
                continue
            print(chr_name)
            temp_chr_combine_df = test_chr_combine_df_dic[chr_name]
            temp_bin_matrix_dic = {}
            for bin_name, resolution in bin_dic.items():
                print(bin_name, resolution)

                temp_chr_combine_df = temp_chr_combine_df.reset_index(drop=True)
                row = temp_chr_combine_df[[bin_name + '1', bin_name + '2']].min(axis=1).values
                col = temp_chr_combine_df[[bin_name + '1', bin_name + '2']].max(axis=1).values
                count = temp_chr_combine_df['count'].values
                temp_bin_length = chr_length // resolution + 1
                coo_CertainChr_CertainBin_combine = coo_matrix((count, (row, col)),
                                                               shape=(int(temp_bin_length), int(temp_bin_length)),
                                                               dtype=np.int)
                upper_matrix = coo_CertainChr_CertainBin_combine.todense()
                lower_matrix = np.mat(np.tril(upper_matrix.T, -1))
                full_matrix = upper_matrix + lower_matrix
                temp_bin_matrix_dic[bin_name] = full_matrix

            pool_chr_bin_matrix[chr_name] = temp_bin_matrix_dic
        sim_info.store_variable_from_pikle_file(pool_chr_bin_matrix_file_path,pool_chr_bin_matrix)

    end = datetime.datetime.now()
    print("end, using ",end - start)

    pairs_sum_std_df_chr_bin_dic = {}
    pairs_sum_std_df_chr_bin_dic_file_path=os.path.join(pcc_intermediate_file_dir,'pairs_sum_std_df_chr_bin_dic.pkl')
    print("start to get pairs_sum_std_df_chr_bin_dic")
    start = datetime.datetime.now()
    if(not os.path.exists(pairs_sum_std_df_chr_bin_dic_file_path)):
        def design_for_parallel_test_chr_combine_df_dic(chr_name, temp_chr_combine_df):
            chr_length=sim_info.chr_length_dic[chr_name]
            temp_chr_combine_df = temp_chr_combine_df.reset_index(drop=True)
            temp_pairs_sum_std_df_bin_dic = {}
            for bin_name, resolution in bin_dic.items():
                pairs_sum_std = pd.DataFrame()
                min_temp=temp_chr_combine_df[[bin_name + '1', bin_name + '2']].min(axis=1)
                max_temp=temp_chr_combine_df[[bin_name + '1', bin_name + '2']].max(axis=1)
                temp_chr_combine_df[bin_name + '1'] = min_temp
                temp_chr_combine_df[bin_name + '2'] = max_temp
                bin_pair_with_name_df = \
                    temp_chr_combine_df.groupby(['cell_name', bin_name + '1', bin_name + '2'], as_index=False)[
                        'count'].sum()
                print(chr_name, bin_name, resolution)
                chr_bin_length = chr_length // resolution + 1
                print('chr_bin_length', chr_bin_length)
                all_bin_pair_df = bin_pair_with_name_df.groupby([bin_name + '1', bin_name + '2'])['count'].sum()
                # all_bin_pair_more_than_threshold_series = all_bin_pair_df.loc[all_bin_pair_df.values >= 50]
                all_bin_pair_more_than_threshold_series = all_bin_pair_df.loc[all_bin_pair_df.values >= max_value]
                bin_pair_with_name_df_with_index = bin_pair_with_name_df.set_index([bin_name + '1', bin_name + '2'])

                for bin_tuple, bin_sum in all_bin_pair_more_than_threshold_series.items():
                    bin1 = bin_tuple[0]
                    bin2 = bin_tuple[1]

                    print(bin_name, chr_name, bin1, bin2, "start to get all_sc_pair_contacts")
                    start = datetime.datetime.now()

                    all_sc_pair_contacts = \
                        bin_pair_with_name_df_with_index.loc[bin1, bin2][['cell_name', 'count']].set_index('cell_name',
                                                                                                           drop=True)[
                            'count']

                    end = datetime.datetime.now()
                    print("end,using", end - start)
                    all_sc_pair_contacts = all_sc_pair_contacts * 100000 / all_cell_total_contacts_df['total_contacts']
                    all_sc_pair_contacts=all_sc_pair_contacts.fillna(0)

                    pairs_sum = all_sc_pair_contacts.sum()
                    pairs_std = all_sc_pair_contacts.values.std()
                    temp_pair_sum_std = pd.Series([pairs_sum, pairs_std], index=['sum', 'std'])
                    temp_pair_sum_std.loc['values'] = all_sc_pair_contacts
                    pairs_sum_std[chr_name + '+' + str(bin1) + '+' + str(bin2)] = temp_pair_sum_std
                temp_pairs_sum_std_df_bin_dic[bin_name] = pairs_sum_std
            return temp_pairs_sum_std_df_bin_dic, chr_name
        result_list = Parallel(n_jobs=kernel_number)(
            delayed(design_for_parallel_test_chr_combine_df_dic)(chr_name, temp_chr_combine_df) for
            chr_name, temp_chr_combine_df in test_chr_combine_df_dic.items())

        for temp_pairs_sum_std_df_bin_dic_And_chr_name in result_list:
            pairs_sum_std_df_chr_bin_dic[temp_pairs_sum_std_df_bin_dic_And_chr_name[1]] = \
            temp_pairs_sum_std_df_bin_dic_And_chr_name[0]
        # for chr_name, chr_length in sim_info.chr_length_dic.items():
        #     if (chr_name == 'chrY'):
        #         continue
        #     temp_chr_combine_df = test_chr_combine_df_dic[chr_name]
        #     temp_chr_combine_df = temp_chr_combine_df.reset_index(drop=True)
        #     temp_pairs_sum_std_df_bin_dic = {}
        #     for bin_name, resolution in bin_dic.items():
        #         pairs_sum_std = pd.DataFrame()
        #         temp_chr_combine_df[bin_name + '1'] = temp_chr_combine_df[[bin_name + '1', bin_name + '2']].min(axis=1)
        #         temp_chr_combine_df[bin_name + '2'] = temp_chr_combine_df[[bin_name + '1', bin_name + '2']].max(axis=1)
        #         bin_pair_with_name_df = \
        #         temp_chr_combine_df.groupby(['cell_name', bin_name + '1', bin_name + '2'], as_index=False)['count'].sum()
        #         print(chr_name, bin_name, resolution)
        #         chr_bin_length = chr_length // resolution + 1
        #         print('chr_bin_length', chr_bin_length)
        #         all_bin_pair_df = bin_pair_with_name_df.groupby([bin_name + '1', bin_name + '2'])['count'].sum()
        #         all_bin_pair_more_than_threshold_series = all_bin_pair_df.loc[all_bin_pair_df.values >= 50]
        #
        #         bin_pair_with_name_df_with_index = bin_pair_with_name_df.set_index([bin_name + '1', bin_name + '2'])
        #
        #         # def get_temp_pair_sum_std(bin_pair_with_name_df_with_index, bin_tuple, bin_sum):
        #         #     bin1 = bin_tuple[0]
        #         #     bin2 = bin_tuple[1]
        #         #
        #         #     all_sc_pair_contacts = \
        #         #     bin_pair_with_name_df_with_index.loc[bin1, bin2][['cell_name', 'count']].set_index('cell_name',
        #         #                                                                                        drop=True)['count']
        #         #
        #         #     pairs_sum = bin_sum
        #         #     pairs_std = all_sc_pair_contacts.values.std()
        #         #     temp_pair_sum_std = pd.Series([pairs_sum, pairs_std], index=['sum', 'std'])
        #         #     temp_pair_sum_std.loc['values'] = all_sc_pair_contacts
        #         #     temp_pair_sum_std.name = chr_name + '+' + str(bin1) + '+' + str(bin2)
        #         #     return temp_pair_sum_std
        #
        #
        #         # result_list = Parallel(n_jobs=24)(
        #         #     delayed(get_temp_pair_sum_std)(bin_pair_with_name_df_with_index, bin_tuple, bin_sum) for
        #         #     bin_tuple, bin_sum in all_bin_pair_more_than_threshold_series.items())
        #         # for temp_pair_sum_std in result_list:
        #         #     pairs_sum_std[temp_pair_sum_std.name] = temp_pair_sum_std
        #
        #         for bin_tuple, bin_sum in all_bin_pair_more_than_threshold_series.items():
        #             # print(bin_name,bin_tuple,bin_sum)
        #             bin1 = bin_tuple[0]
        #             bin2 = bin_tuple[1]
        #
        #             print(bin_name,chr_name,bin1,bin2,"start to get all_sc_pair_contacts")
        #             start=datetime.datetime.now()
        #
        #             all_sc_pair_contacts = \
        #             bin_pair_with_name_df_with_index.loc[bin1, bin2][['cell_name', 'count']].set_index('cell_name',
        #                                                                                                drop=True)['count']
        #
        #             end=start=datetime.datetime.now()
        #             print("end,using",end-start)
        #
        #             pairs_sum = bin_sum
        #             pairs_std = all_sc_pair_contacts.values.std()
        #             temp_pair_sum_std = pd.Series([pairs_sum, pairs_std], index=['sum', 'std'])
        #             temp_pair_sum_std.loc['values'] = all_sc_pair_contacts
        #             pairs_sum_std[chr_name + '+' + str(bin1) + '+' + str(bin2)] = temp_pair_sum_std
        #         temp_pairs_sum_std_df_bin_dic[bin_name] = pairs_sum_std
        #     pairs_sum_std_df_chr_bin_dic[chr_name] = temp_pairs_sum_std_df_bin_dic
        sim_info.store_variable_from_pikle_file(pairs_sum_std_df_chr_bin_dic_file_path, pairs_sum_std_df_chr_bin_dic)

    end = datetime.datetime.now()
    print("end, using ",end - start)

    genome_exp_vectors_chr_bin = {}
    genome_exp_vectors_chr_bin_file_path=os.path.join(pcc_intermediate_file_dir,'genome_exp_vectors_chr_bin.pkl')
    print("start to get genome_exp_vectors_chr_bin")
    start = datetime.datetime.now()
    if(not os.path.exists(genome_exp_vectors_chr_bin_file_path)):
        for chr_name in sim_info.chr_length_dic:
            if (chr_name == 'chrY'):
                continue
            temp_genome_exp_vectors_bin = {}
            for bin_name, resolution in bin_dic.items():
                print(chr_name, bin_name, resolution)

                one_matrix = pool_chr_bin_matrix[chr_name][bin_name]

                bin_size = sim_info.chr_length_dic[chr_name] // resolution + 1

                chr_pool_map = one_matrix
                sparse_Mat = sparse.coo_matrix(chr_pool_map)
                sparse_tril_Mat = sparse.tril(sparse_Mat)
                triplets = np.column_stack((sparse_tril_Mat.col, sparse_tril_Mat.row, sparse_tril_Mat.data))
                triplets_DF = pd.DataFrame(triplets, columns=['bin1', 'bin2', 'values'])
                triplets_DF = triplets_DF.fillna(0)
                exp_vectors = np.zeros(bin_size)
                for k in range(bin_size):
                    all_counts = triplets_DF.loc[(triplets_DF['bin2'] - triplets_DF['bin1'] == k)]['values']
                    temp_sum = np.sum(all_counts)
                    if not np.isnan(temp_sum):
                        exp_vectors[k] = temp_sum / (bin_size - k)
                temp_genome_exp_vectors_bin[bin_name] = exp_vectors
            genome_exp_vectors_chr_bin[chr_name] = temp_genome_exp_vectors_bin
        sim_info.store_variable_from_pikle_file(genome_exp_vectors_chr_bin_file_path, genome_exp_vectors_chr_bin)
    end = datetime.datetime.now()
    print("end, using ", end - start)


    print("create PCC de novo intermediate files using ",end-all_start)
    return 0

def get_PCC_vector_de_novo_load_intermediate_file_sepcific_threshold_MultiProcess(sim_info,cell_name_list,all_pcc_file,pcc_intermediate_file_dir,threshold,one_or_sum,kernel_number):
    '''
    one_or_sum: 'one' or 'sum' in exp_vector
    :param sim_info:
    :param test_data_df_dic: make sure the index of test_data_df_dic has been reset
    :return:
    '''
    # print("start to get PCC vector de novo, loading file")
    kernel_number=int(kernel_number)
    bin_dic = {'100kb': 100000, '500kb': 500000, '1Mb': 1000000}


    print("start to load chr_combine_df_dic")


    pool_chr_bin_matrix_file_path=os.path.join(pcc_intermediate_file_dir,'pool_chr_bin_matrix.pkl')

    all_start=datetime.datetime.now()
    print("start to load pool_chr_bin_matrix")
    start = datetime.datetime.now()
    if(os.path.exists(pool_chr_bin_matrix_file_path)):
        pool_chr_bin_matrix=sim_info.load_variable_from_pikle_file(pool_chr_bin_matrix_file_path)
    else:
        print(pool_chr_bin_matrix_file_path,"not exist")
        exit(-1)
    end = datetime.datetime.now()
    print("end, using ",end - start)


    pairs_sum_std_df_chr_bin_dic = {}
    pairs_sum_std_df_chr_bin_dic_file_path=os.path.join(pcc_intermediate_file_dir,'pairs_sum_std_df_chr_bin_dic.pkl')
    print("start to get pairs_sum_std_df_chr_bin_dic",pairs_sum_std_df_chr_bin_dic_file_path)
    start = datetime.datetime.now()
    if(os.path.exists(pairs_sum_std_df_chr_bin_dic_file_path)):
        pairs_sum_std_df_chr_bin_dic=sim_info.load_variable_from_pikle_file(pairs_sum_std_df_chr_bin_dic_file_path)
    else:
        print(pairs_sum_std_df_chr_bin_dic_file_path,"not exist")
        exit(-1)
    end = datetime.datetime.now()
    print("end, using ",end - start)

    genome_exp_vectors_chr_bin = {}
    genome_exp_vectors_chr_bin_file_path=os.path.join(pcc_intermediate_file_dir,'genome_exp_vectors_chr_bin.pkl')
    print("start to get genome_exp_vectors_chr_bin")
    start = datetime.datetime.now()
    if(os.path.exists(genome_exp_vectors_chr_bin_file_path)):
        genome_exp_vectors_chr_bin = sim_info.load_variable_from_pikle_file(genome_exp_vectors_chr_bin_file_path)
    else:
        print(genome_exp_vectors_chr_bin_file_path,"not exist")
        exit(-1)
    end = datetime.datetime.now()
    print("end, using ", end - start)

    print("start to get exp_vectors_bin")
    start = datetime.datetime.now()
    exp_vectors_bin = {}
    for bin_name, resolution in bin_dic.items():
        print(bin_name, resolution)

        chr_length_bin_dic = sim_info.get_chr_length_bin_without_Y(resolution)
        chr_bin_size_list = []
        for chr_name, chr_bin in chr_length_bin_dic.items():
            chr_bin_size_list.append(chr_bin)

        exp_len = np.max(np.array(chr_bin_size_list))
        exp_vectors = np.zeros(int(exp_len))
        for k in range(int(exp_len)):
            all_contact_counts = 0
            n_counts = 0
            for chr_name, chr_bin in chr_length_bin_dic.items():
                if chr_bin > k:
                    all_contact_counts += genome_exp_vectors_chr_bin[chr_name][bin_name][k] * (chr_bin - k)
                    n_counts += (chr_bin - k)
            if n_counts != 0:
                exp_vectors[k] = all_contact_counts / n_counts
        exp_vectors_bin[bin_name] = exp_vectors

    end = datetime.datetime.now()
    print("end, using ", end - start)
    print("start to pairs_features_chr_bin_dic")
    start= datetime.datetime.now()
    pairs_features_chr_bin_dic = {}

    def design_for_parallel_temp_pairs_features_bin_dic(chr_name, temp_pairs_sum_std_df_bin_dic):
        #exp : expection values
        temp_pairs_features_bin_dic = {}
        for bin_name, pairs_sum_std in temp_pairs_sum_std_df_bin_dic.items():
            print(chr_name, bin_name)
            pairs_sum_std_T = pairs_sum_std.T
            pairs_X = pairs_sum_std_T['sum']

            exp_pairs_X = pd.Series()
            for index in pairs_X.index:
                temp = index.split("+")
                location_x = int(temp[2]) - int(temp[1])
                if(location_x==0):
                    exp_pairs_X[index]=0
                    continue
                else:
                    exp_pairs_X[index] = 1
                    continue
                EV = exp_vectors_bin[bin_name][location_x]

                if EV != 0:
                    if (one_or_sum == 'one'):
                        exp_pairs_X[index] = 1 / exp_vectors_bin[bin_name][int(temp[2]) - int(temp[1])]
                    elif (one_or_sum == 'sum'):
                        exp_pairs_X[index] = pairs_X[index] / exp_vectors_bin[bin_name][int(temp[2]) - int(temp[1])]
                    else:
                        print(one_or_sum, "not in one_or_sum")
                        exit(0)
                else:
                    exp_pairs_X[index] = 0

            pairs_Y = pairs_sum_std_T['std']

            pairs_sumstd = exp_pairs_X * pairs_Y

            sumstd_thr = np.percentile(pairs_sumstd, threshold)
            pairs_features = pairs_sumstd[pairs_sumstd > sumstd_thr].index
            temp_pairs_features_bin_dic[bin_name] = pairs_features
        return temp_pairs_features_bin_dic, chr_name

    pairs_features_chr_bin_dic = {}
    result_list = Parallel(n_jobs=kernel_number)(
        delayed(design_for_parallel_temp_pairs_features_bin_dic)(chr_name, temp_pairs_sum_std_df_bin_dic) for
        chr_name, temp_pairs_sum_std_df_bin_dic in pairs_sum_std_df_chr_bin_dic.items())
    for temp_pairs_features_bin_dic_And_chr_name in result_list:
        pairs_features_chr_bin_dic[temp_pairs_features_bin_dic_And_chr_name[1]] = \
        temp_pairs_features_bin_dic_And_chr_name[0]
    # for chr_name, temp_pairs_sum_std_df_bin_dic in pairs_sum_std_df_chr_bin_dic.items():
    #     temp_pairs_features_bin_dic = {}
    #     for bin_name, pairs_sum_std in temp_pairs_sum_std_df_bin_dic.items():
    #         print(chr_name, bin_name)
    #         pairs_sum_std_T = pairs_sum_std.T
    #         pairs_X = pairs_sum_std_T['sum']
    #
    #         exp_pairs_X = pd.Series()
    #         for index in pairs_X.index:
    #             temp = index.split("+")
    #             location_x = int(temp[2]) - int(temp[1])
    #             EV = exp_vectors_bin[bin_name][location_x]
    #
    #             if EV != 0:
    #                 if(one_or_sum=='one'):
    #                     exp_pairs_X[index] = 1/ exp_vectors_bin[bin_name][int(temp[2]) - int(temp[1])]
    #                 elif(one_or_sum=='sum'):
    #                     exp_pairs_X[index] = pairs_X[index] / exp_vectors_bin[bin_name][int(temp[2]) - int(temp[1])]
    #                 else:
    #                     print(one_or_sum,"not in one_or_sum")
    #                     exit(0)
    #             else:
    #                 exp_pairs_X[index] = 0
    #
    #         pairs_Y = pairs_sum_std_T['std']
    #
    #         pairs_sumstd = exp_pairs_X * pairs_Y
    #
    #         sumstd_thr = np.percentile(pairs_sumstd, threshold)
    #         pairs_features = pairs_sumstd[pairs_sumstd > sumstd_thr].index
    #         temp_pairs_features_bin_dic[bin_name] = pairs_features
    #     pairs_features_chr_bin_dic[chr_name] = temp_pairs_features_bin_dic
    end = datetime.datetime.now()
    print("end, using ", end - start)


    print("start to get all_pairs_features")
    start = datetime.datetime.now()
    all_pairs_features_bin_chr = {}
    all_pairs_features_bin_chr_list = []
    for bin_name in bin_dic:
        all_pairs_features_bin_chr[bin_name] = {}

    # def design_for_parallel_all_pairs_feature_all_bin_one_chr_dic(chr_name, temp_pairs_featrues_bin_dic):
    #     temp_all_bin_one_chr_pairs_features = {}
    #     for bin_name, pairs_features in temp_pairs_featrues_bin_dic.items():
    #         temp_specific_bin_name_AND_chr_name_pairs_features_Series_list = []
    #         pairs_sum_std = pairs_sum_std_df_chr_bin_dic[chr_name][bin_name].T
    #         for fea_index in pairs_features:
    #             print(fea_index)
    #             feature = pairs_sum_std.loc[fea_index, 'values']
    #             reso_pairs_features = np.zeros(len(cell_name_list))
    #             reso_pairs_features_Series = pd.DataFrame(reso_pairs_features, index=cell_name_list,
    #                                                       columns=[bin_name + '+' + fea_index])
    #             for value_index in feature.index:
    #                 reso_pairs_features_Series.loc[value_index] = feature[value_index]
    #             temp_specific_bin_name_AND_chr_name_pairs_features_Series_list.append(reso_pairs_features_Series)
    #         temp_all_bin_one_chr_pairs_features[bin_name] = pd.concat(
    #             temp_specific_bin_name_AND_chr_name_pairs_features_Series_list, axis=1, sort=False)
    #     return temp_all_bin_one_chr_pairs_features, chr_name
    #
    # result_list = Parallel(n_jobs=8)(
    #     delayed(design_for_parallel_all_pairs_feature_all_bin_one_chr_dic)(chr_name, temp_pairs_featrues_bin_dic) for chr_name, temp_pairs_featrues_bin_dic in pairs_features_chr_bin_dic.items())
    # for temp_all_bin_one_chr_pairs_features_And_chr_name in result_list:
    #     chr_name = temp_all_bin_one_chr_pairs_features_And_chr_name[1]
    #     temp_all_bin_one_chr_pairs_features = temp_all_bin_one_chr_pairs_features_And_chr_name[0]
    #     for bin_name, temp_specific_bin_name_AND_chr_name_pairs_features_Series in temp_all_bin_one_chr_pairs_features.items():
    #         all_pairs_features_bin_chr[bin_name][chr_name] = temp_specific_bin_name_AND_chr_name_pairs_features_Series
    for chr_name, temp_pairs_featrues_bin_dic in pairs_features_chr_bin_dic.items():

        for bin_name, pairs_features in temp_pairs_featrues_bin_dic.items():
            temp_specific_bin_name_AND_chr_name_pairs_features_Series_list = []
            pairs_sum_std = pairs_sum_std_df_chr_bin_dic[chr_name][bin_name].T
            for fea_index in pairs_features:
                print(fea_index)
                feature = pairs_sum_std.loc[fea_index, 'values']
                reso_pairs_features = np.zeros(len(cell_name_list))
                reso_pairs_features_Series = pd.DataFrame(reso_pairs_features, index=cell_name_list,
                                                          columns=[bin_name + '+' + fea_index])
                for value_index in feature.index:
                    reso_pairs_features_Series.loc[value_index] = feature[value_index]
                temp_specific_bin_name_AND_chr_name_pairs_features_Series_list.append(reso_pairs_features_Series)
            all_pairs_features_bin_chr[bin_name][chr_name] = pd.concat(
                temp_specific_bin_name_AND_chr_name_pairs_features_Series_list, axis=1, sort=False)

    for bin_name, temp_all_pairs_features in all_pairs_features_bin_chr.items():
        for chr_name, reso_pairs_features_Series in temp_all_pairs_features.items():
            print(bin_name, chr_name)
            all_pairs_features_bin_chr_list.append(reso_pairs_features_Series)
    all_pairs_features = pd.concat(all_pairs_features_bin_chr_list, axis=1, sort=False)
    end = datetime.datetime.now()
    print("end, using ", end - start)
    print('--------------------------')
    all_pairs_features = all_pairs_features.fillna(0)

    temp_name = pd.Series(all_pairs_features.index, name='cell_nm')

    all_pairs_features.index = temp_name

    all_pairs_features.to_csv(all_pcc_file, sep='\t', header=True, index=True)
    print("get PCC de novo using ",end-all_start)
    return all_pairs_features


def get_PCC_vector_de_novo_NotMultiProcess_save_intermediate_file_sepcific_threshold(sim_info,test_data_df_dic,cell_name_list,all_pcc_file,pcc_intermediate_file_dir,threshold,one_or_sum,max_value=50):
    '''

    :param sim_info:
    :param test_data_df_dic: make sure the index of test_data_df_dic has been reset
    :return:
    '''
    print("start to get PCC vector de novo")
    bin_dic = {'100kb': 100000, '500kb': 500000, '1Mb': 1000000}
    print("get bin")
    all_start=datetime.datetime.now()
    for bin_name, resolution in bin_dic.items():
        print(bin_name, resolution)
        for i, enum in enumerate(test_data_df_dic.items()):
            print(i, enum[0])
            one_df = enum[1]
            one_df[bin_name + '1'] = one_df['pos1'] // resolution
            one_df[bin_name + '2'] = one_df['pos2'] // resolution
            test_data_df_dic[enum[0]] = one_df
    all_cell_total_contacts_list=[]
    for cell_name in cell_name_list:
        all_cell_total_contacts_list.append(test_data_df_dic[cell_name]['count'].sum())
    all_cell_total_contacts_df=pd.DataFrame({'total_contacts':all_cell_total_contacts_list},index=cell_name_list)

    test_chr_data_df_dic = {}
    test_chr_combine_df_dic = {}
    print("start to get chr_combine_df_dic")
    start=datetime.datetime.now()
    for chr_name in sim_info.chr_length_dic:
        if (chr_name == 'chrY'):
            continue
        print(chr_name)
        temp_chr_dic = {}
        temp_chr_list = []
        for cell_name, one_df in test_data_df_dic.items():
            chr_cis_df = one_df[(one_df['chr1'] == chr_name) & (one_df['chr2'] == chr_name)]
            temp_chr_dic[cell_name] = chr_cis_df
            temp_chr_list.append(chr_cis_df)
        test_chr_combine_df_dic[chr_name] = pd.concat(temp_chr_list, axis=0, sort=False)
        test_chr_data_df_dic[chr_name] = temp_chr_dic
    end=datetime.datetime.now()
    print("end, using ",end-start)
    pool_chr_bin_matrix = {}
    pool_chr_bin_matrix_file_path=os.path.join(pcc_intermediate_file_dir,'pool_chr_bin_matrix.pkl')

    print("start to get pool_chr_bin_matrix")
    start = datetime.datetime.now()
    if(os.path.exists(pool_chr_bin_matrix_file_path)):
        pool_chr_bin_matrix=sim_info.load_variable_from_pikle_file(pool_chr_bin_matrix_file_path)
    else:
        for chr_name, chr_length in sim_info.chr_length_dic.items():
            if (chr_name == 'chrY'):
                continue
            print(chr_name)
            temp_chr_combine_df = test_chr_combine_df_dic[chr_name]
            temp_bin_matrix_dic = {}
            for bin_name, resolution in bin_dic.items():
                print(bin_name, resolution)

                temp_chr_combine_df = temp_chr_combine_df.reset_index(drop=True)
                row = temp_chr_combine_df[[bin_name + '1', bin_name + '2']].min(axis=1).values
                col = temp_chr_combine_df[[bin_name + '1', bin_name + '2']].max(axis=1).values
                count = temp_chr_combine_df['count'].values
                temp_bin_length = chr_length // resolution + 1
                coo_CertainChr_CertainBin_combine = coo_matrix((count, (row, col)),
                                                               shape=(int(temp_bin_length), int(temp_bin_length)),
                                                               dtype=np.int)
                upper_matrix = coo_CertainChr_CertainBin_combine.todense()
                lower_matrix = np.mat(np.tril(upper_matrix.T, -1))
                full_matrix = upper_matrix + lower_matrix
                temp_bin_matrix_dic[bin_name] = full_matrix

            pool_chr_bin_matrix[chr_name] = temp_bin_matrix_dic
        sim_info.store_variable_from_pikle_file(pool_chr_bin_matrix_file_path,pool_chr_bin_matrix)

    end = datetime.datetime.now()
    print("end, using ",end - start)

    chr_pair_df = pd.DataFrame()
    pairs_sum_std_df_chr_bin_dic = {}
    pairs_sum_std_df_chr_bin_dic_file_path=os.path.join(pcc_intermediate_file_dir,'pairs_sum_std_df_chr_bin_dic.pkl')
    print("start to get pairs_sum_std_df_chr_bin_dic")
    start = datetime.datetime.now()
    if(os.path.exists(pairs_sum_std_df_chr_bin_dic_file_path)):
        pairs_sum_std_df_chr_bin_dic=sim_info.load_variable_from_pikle_file(pairs_sum_std_df_chr_bin_dic_file_path)
    else:

        for chr_name, chr_length in sim_info.chr_length_dic.items():
            if (chr_name == 'chrY'):
                continue
            temp_chr_combine_df = test_chr_combine_df_dic[chr_name]
            temp_chr_combine_df = temp_chr_combine_df.reset_index(drop=True)
            temp_pairs_sum_std_df_bin_dic = {}
            for bin_name, resolution in bin_dic.items():
                pairs_sum_std = pd.DataFrame()
                min_temp=temp_chr_combine_df[[bin_name + '1', bin_name + '2']].min(axis=1)
                max_temp=temp_chr_combine_df[[bin_name + '1', bin_name + '2']].max(axis=1)
                temp_chr_combine_df[bin_name + '1'] = min_temp
                temp_chr_combine_df[bin_name + '2'] = max_temp
                bin_pair_with_name_df = \
                temp_chr_combine_df.groupby(['cell_name', bin_name + '1', bin_name + '2'], as_index=False)['count'].sum()
                print(chr_name, bin_name, resolution)
                chr_bin_length = chr_length // resolution + 1
                print('chr_bin_length', chr_bin_length)
                all_bin_pair_df = bin_pair_with_name_df.groupby([bin_name + '1', bin_name + '2'])['count'].sum()
                # all_bin_pair_more_than_threshold_series = all_bin_pair_df.loc[all_bin_pair_df.values >= 50]
                all_bin_pair_more_than_threshold_series = all_bin_pair_df.loc[all_bin_pair_df.values >= max_value]

                bin_pair_with_name_df_with_index = bin_pair_with_name_df.set_index([bin_name + '1', bin_name + '2'])

                # def get_temp_pair_sum_std(bin_pair_with_name_df_with_index, bin_tuple, bin_sum):
                #     bin1 = bin_tuple[0]
                #     bin2 = bin_tuple[1]
                #
                #     all_sc_pair_contacts = \
                #     bin_pair_with_name_df_with_index.loc[bin1, bin2][['cell_name', 'count']].set_index('cell_name',
                #                                                                                        drop=True)['count']
                #
                #     pairs_sum = bin_sum
                #     pairs_std = all_sc_pair_contacts.values.std()
                #     temp_pair_sum_std = pd.Series([pairs_sum, pairs_std], index=['sum', 'std'])
                #     temp_pair_sum_std.loc['values'] = all_sc_pair_contacts
                #     temp_pair_sum_std.name = chr_name + '+' + str(bin1) + '+' + str(bin2)
                #     return temp_pair_sum_std


                # result_list = Parallel(n_jobs=24)(
                #     delayed(get_temp_pair_sum_std)(bin_pair_with_name_df_with_index, bin_tuple, bin_sum) for
                #     bin_tuple, bin_sum in all_bin_pair_more_than_threshold_series.items())
                # for temp_pair_sum_std in result_list:
                #     pairs_sum_std[temp_pair_sum_std.name] = temp_pair_sum_std

                for bin_tuple, bin_sum in all_bin_pair_more_than_threshold_series.items():
                    # print(bin_name,bin_tuple,bin_sum)
                    bin1 = bin_tuple[0]
                    bin2 = bin_tuple[1]

                    print(bin_name,chr_name,bin1,bin2,"start to get all_sc_pair_contacts")
                    start=datetime.datetime.now()

                    all_sc_pair_contacts = \
                    bin_pair_with_name_df_with_index.loc[bin1, bin2][['cell_name', 'count']].set_index('cell_name',
                                                                                                       drop=True)['count']

                    end=start=datetime.datetime.now()
                    print("end,using",end-start)

                    pairs_sum = all_sc_pair_contacts.sum()
                    pairs_std = all_sc_pair_contacts.values.std()
                    temp_pair_sum_std = pd.Series([pairs_sum, pairs_std], index=['sum', 'std'])
                    temp_pair_sum_std.loc['values'] = all_sc_pair_contacts
                    pairs_sum_std[chr_name + '+' + str(bin1) + '+' + str(bin2)] = temp_pair_sum_std
                temp_pairs_sum_std_df_bin_dic[bin_name] = pairs_sum_std
            pairs_sum_std_df_chr_bin_dic[chr_name] = temp_pairs_sum_std_df_bin_dic
        sim_info.store_variable_from_pikle_file(pairs_sum_std_df_chr_bin_dic_file_path, pairs_sum_std_df_chr_bin_dic)
    end = datetime.datetime.now()
    print("end, using ",end - start)

    genome_exp_vectors_chr_bin = {}
    genome_exp_vectors_chr_bin_file_path=os.path.join(pcc_intermediate_file_dir,'genome_exp_vectors_chr_bin.pkl')
    print("start to get genome_exp_vectors_chr_bin")
    start = datetime.datetime.now()
    if(os.path.exists(genome_exp_vectors_chr_bin_file_path)):
        genome_exp_vectors_chr_bin = sim_info.load_variable_from_pikle_file(genome_exp_vectors_chr_bin_file_path)
    else:
        for chr_name in sim_info.chr_length_dic:
            if (chr_name == 'chrY'):
                continue
            temp_genome_exp_vectors_bin = {}
            for bin_name, resolution in bin_dic.items():
                print(chr_name, bin_name, resolution)

                one_matrix = pool_chr_bin_matrix[chr_name][bin_name]

                bin_size = sim_info.chr_length_dic[chr_name] // resolution + 1

                chr_pool_map = one_matrix
                sparse_Mat = sparse.coo_matrix(chr_pool_map)
                sparse_tril_Mat = sparse.tril(sparse_Mat)
                triplets = np.column_stack((sparse_tril_Mat.col, sparse_tril_Mat.row, sparse_tril_Mat.data))
                triplets_DF = pd.DataFrame(triplets, columns=['bin1', 'bin2', 'values'])
                triplets_DF = triplets_DF.fillna(0)
                exp_vectors = np.zeros(bin_size)
                for k in range(bin_size):
                    all_counts = triplets_DF.loc[(triplets_DF['bin2'] - triplets_DF['bin1'] == k)]['values']
                    temp_sum = np.sum(all_counts)
                    if not np.isnan(temp_sum):
                        exp_vectors[k] = temp_sum / (bin_size - k)
                temp_genome_exp_vectors_bin[bin_name] = exp_vectors
            genome_exp_vectors_chr_bin[chr_name] = temp_genome_exp_vectors_bin
        sim_info.store_variable_from_pikle_file(genome_exp_vectors_chr_bin_file_path, genome_exp_vectors_chr_bin)
    end = datetime.datetime.now()
    print("end, using ", end - start)

    print("start to get exp_vectors_bin")
    start = datetime.datetime.now()
    exp_vectors_bin = {}
    for bin_name, resolution in bin_dic.items():
        print(bin_name, resolution)

        chr_length_bin_dic = sim_info.get_chr_length_bin_without_Y(resolution)
        chr_bin_size_list = []
        for chr_name, chr_bin in chr_length_bin_dic.items():
            chr_bin_size_list.append(chr_bin)

        exp_len = np.max(np.array(chr_bin_size_list))
        exp_vectors = np.zeros(int(exp_len))
        for k in range(int(exp_len)):
            all_contact_counts = 0
            n_counts = 0
            for chr_name, chr_bin in chr_length_bin_dic.items():
                if chr_bin > k:
                    all_contact_counts += genome_exp_vectors_chr_bin[chr_name][bin_name][k] * (chr_bin - k)
                    n_counts += (chr_bin - k)
            if n_counts != 0:
                exp_vectors[k] = all_contact_counts / n_counts
        exp_vectors_bin[bin_name] = exp_vectors

    end = datetime.datetime.now()
    print("end, using ", end - start)
    print("start to pairs_features_chr_bin_dic")
    start= datetime.datetime.now()
    pairs_features_chr_bin_dic = {}
    for chr_name, temp_pairs_sum_std_df_bin_dic in pairs_sum_std_df_chr_bin_dic.items():
        temp_pairs_features_bin_dic = {}
        for bin_name, pairs_sum_std in temp_pairs_sum_std_df_bin_dic.items():
            print(chr_name, bin_name)
            pairs_sum_std_T = pairs_sum_std.T
            pairs_X = pairs_sum_std_T['sum']

            exp_pairs_X = pd.Series()
            for index in pairs_X.index:
                temp = index.split("+")
                location_x = int(temp[2]) - int(temp[1])

                if(location_x==0):
                    exp_pairs_X[index]=0
                    continue
                else:
                    exp_pairs_X[index] = 1
                    continue
                EV = exp_vectors_bin[bin_name][location_x]

                if EV != 0:
                    if(one_or_sum=='one'):
                        exp_pairs_X[index] = 1/ exp_vectors_bin[bin_name][int(temp[2]) - int(temp[1])]
                    elif(one_or_sum=='sum'):
                        exp_pairs_X[index] = pairs_X[index] / exp_vectors_bin[bin_name][int(temp[2]) - int(temp[1])]
                    else:
                        print(one_or_sum,"not in one_or_sum")
                        exit(0)
                else:
                    exp_pairs_X[index] = 0

            pairs_Y = pairs_sum_std_T['std']

            pairs_sumstd = exp_pairs_X * pairs_Y

            #sumstd_thr = np.percentile(pairs_sumstd, 99.5)

            sumstd_thr = np.percentile(pairs_sumstd, threshold)
            pairs_features = pairs_sumstd[pairs_sumstd > sumstd_thr].index
            temp_pairs_features_bin_dic[bin_name] = pairs_features
        pairs_features_chr_bin_dic[chr_name] = temp_pairs_features_bin_dic
    end = datetime.datetime.now()
    print("end, using ", end - start)


    print("start to get all_pairs_features")
    start = datetime.datetime.now()
    all_pairs_features = pd.DataFrame()
    all_pairs_features_bin_chr = {}
    all_pairs_features_bin_chr_list = []
    for bin_name in bin_dic:
        all_pairs_features_bin_chr[bin_name] = {}

    for chr_name, temp_pairs_featrues_bin_dic in pairs_features_chr_bin_dic.items():

        for bin_name, pairs_features in temp_pairs_featrues_bin_dic.items():
            temp_specific_bin_name_AND_chr_name_pairs_features_Series_list = []
            pairs_sum_std = pairs_sum_std_df_chr_bin_dic[chr_name][bin_name].T
            for fea_index in pairs_features:
                print(fea_index)
                feature = pairs_sum_std.loc[fea_index, 'values']
                reso_pairs_features = np.zeros(len(cell_name_list))
                reso_pairs_features_Series = pd.DataFrame(reso_pairs_features, index=cell_name_list,
                                                          columns=[bin_name + '+' + fea_index])
                for value_index in feature.index:
                    reso_pairs_features_Series.loc[value_index] = feature[value_index]
                temp_specific_bin_name_AND_chr_name_pairs_features_Series_list.append(reso_pairs_features_Series)
            all_pairs_features_bin_chr[bin_name][chr_name] = pd.concat(
                temp_specific_bin_name_AND_chr_name_pairs_features_Series_list, axis=1, sort=False)

    for bin_name, temp_all_pairs_features in all_pairs_features_bin_chr.items():
        for chr_name, reso_pairs_features_Series in temp_all_pairs_features.items():
            print(bin_name, chr_name)
            all_pairs_features_bin_chr_list.append(reso_pairs_features_Series)
    all_pairs_features = pd.concat(all_pairs_features_bin_chr_list, axis=1, sort=False)
    end = datetime.datetime.now()
    print("end, using ", end - start)
    print('--------------------------')
    all_pairs_features = all_pairs_features.fillna(0)

    temp_name = pd.Series(all_pairs_features.index, name='cell_nm')

    all_pairs_features.index = temp_name

    all_pairs_features.to_csv(all_pcc_file, sep='\t', header=True, index=True)
    print("get PCC de novo using ",end-all_start)
    return all_pairs_features

def run_pca_package(X,n_components):
    from sklearn.decomposition import PCA
#     n_components=100
    pca1 = PCA(n_components=n_components)
    pca1.fit(X)
    X_new = pca1.transform(X)
    return X_new

def write_out_cell_cell_distance_df(cell_cell_distance_df,features_dir):
    cell_cell_distance_df_path=os.path.join(features_dir,"cell_cell_distance.txt")
    print("Store cell-cell distance in:", cell_cell_distance_df_path)
    cell_cell_distance_df.to_csv(cell_cell_distance_df_path,sep='\t')

def read_in_cell_cell_distance_df(features_dir):
    cell_cell_distance_df_path=os.path.join(features_dir,"cell_cell_distance.txt")
    cell_cell_distance_df=pd.read_table(cell_cell_distance_df_path, index_col=0)
    return cell_cell_distance_df

def get_cell_cell_distance(sim_info,features_dir,n_components=100):
    from sklearn.preprocessing import minmax_scale

    feature_files = ['pcc_file.txt', 'cdd_file.txt']
    Isminmax = True

    temp_data = pd.DataFrame()
    # filename=sub_feature_files[0]
    for k, filename in enumerate(feature_files):
        feature_data = pd.read_table(os.path.join(features_dir,filename), sep='\t', header=0, index_col=0)
        if np.isnan(np.max(feature_data.values)):
            feature_data = feature_data.fillna(0)
        if Isminmax:
            fea_data = minmax_scale(feature_data, feature_range=(0.01, 1), axis=0, copy=True)
            feature_data = pd.DataFrame(fea_data, columns=feature_data.columns, index=feature_data.index)
        else:
            feature_data = feature_data + 0.01
        temp_data = pd.concat([temp_data, feature_data], axis=1)
    data = temp_data
    molecule_counts = data.sum(axis=1)
    data = data.div(molecule_counts, axis=0) \
        .mul(np.median(molecule_counts), axis=0)
    nonzero_genes = data.sum(axis=0) != 0
    data = data.loc[:, nonzero_genes].astype(np.float32)
    X = data.values
    # n_components = 100
    shape_min = min(X.shape[0], X.shape[1])
    if (n_components > shape_min):
        n_components = shape_min
    pca_X = run_pca_package(X, n_components)
    pca_X_df = pd.DataFrame(pca_X, index=data.index)
    cell_name_list = sim_info.cell_name_list

    cell_cell_distance_list = []
    for cell_name_1 in cell_name_list:
        vec1 = pca_X_df.loc[cell_name_1].values
        temp_distance_list = []
        for cell_name_2 in cell_name_list:
            vec2 = pca_X_df.loc[cell_name_2].values
            dist = np.linalg.norm(vec1 - vec2)
            temp_distance_list.append(dist)
        cell_cell_distance_list.append(temp_distance_list)
    cell_cell_distance_df = pd.DataFrame(cell_cell_distance_list, columns=cell_name_list, index=cell_name_list)

    write_out_cell_cell_distance_df(cell_cell_distance_df, features_dir)

def get_adj_cell_name_by_cell_cell_distance(cell_cell_distance_df,cell_name,combine_number):
    one_cell_series = cell_cell_distance_df[cell_name]
    one_cell_sort_series = one_cell_series.sort_values(ascending=True)
    result_list=list(one_cell_sort_series.index)[0:combine_number+1]
    if cell_name not in result_list:
        result_list=[cell_name]+result_list[0:-1]
    return result_list
def get_MCM_vector(sim_info,sim_mcm_file_path,sim_name_list,sim_suffix):
    '''
    the values of sim_name_list not contain sim_suffix. For example, it's 1CDX1_434, not 1CDX1_434+SIM_seqDepth_1.
    :param sim_info:
    :param sim_mcm_file_path:
    :param sim_name_list:
    :return:
    '''
    mcm_file = os.path.join(sim_info.cell_cycle_dir, "MCM.txt")
    mcm_df = pd.read_table(mcm_file, header=None, index_col=0)

    sim_mcm_df = mcm_df.loc[sim_name_list]

    sim_mcm_df.index = sim_mcm_df.index.values + sim_suffix
    sim_mcm_df.index.name='cell_nm'
    sim_mcm_df.to_csv(sim_mcm_file_path, sep='\t', header=True,index=True)

    return sim_mcm_df
def get_passed_qc_sc_DF_vector(sim_info,sim_passed_qc_sc_df_file_path,sim_name_list,sim_suffix):
    '''
    the values of sim_name_list not contain sim_suffix. For example, it's 1CDX1_434, not 1CDX1_434+SIM_seqDepth_1.
    :param sim_info:
    :param sim_passed_qc_sc_df_file_path:
    :param sim_name_list:
    :param sim_suffix:
    :return:
    '''
    passed_qc_sc_file = os.path.join(sim_info.cell_cycle_dir, "passed_qc_sc_DF.txt")
    passed_qc_sc_df = pd.read_table(passed_qc_sc_file, index_col=0,header=0)

    sim_passed_qc_sc_df = passed_qc_sc_df.loc[sim_name_list]

    sim_passed_qc_sc_df.index = sim_passed_qc_sc_df.index.values + sim_suffix
    sim_passed_qc_sc_df.index.name=passed_qc_sc_df.index.name
    sim_passed_qc_sc_df.to_csv(sim_passed_qc_sc_df_file_path, sep='\t',header=True,index=True)
    return sim_passed_qc_sc_df
def plot_phased_cell_heatmap(sim_info,circlet_result_file,fig_save_path,ascending=True):
    #raw_result_path = r"E:\Users\scfan\data\CellCycle\experiment\CIRCLET\raw\rebuild_norm_pcc_de_novo_98.0\result_cdd+pcc\CIRC_result.txt"
    raw_result_path=circlet_result_file
    raw_circlet_df = pd.read_table(raw_result_path, index_col=0, header=None)
    raw_circlet_df.columns = ['circlet_values']
    raw_circlet_df = raw_circlet_df.sort_values(['circlet_values'], ascending=ascending)
    circlet_df = raw_circlet_df
    cell_stage_label = {"G1": 1, "eS": 2, "mS": 3, "lS_G2": 4}
    for cell_stage, cell_name_list in sim_info.KeyStage_ValueCellNameList_dic.items():
        circlet_df[cell_stage] = 0
        for cell_name in cell_name_list:
            circlet_df[cell_stage][cell_name] = cell_stage_label[cell_stage]
    heatmap_array = np.array(
        [circlet_df['G1'].values, circlet_df['eS'].values, circlet_df['mS'].values, circlet_df['lS_G2'].values])
    import seaborn as sns
    import matplotlib.pyplot as plt
    sns.set()
    np.random.seed(0)
    uniform_data = np.random.rand(10, 12)
    uniform_data = np.ones([4, 12])
    flatui = ["white", "red", "green", "blue", "pink"]
    cmap = sns.color_palette(flatui)
    my_cmap = ListedColormap(cmap.as_hex())
    ax = sns.heatmap(heatmap_array, cmap=my_cmap,cbar=False)
    plt.yticks([0, 1, 2, 3], ['G1', 'ES', 'MS', 'LS/G2'], rotation=0,fontsize=15)
    plt.xticks([])
    plt.xlabel('phased cells')
    fig = plt.gcf()
    plt.show()
    fig.legend()
    fig.legends=[]
    # fig.savefig(
    #     r"E:\Users\scfan\data\CellCycle\experiment\CIRCLET\raw\rebuild_norm_pcc_de_novo_98.0\result_cdd+pcc\fig.png")
    fig.savefig(fig_save_path)