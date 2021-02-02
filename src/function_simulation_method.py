# -*- coding: utf-8 -*-
# @Time    : 2020/8/25 15:13
# @Author  : scfan
# @FileName: function_distance_stratified_simulation.py
# @Software: PyCharm
import warnings

warnings.simplefilter(action='ignore',category=FutureWarning)
warnings.filterwarnings('ignore')
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
import math
import sys
from joblib import Parallel, delayed
from tqdm import tqdm
# sys.path.append(r"E:\Users\scfan\program")

# print(sys.path)
import src.function_prepare_features as features
# import function_prepare_features as features

def store_interval_bin_df_total_trans_cis_dic(sim_info,total_contacts_bin_df,trans_ratio_bin_df,cis_cell_dis_bin_df_dic):
    bin_df_total_trans_cis_dic = {}
    bin_df_total_trans_cis_dic['total_contacts_bin_df'] = total_contacts_bin_df
    bin_df_total_trans_cis_dic['trans_ratio_bin_df'] = trans_ratio_bin_df
    bin_df_total_trans_cis_dic['cis_cell_dis_bin_df_dic'] = cis_cell_dis_bin_df_dic
    bin_df_total_trans_cis_dic_path=os.path.join(sim_info.raw_single_cell_dir,"interval_bin_df_total_trans_cis_dic.pkl")
    sim_info.store_variable_from_pikle_file(bin_df_total_trans_cis_dic_path,bin_df_total_trans_cis_dic)

def load_interval_bin_df_total_trans_cis_dic(sim_info):
    bin_df_total_trans_cis_dic_path = os.path.join(sim_info.raw_single_cell_dir,
                                                   "interval_bin_df_total_trans_cis_dic.pkl")
    bin_df_total_trans_cis_dic=sim_info.load_variable_from_pikle_file(bin_df_total_trans_cis_dic_path)
    total_contacts_bin_df=bin_df_total_trans_cis_dic['total_contacts_bin_df']
    trans_ratio_bin_df =bin_df_total_trans_cis_dic['trans_ratio_bin_df']
    cis_cell_dis_bin_df_dic=bin_df_total_trans_cis_dic['cis_cell_dis_bin_df_dic']
    return total_contacts_bin_df, trans_ratio_bin_df, cis_cell_dis_bin_df_dic

def get_bin_interval_df(task_df, Bin_interval_number, task_column_name):
    #     Bin_interval_number=200
    cell_name_array = task_df.index.values
    contacts_min = task_df[task_column_name].min()
    contacts_max = task_df[task_column_name].max()
    if (contacts_min == contacts_max):
        task_bin_df = pd.DataFrame(index=cell_name_array)
        task_bin_df[task_column_name] = task_df[task_column_name]
        task_bin_df['Bin'] = 0
        temp_series = pd.Series()
        for cell_name in task_bin_df.index.values:
            temp_series.loc[cell_name] = [contacts_min]
        temp_series.name = 'sample'
        task_bin_df = pd.concat([task_bin_df, temp_series], axis=1, sort=False)
    else:
        #    print('min,max',contacts_min,contacts_max)
        Bin_interval_len = (contacts_max - contacts_min) / Bin_interval_number
        print('min,max,Bin_interval_len', contacts_min, contacts_max, Bin_interval_len)
        temp_series = pd.Series()
        for cell_name in task_df.index.values:
            temp_number = task_df.loc[cell_name][task_column_name]
            # print(temp_number)
            temp_bin = int((temp_number - contacts_min) // Bin_interval_len)
            if (temp_bin * Bin_interval_len + contacts_min == temp_number):
                if (temp_bin == 0):
                    temp_series.loc[cell_name] = temp_bin
                else:
                    print(cell_name, 'on the boundary', temp_number, temp_bin)
                    temp_series.loc[cell_name] = temp_bin - 1
            else:
                temp_series.loc[cell_name] = temp_bin
        temp_series.name = 'Bin'
        Bin_task_df = pd.DataFrame(index=cell_name_array)
        Bin_task_df = pd.concat([Bin_task_df, temp_series], axis=1, sort=False)

        max_Bin = Bin_task_df['Bin'].max()
        Bin_cell_name_list_dic = {}
        temp_Series = pd.Series()
        for i in range(max_Bin + 1):
            Bin_cell_name_list_dic[i] = []
        for cell_name in Bin_task_df.index.values:
            temp_bin = Bin_task_df.loc[cell_name]['Bin']
            Bin_cell_name_list_dic[temp_bin].append(task_df.loc[cell_name][task_column_name])

        for cell_name in Bin_task_df.index.values:
            temp_bin = Bin_task_df.loc[cell_name]['Bin']
            temp_Series.loc[cell_name] = Bin_cell_name_list_dic[temp_bin]
        temp_Series.name = 'sample'
        Bin_task_df = pd.concat([Bin_task_df, temp_Series], axis=1, sort=False)

        task_bin_df = pd.DataFrame(index=cell_name_array)
        task_bin_df[task_column_name] = task_df[task_column_name]
        task_bin_df['Bin'] = Bin_task_df['Bin']
        task_bin_df['sample'] = Bin_task_df['sample']
    return task_bin_df

def get_bin_interval_df_multi_process(task_df, Bin_interval_number, task_column_name):
    #     Bin_interval_number=200
    cell_name_array = task_df.index.values
    contacts_min = task_df[task_column_name].min()
    contacts_max = task_df[task_column_name].max()
    if (contacts_min == contacts_max):
        task_bin_df = pd.DataFrame(index=cell_name_array)
        task_bin_df[task_column_name] = task_df[task_column_name]
        task_bin_df['Bin'] = 0
        temp_series = pd.Series()
        for cell_name in task_bin_df.index.values:
            temp_series.loc[cell_name] = [contacts_min]
        temp_series.name = 'sample'
        task_bin_df = pd.concat([task_bin_df, temp_series], axis=1, sort=False)
    else:
        #    print('min,max',contacts_min,contacts_max)
        Bin_interval_len = (contacts_max - contacts_min) / Bin_interval_number
        print('min,max,Bin_interval_len', contacts_min, contacts_max, Bin_interval_len)
        temp_series = pd.Series()
        for cell_name in task_df.index.values:
            temp_number = task_df.loc[cell_name][task_column_name]
            # print(temp_number)
            temp_bin = int((temp_number - contacts_min) // Bin_interval_len)
            if (temp_bin * Bin_interval_len + contacts_min == temp_number):
                if (temp_bin == 0):
                    temp_series.loc[cell_name] = temp_bin
                else:
                    print(cell_name, 'on the boundary', temp_number, temp_bin)
                    temp_series.loc[cell_name] = temp_bin - 1
            else:
                temp_series.loc[cell_name] = temp_bin
        temp_series.name = 'Bin'
        Bin_task_df = pd.DataFrame(index=cell_name_array)
        Bin_task_df = pd.concat([Bin_task_df, temp_series], axis=1, sort=False)

        max_Bin = Bin_task_df['Bin'].max()
        Bin_cell_name_list_dic = {}
        temp_Series = pd.Series()
        for i in range(max_Bin + 1):
            Bin_cell_name_list_dic[i] = []
        for cell_name in Bin_task_df.index.values:
            temp_bin = Bin_task_df.loc[cell_name]['Bin']
            Bin_cell_name_list_dic[temp_bin].append(task_df.loc[cell_name][task_column_name])

        for cell_name in Bin_task_df.index.values:
            temp_bin = Bin_task_df.loc[cell_name]['Bin']
            temp_Series.loc[cell_name] = Bin_cell_name_list_dic[temp_bin]
        temp_Series.name = 'sample'
        Bin_task_df = pd.concat([Bin_task_df, temp_Series], axis=1, sort=False)

        task_bin_df = pd.DataFrame(index=cell_name_array)
        task_bin_df[task_column_name] = task_df[task_column_name]
        task_bin_df['Bin'] = Bin_task_df['Bin']
        task_bin_df['sample'] = Bin_task_df['sample']
    return task_bin_df,task_column_name

def interval_sample(sample_df,result_df,column_name):
    sample_temp_series=pd.Series()
    bin_min=sample_df['sample'].min()
    bin_max=sample_df['sample'].max()
    sample_value=sample_df['sample'].iloc[0]
    if(bin_max==bin_min and bin_max ==0 and len(sample_value)==1):
        sample_temp_series=pd.Series(sample_value*sample_df.shape[0],index=sample_df.index.values)
        sample_temp_series.name=column_name
        result_df=pd.concat([result_df,sample_temp_series],axis=1,sort=False)
    else:
        for cell_name in sample_df.index.values:
            temp_sample=sample_df.loc[cell_name]['sample']

            propItem=np.ones(len(temp_sample))/len(temp_sample)
            all_item_number=len(temp_sample)
            sample_item_number=1
            sample_index = np.random.choice(all_item_number, sample_item_number, replace=False,
                                                         p=list(propItem))

            sample_temp_series.loc[cell_name]=temp_sample[sample_index[0]]
        sample_temp_series.name=column_name
        result_df=pd.concat([result_df,sample_temp_series],axis=1,sort=False)
    return result_df

def interval_sample_multi_process(sample_df,column_name):
    sample_temp_series=pd.Series()
    bin_min=sample_df['sample'].min()
    bin_max=sample_df['sample'].max()
    sample_value=sample_df['sample'].iloc[0]
    if(bin_max==bin_min and bin_max ==0 and len(sample_value)==1):
        sample_temp_series=pd.Series(sample_value*sample_df.shape[0],index=sample_df.index.values)
        sample_temp_series.name=column_name
    else:
        for cell_name in sample_df.index.values:
            temp_sample=sample_df.loc[cell_name]['sample']

            propItem=np.ones(len(temp_sample))/len(temp_sample)
            all_item_number=len(temp_sample)
            sample_item_number=1
            sample_index = np.random.choice(all_item_number, sample_item_number, replace=False,
                                                         p=list(propItem))

            sample_temp_series.loc[cell_name]=temp_sample[sample_index[0]]
        sample_temp_series.name=column_name
    return sample_temp_series,column_name

def interval_sample_combine(sample_temp_series,result_df):
    result_df = pd.concat([result_df, sample_temp_series], axis=1, sort=False)
    return result_df
def get_using_raw_neighbor_HiCRep(sim_info,neighbor_number,target_sample_prop):
    raw_sim_HiCRep_file_path = r"E:\Users\scfan\data\CellCycle\experiment\Exp3_loop\result\raw_sim_from_combine_seqDepth1_HiCRep_100kb_stratum20.txt"
    raw_sim_HiCRep_df = pd.read_csv(raw_sim_HiCRep_file_path, index_col=0, header=0, sep="\t")
    #neighbor_number = 20
    sim_suffix = "+SIM_seqDepth_1"
    raw_sim_HiCRep_df
    using_raw_sim_HiCRep_Series, using_raw_neighbor_HiCRep, using_sim_neighbor_HiCRep = sbi.get_RawSim_RawNeighbor_SimNeighbor_HiCRep(
        sim_info,
        raw_sim_HiCRep_df,
        neighbor_number, sim_suffix)
    all_cell_neighbor_sample_prop_dic = {}
    # target_sample_prop = 0.1
    #neighbor_number = 20
    for i, cell_name in enumerate(using_raw_neighbor_HiCRep.columns.values):
        print(i, cell_name)
        neighbor_HiCRep_without_target = using_raw_neighbor_HiCRep[cell_name].values
        neighbor_HiCRep_without_target_prop = neighbor_HiCRep_without_target / np.sum(neighbor_HiCRep_without_target)

        neighbor_HiCRep_without_target_prop = neighbor_HiCRep_without_target_prop * (1 - target_sample_prop)

        neighbor_HiCRep_prop = np.insert(neighbor_HiCRep_without_target_prop, neighbor_number // 2, target_sample_prop)
        all_cell_neighbor_sample_prop_dic[cell_name] = neighbor_HiCRep_prop
    extreme_min = 0.0000001
    all_positive_all_cell_neighbor_sample_prop_dic = {}

    for cell_name, one_cell_neighbor_prop in all_cell_neighbor_sample_prop_dic.items():
        temp = np.clip(one_cell_neighbor_prop, extreme_min, np.inf)
        all_positive_all_cell_neighbor_sample_prop_dic[cell_name] = temp
    return all_positive_all_cell_neighbor_sample_prop_dic
def convert_distance_to_bin(distance, step):
    return (np.log2(distance) + step) // step

def convert_bin_to_distance(bin_of_distance, step):
    return int(2 ** ((bin_of_distance - 1) * step))

def get_max_bin(sim_info,step):
    # step = (math.log(10, 2)) / 80
    #print(step)

    max_length = 0
    for chr_name, chr_length in sim_info.chr_length_dic.items():
        if (max_length < chr_length):
            max_length = chr_length

    max_bin = convert_distance_to_bin(max_length, step)
    #print(max_bin)
    return int(max_bin)
def get_genomic_distance_stratified_cis_trans_from_cell_name_list(sim_info,cell_name_list):
    all_cell_cis_df_dic = {}
    all_cell_cis_distance_stratified_df_dic_dic = {}
    all_cell_trans_dic = {}

    step = (math.log(10, 2)) / 80
    print(step)

    max_length = 0
    for chr_name, chr_length in sim_info.chr_length_dic.items():
        if (max_length < chr_length):
            max_length = chr_length

    max_bin = convert_distance_to_bin(max_length, step)
    print(max_bin)

    i = 0
    for cell_name in cell_name_list:

        one_df=sim_info.all_raw_cell_acp_count_one_df_dic[cell_name]
        print(i, cell_name)
        i = i + 1
        one_df = one_df.reset_index(drop=False)
        trans_df = one_df[one_df['chr1'] != one_df['chr2']]
        trans_df.reset_index(inplace=True, drop=True)

        trans_df['prop'] = trans_df['count'].values / trans_df['count'].sum()
        all_cell_trans_dic[cell_name] = trans_df

        cis_df = one_df[one_df['chr1'] == one_df['chr2']]
        cis_df.reset_index(inplace=True, drop=True)
        all_cell_cis_df_dic[cell_name] = cis_df

        cis_distance_stratified_dic = {}

        for each_distance in np.arange(0, int(max_bin) + 1):
            cis_distance_stratified_dic[int(each_distance)] = pd.DataFrame()

        cis_df.reset_index(inplace=True, drop=True)
        distance_series = (cis_df['pos1'] - cis_df['pos2']).abs()
        bin_series = (np.log2(distance_series) + step) // step
        bin_series = bin_series.astype('int64')
        cis_df['distance'] = bin_series
        group_by_result = cis_df.groupby(['distance'])
        for distance_name, temp_df in group_by_result:
            temp_df.reset_index(inplace=True, drop=True)
            temp_df['prop'] = temp_df['count'].values / temp_df['count'].sum()
            cis_distance_stratified_dic[distance_name] = temp_df
        all_cell_cis_distance_stratified_df_dic_dic[cell_name] = cis_distance_stratified_dic
    return all_cell_cis_df_dic,all_cell_cis_distance_stratified_df_dic_dic,all_cell_trans_dic

def get_genomic_distance_stratified_cis_trans_from_df(sim_info,one_df,step):
    # step = (math.log(10, 2)) / 80
    # print(step)

    max_length = 0
    for chr_name, chr_length in sim_info.chr_length_dic.items():
        if (max_length < chr_length):
            max_length = chr_length

    max_bin = convert_distance_to_bin(max_length, step)
    # print(max_bin)

    # one_df = sim_info.read_chr_pos(raw_data_dir, cell_name)


    trans_df = one_df[one_df['chr1'] != one_df['chr2']]
    trans_df.reset_index(inplace=True, drop=True)

    trans_df['prop'] = trans_df['count'].values / trans_df['count'].sum()


    cis_df = one_df[one_df['chr1'] == one_df['chr2']]
    cis_df.reset_index(inplace=True, drop=True)


    cis_distance_stratified_dic = {}

    for each_distance in np.arange(0, int(max_bin) + 1):
        cis_distance_stratified_dic[int(each_distance)] = pd.DataFrame()

    cis_df.reset_index(inplace=True, drop=True)
    distance_series = (cis_df['pos1'] - cis_df['pos2']).abs()
    bin_series = (np.log2(distance_series) + step) // step
    bin_series = bin_series.astype('int64')
    cis_df['distance'] = bin_series
    group_by_result = cis_df.groupby(['distance'])
    for distance_name, temp_df in group_by_result:
        temp_df.reset_index(inplace=True, drop=True)
        temp_df['prop'] = temp_df['count'].values / temp_df['count'].sum()
        cis_distance_stratified_dic[distance_name] = temp_df


    return trans_df,cis_df,cis_distance_stratified_dic



def generate_statistic_info_and_count(sim_info,raw_data_dir,cell_name,step):
#     cell_name='1CDX1_434'

    one_df = sim_info.read_chr_pos(raw_data_dir, cell_name)
    trans_df,cis_df,cis_distance_stratified_dic=get_genomic_distance_stratified_cis_trans_from_df(sim_info,one_df,step)

    intermediate_file_dir=os.path.join(raw_data_dir,cell_name,'intermediate_file')
    sim_info.recursive_mkdir(intermediate_file_dir)

    trans_df_pkl_path=os.path.join(intermediate_file_dir,'trans_df.pkl')
    sim_info.store_variable_from_pikle_file(trans_df_pkl_path,trans_df)

    cis_df_pkl_path=os.path.join(intermediate_file_dir,'cis_df.pkl')
    sim_info.store_variable_from_pikle_file(cis_df_pkl_path,cis_df)

    cis_distance_stratified_dic_pkl_path=os.path.join(intermediate_file_dir,'cis_distance_stratified_dic.pkl')
    sim_info.store_variable_from_pikle_file(cis_distance_stratified_dic_pkl_path,cis_distance_stratified_dic)

    statistic_info_dic={}

    statistic_info_dic['total_count']=one_df['count'].sum()

    trans_ratio=trans_df['count'].sum()/one_df['count'].sum()
    statistic_info_dic['trans_ratio']=trans_ratio


    temp_series=pd.Series()
    for one_dis,one_dis_df in cis_distance_stratified_dic.items():
        if(len(one_dis_df)!=0):
            temp_series.loc[one_dis]=one_dis_df['count'].sum()/cis_df['count'].sum()
        else:
            temp_series.loc[one_dis]=0
    temp_series.name=cell_name
    statistic_info_dic['cis_dis_ratio']=temp_series

    statistic_info_dic_pkl_path=os.path.join(intermediate_file_dir,'statistic_info_dic.pkl')
    sim_info.store_variable_from_pikle_file(statistic_info_dic_pkl_path,statistic_info_dic)

def run_generate_statisc_info(sim_info,raw_data_dir,cell_name_list,parallel,kernel_number,step):
    if(not parallel):
        for cell_name in cell_name_list:
            generate_statistic_info_and_count(sim_info, raw_data_dir,cell_name,step)
    else:
        Parallel(n_jobs=kernel_number)(
            delayed(generate_statistic_info_and_count)(sim_info, raw_data_dir,cell_name,step) for
            cell_name in cell_name_list)
def collect_generate_and_store_interval_bin_for_raw(sim_info,raw_data_dir,cell_name_list,kernel_number,parallel,step,Bin_interval_number = 200):
    statistic_info_dic_dic = {}

    statistic_info_interval_dic={}
    for cell_name in cell_name_list:
        intermediate_file_dir = os.path.join(raw_data_dir, cell_name, 'intermediate_file')
        statistic_info_dic_pkl_path = os.path.join(intermediate_file_dir, 'statistic_info_dic.pkl')
        statistic_info_dic = sim_info.load_variable_from_pikle_file(statistic_info_dic_pkl_path)
        statistic_info_dic_dic[cell_name] = statistic_info_dic

    max_bin = get_max_bin(sim_info,step)
    cis_dis_df = pd.DataFrame(index=range(max_bin + 1))
    for cell_name in tqdm(cell_name_list):
        temp_series = statistic_info_dic_dic[cell_name]['cis_dis_ratio']
        temp_series.name = cell_name
        cis_dis_df = pd.concat([cis_dis_df, temp_series], axis=1, sort=False)
    cis_cell_dis_df = cis_dis_df.T
    cis_cell_dis_bin_df_dic={}
    # not parallel
    if(not parallel):
        for one_bin in cis_cell_dis_df.columns.values:
            print(one_bin)
            task_column_name = one_bin
            task_df = pd.DataFrame(cis_cell_dis_df[one_bin])
            one_bin_contacts_ratio_bin_df = get_bin_interval_df(task_df, Bin_interval_number, task_column_name)
            cis_cell_dis_bin_df_dic[one_bin] = one_bin_contacts_ratio_bin_df
    else:
    # parallel
        result_list = Parallel(n_jobs=kernel_number)(
            delayed(get_bin_interval_df_multi_process)(pd.DataFrame(cis_cell_dis_df[task_column_name]), Bin_interval_number,task_column_name) for
            task_column_name in cis_cell_dis_df.columns.values)
        for one_bin_contacts_ratio_bin_df_And_one_bin_name in result_list:
            cis_cell_dis_bin_df_dic[one_bin_contacts_ratio_bin_df_And_one_bin_name[1]] = \
                one_bin_contacts_ratio_bin_df_And_one_bin_name[0]

    statistic_info_interval_dic['cis_cell_dis_bin_df_dic']=cis_cell_dis_bin_df_dic

    total_count_df = pd.DataFrame(index=cell_name_list)
    temp_series = pd.Series()
    for cell_name in tqdm(cell_name_list):
        temp_series.loc[cell_name] = statistic_info_dic_dic[cell_name]['total_count']
    temp_series.name = 'total_count'
    total_count_df = pd.concat([total_count_df, temp_series], axis=1, sort=False)
    # Bin_interval_number = 200
    task_column_name = 'total_count'
    task_df = total_count_df
    total_count_bin_df = get_bin_interval_df(task_df, Bin_interval_number, task_column_name)
    statistic_info_interval_dic['total_count_bin_df']=total_count_bin_df

    temp_series=pd.Series()
    for cell_name in tqdm(cell_name_list):
        temp_series.loc[cell_name]=statistic_info_dic_dic[cell_name]['trans_ratio']
    column_name='trans_ratio'
    temp_series.name=column_name
    trans_ratio_df=pd.DataFrame(index=cell_name_list)
    trans_ratio_df=pd.concat([trans_ratio_df,temp_series],axis=1,sort=False)

    task_column_name = 'trans_ratio'
    task_df = trans_ratio_df
    trans_ratio_bin_df = get_bin_interval_df(task_df, Bin_interval_number, task_column_name)
    statistic_info_interval_dic['trans_ratio_bin_df']=trans_ratio_bin_df

    statistic_info_interval_dic_pkl_path=os.path.join(raw_data_dir,"statistic_info_interval_dic.pkl")
    sim_info.store_variable_from_pikle_file(statistic_info_interval_dic_pkl_path,statistic_info_interval_dic)

def generate_sim_statistic_info(sim_info,raw_data_dir,sim_data_dir,parallel,kernel_number,simSeqTime,step):

    max_bin = get_max_bin(sim_info,step)
    statistic_info_interval_dic_pkl_path = os.path.join(raw_data_dir, "statistic_info_interval_dic.pkl")
    statistic_info_interval_dic = sim_info.load_variable_from_pikle_file(statistic_info_interval_dic_pkl_path)

    sim_contacts_total_CisDis_Trans_ratio_df = pd.DataFrame(index=sim_info.cell_name_list)
    task_column_name = 'total_count'
    total_count_bin_df = statistic_info_interval_dic['total_count_bin_df']
    sim_contacts_total_CisDis_Trans_ratio_df = interval_sample(total_count_bin_df,
                                                                     sim_contacts_total_CisDis_Trans_ratio_df,
                                                                     task_column_name)

    column_name = 'trans_ratio'
    trans_ratio_bin_df = statistic_info_interval_dic['trans_ratio_bin_df']
    sim_contacts_total_CisDis_Trans_ratio_df = interval_sample(trans_ratio_bin_df,
                                                                    sim_contacts_total_CisDis_Trans_ratio_df,
                                                                    column_name)
    cis_cell_dis_bin_df_dic = statistic_info_interval_dic['cis_cell_dis_bin_df_dic']
    # not parallel
    if(not parallel):
        for one_dis_loci_name, one_dis_loci_bin_df in cis_cell_dis_bin_df_dic.items():
            print(one_dis_loci_name)
            column_name = one_dis_loci_name
            sim_contacts_total_CisDis_Trans_ratio_df = interval_sample(one_dis_loci_bin_df,
                                                                             sim_contacts_total_CisDis_Trans_ratio_df,
                                                                             column_name)
    # parallel
    else:
        result_list = Parallel(n_jobs=kernel_number)(
            delayed(interval_sample_multi_process)(one_dis_loci_bin_df, one_dis_loci_name) for one_dis_loci_name, one_dis_loci_bin_df in cis_cell_dis_bin_df_dic.items())
        result_dic={}
        for one_bin_all_cell_df_And_one_bin_name in result_list:
            result_dic[one_bin_all_cell_df_And_one_bin_name[1]] = \
                one_bin_all_cell_df_And_one_bin_name[0]
        for one_dis_loci_name, one_dis_loci_bin_df in cis_cell_dis_bin_df_dic.items():
            sample_temp_series=result_dic[one_dis_loci_name]
            sim_contacts_total_CisDis_Trans_ratio_df=interval_sample_combine(sample_temp_series, sim_contacts_total_CisDis_Trans_ratio_df)
    sim_contacts_total_CisDis_Trans_ratio_df_pkl_path=os.path.join(sim_data_dir,"sim_contacts_total_CisDis_Trans_ratio_df.pkl")
    sim_info.store_variable_from_pikle_file(sim_contacts_total_CisDis_Trans_ratio_df_pkl_path,sim_contacts_total_CisDis_Trans_ratio_df)

    number_index_name = []
    number_index_name.append('total_contacts')
    number_index_name.append('trans_contacts')
    for i in range(max_bin + 1):
        number_index_name.append(i)
    sim_contacts_total_CisDis_Trans_number_df = pd.DataFrame(index=number_index_name)

    for i, cell_name in enumerate(sim_contacts_total_CisDis_Trans_ratio_df.index.values):
        temp_total_contacts = sim_contacts_total_CisDis_Trans_ratio_df.loc[cell_name]['total_count']*simSeqTime
        temp_trans_ratio = sim_contacts_total_CisDis_Trans_ratio_df.loc[cell_name]['trans_ratio']
        temp_trans_contacts = int(temp_total_contacts * temp_trans_ratio)
        temp_trans_contacts_series = pd.Series()

        temp_trans_contacts_series.loc['total_contacts'] = temp_total_contacts
        temp_trans_contacts_series.loc['trans_contacts'] = temp_trans_contacts

        temp_cis_ratio = 1 - temp_trans_ratio

        temp_CisDis_series = sim_contacts_total_CisDis_Trans_ratio_df.loc[cell_name][2:]
        if (temp_CisDis_series.sum() != 0):
            temp_CisDis_series_norm = (temp_CisDis_series / temp_CisDis_series.sum()) * temp_cis_ratio
        else:
            temp_CisDis_series_norm = temp_CisDis_series
        temp_CisDis_series_contacts = (temp_CisDis_series_norm * temp_total_contacts)

        temp_contacts_series = pd.concat([temp_trans_contacts_series, temp_CisDis_series_contacts], axis=0, sort=False)
        temp_contacts_series = temp_contacts_series.astype(int)
        temp_contacts_series.name = cell_name
        sim_contacts_total_CisDis_Trans_number_df = pd.concat(
            [sim_contacts_total_CisDis_Trans_number_df, temp_contacts_series], axis=1, sort=False)
    sim_contacts_total_CisDis_Trans_number_df_T = sim_contacts_total_CisDis_Trans_number_df.T
    sim_contacts_total_CisDis_Trans_number_df_T_pkl_path=os.path.join(sim_data_dir,"sim_contacts_total_CisDis_Trans_number_df_T.pkl")
    sim_info.store_variable_from_pikle_file(sim_contacts_total_CisDis_Trans_number_df_T_pkl_path,sim_contacts_total_CisDis_Trans_number_df_T)

def generate_sim_statistic_info_designating_each_cell_fragment_number(sim_info,raw_data_dir,sim_data_dir,parallel,kernel_number,each_cell_fragment_number_dic,step):

    max_bin = get_max_bin(sim_info,step)
    statistic_info_interval_dic_pkl_path = os.path.join(raw_data_dir, "statistic_info_interval_dic.pkl")
    statistic_info_interval_dic = sim_info.load_variable_from_pikle_file(statistic_info_interval_dic_pkl_path)

    sim_contacts_total_CisDis_Trans_ratio_df = pd.DataFrame(index=sim_info.cell_name_list)
    task_column_name = 'total_count'
    total_count_bin_df = statistic_info_interval_dic['total_count_bin_df']
    sim_contacts_total_CisDis_Trans_ratio_df = interval_sample(total_count_bin_df,
                                                                     sim_contacts_total_CisDis_Trans_ratio_df,
                                                                     task_column_name)

    column_name = 'trans_ratio'
    trans_ratio_bin_df = statistic_info_interval_dic['trans_ratio_bin_df']
    sim_contacts_total_CisDis_Trans_ratio_df = interval_sample(trans_ratio_bin_df,
                                                                    sim_contacts_total_CisDis_Trans_ratio_df,
                                                                    column_name)
    cis_cell_dis_bin_df_dic = statistic_info_interval_dic['cis_cell_dis_bin_df_dic']
    # not parallel
    if(not parallel):
        for one_dis_loci_name, one_dis_loci_bin_df in cis_cell_dis_bin_df_dic.items():
            # print(one_dis_loci_name)
            column_name = one_dis_loci_name
            sim_contacts_total_CisDis_Trans_ratio_df = interval_sample(one_dis_loci_bin_df,
                                                                             sim_contacts_total_CisDis_Trans_ratio_df,
                                                                             column_name)
    # parallel
    else:
        result_list = Parallel(n_jobs=kernel_number)(
            delayed(interval_sample_multi_process)(one_dis_loci_bin_df, one_dis_loci_name) for one_dis_loci_name, one_dis_loci_bin_df in cis_cell_dis_bin_df_dic.items())
        result_dic={}
        for one_bin_all_cell_df_And_one_bin_name in result_list:
            result_dic[one_bin_all_cell_df_And_one_bin_name[1]] = \
                one_bin_all_cell_df_And_one_bin_name[0]
        for one_dis_loci_name, one_dis_loci_bin_df in cis_cell_dis_bin_df_dic.items():
            sample_temp_series=result_dic[one_dis_loci_name]
            sim_contacts_total_CisDis_Trans_ratio_df=interval_sample_combine(sample_temp_series, sim_contacts_total_CisDis_Trans_ratio_df)
    sim_contacts_total_CisDis_Trans_ratio_df_pkl_path=os.path.join(sim_data_dir,"sim_contacts_total_CisDis_Trans_ratio_df.pkl")
    sim_info.store_variable_from_pikle_file(sim_contacts_total_CisDis_Trans_ratio_df_pkl_path,sim_contacts_total_CisDis_Trans_ratio_df)

    number_index_name = []
    number_index_name.append('total_contacts')
    number_index_name.append('trans_contacts')
    for i in range(max_bin + 1):
        number_index_name.append(i)
    sim_contacts_total_CisDis_Trans_number_df = pd.DataFrame(index=number_index_name)

    for i, cell_name in enumerate(sim_contacts_total_CisDis_Trans_ratio_df.index.values):
        # temp_total_contacts = sim_contacts_total_CisDis_Trans_ratio_df.loc[cell_name]['total_count']*simSeqTime
        temp_total_contacts = each_cell_fragment_number_dic[cell_name]
        temp_trans_ratio = sim_contacts_total_CisDis_Trans_ratio_df.loc[cell_name]['trans_ratio']
        temp_trans_contacts = int(temp_total_contacts * temp_trans_ratio)
        temp_trans_contacts_series = pd.Series()

        temp_trans_contacts_series.loc['total_contacts'] = temp_total_contacts
        temp_trans_contacts_series.loc['trans_contacts'] = temp_trans_contacts

        temp_cis_ratio = 1 - temp_trans_ratio

        temp_CisDis_series = sim_contacts_total_CisDis_Trans_ratio_df.loc[cell_name][2:]
        if (temp_CisDis_series.sum() != 0):
            temp_CisDis_series_norm = (temp_CisDis_series / temp_CisDis_series.sum()) * temp_cis_ratio
        else:
            temp_CisDis_series_norm = temp_CisDis_series
        temp_CisDis_series_contacts = (temp_CisDis_series_norm * temp_total_contacts)
        temp_CisDis_series_contacts=temp_CisDis_series_contacts.astype(int)
        temp_trans_contacts_series.loc['trans_contacts'] =temp_total_contacts-temp_CisDis_series_contacts.sum()
        # print('trans_contacts_number',temp_trans_contacts_series.loc['trans_contacts'])
        # print('temp_CisDis_series_contacts', temp_CisDis_series_contacts.sum())
        # print(temp_total_contacts)
        temp_contacts_series = pd.concat([temp_trans_contacts_series, temp_CisDis_series_contacts], axis=0, sort=False)
        temp_contacts_series = temp_contacts_series.astype(int)
        temp_contacts_series.name = cell_name
        sim_contacts_total_CisDis_Trans_number_df = pd.concat(
            [sim_contacts_total_CisDis_Trans_number_df, temp_contacts_series], axis=1, sort=False)
    sim_contacts_total_CisDis_Trans_number_df_T = sim_contacts_total_CisDis_Trans_number_df.T
    sim_contacts_total_CisDis_Trans_number_df_T_pkl_path=os.path.join(sim_data_dir,"sim_contacts_total_CisDis_Trans_number_df_T.pkl")
    sim_info.store_variable_from_pikle_file(sim_contacts_total_CisDis_Trans_number_df_T_pkl_path,sim_contacts_total_CisDis_Trans_number_df_T)
def generate_sim_statistic_info_not_designating_each_cell_fragment_number(sim_info,raw_data_dir,sim_data_dir,parallel,kernel_number,each_cell_seqDepth_time_dic,step):

    max_bin = get_max_bin(sim_info,step)
    statistic_info_interval_dic_pkl_path = os.path.join(raw_data_dir, "statistic_info_interval_dic.pkl")
    statistic_info_interval_dic = sim_info.load_variable_from_pikle_file(statistic_info_interval_dic_pkl_path)

    sim_contacts_total_CisDis_Trans_ratio_df = pd.DataFrame(index=sim_info.cell_name_list)
    task_column_name = 'total_count'
    total_count_bin_df = statistic_info_interval_dic['total_count_bin_df']
    sim_contacts_total_CisDis_Trans_ratio_df = interval_sample(total_count_bin_df,
                                                                     sim_contacts_total_CisDis_Trans_ratio_df,
                                                                     task_column_name)

    column_name = 'trans_ratio'
    trans_ratio_bin_df = statistic_info_interval_dic['trans_ratio_bin_df']
    sim_contacts_total_CisDis_Trans_ratio_df = interval_sample(trans_ratio_bin_df,
                                                                    sim_contacts_total_CisDis_Trans_ratio_df,
                                                                    column_name)
    cis_cell_dis_bin_df_dic = statistic_info_interval_dic['cis_cell_dis_bin_df_dic']
    # not parallel
    if(not parallel):
        for one_dis_loci_name, one_dis_loci_bin_df in cis_cell_dis_bin_df_dic.items():
            print(one_dis_loci_name)
            column_name = one_dis_loci_name
            sim_contacts_total_CisDis_Trans_ratio_df = interval_sample(one_dis_loci_bin_df,
                                                                             sim_contacts_total_CisDis_Trans_ratio_df,
                                                                             column_name)
    # parallel
    else:
        result_list = Parallel(n_jobs=kernel_number)(
            delayed(interval_sample_multi_process)(one_dis_loci_bin_df, one_dis_loci_name) for one_dis_loci_name, one_dis_loci_bin_df in cis_cell_dis_bin_df_dic.items())
        result_dic={}
        for one_bin_all_cell_df_And_one_bin_name in result_list:
            result_dic[one_bin_all_cell_df_And_one_bin_name[1]] = \
                one_bin_all_cell_df_And_one_bin_name[0]
        for one_dis_loci_name, one_dis_loci_bin_df in cis_cell_dis_bin_df_dic.items():
            sample_temp_series=result_dic[one_dis_loci_name]
            sim_contacts_total_CisDis_Trans_ratio_df=interval_sample_combine(sample_temp_series, sim_contacts_total_CisDis_Trans_ratio_df)
    sim_contacts_total_CisDis_Trans_ratio_df_pkl_path=os.path.join(sim_data_dir,"sim_contacts_total_CisDis_Trans_ratio_df.pkl")
    sim_info.store_variable_from_pikle_file(sim_contacts_total_CisDis_Trans_ratio_df_pkl_path,sim_contacts_total_CisDis_Trans_ratio_df)

    number_index_name = []
    number_index_name.append('total_contacts')
    number_index_name.append('trans_contacts')
    for i in range(max_bin + 1):
        number_index_name.append(i)
    sim_contacts_total_CisDis_Trans_number_df = pd.DataFrame(index=number_index_name)

    for i, cell_name in enumerate(sim_contacts_total_CisDis_Trans_ratio_df.index.values):
        # temp_total_contacts = sim_contacts_total_CisDis_Trans_ratio_df.loc[cell_name]['total_count']*simSeqTime
        temp_total_contacts = sim_contacts_total_CisDis_Trans_ratio_df.loc[cell_name]['total_count'] * each_cell_seqDepth_time_dic[cell_name]

        temp_trans_ratio = sim_contacts_total_CisDis_Trans_ratio_df.loc[cell_name]['trans_ratio']
        temp_trans_contacts = int(temp_total_contacts * temp_trans_ratio)
        temp_trans_contacts_series = pd.Series()

        temp_trans_contacts_series.loc['total_contacts'] = temp_total_contacts
        temp_trans_contacts_series.loc['trans_contacts'] = temp_trans_contacts

        temp_cis_ratio = 1 - temp_trans_ratio

        temp_CisDis_series = sim_contacts_total_CisDis_Trans_ratio_df.loc[cell_name][2:]
        if (temp_CisDis_series.sum() != 0):
            temp_CisDis_series_norm = (temp_CisDis_series / temp_CisDis_series.sum()) * temp_cis_ratio
        else:
            temp_CisDis_series_norm = temp_CisDis_series
        temp_CisDis_series_contacts = (temp_CisDis_series_norm * temp_total_contacts)

        temp_contacts_series = pd.concat([temp_trans_contacts_series, temp_CisDis_series_contacts], axis=0, sort=False)
        temp_contacts_series = temp_contacts_series.astype(int)
        temp_contacts_series.name = cell_name
        sim_contacts_total_CisDis_Trans_number_df = pd.concat(
            [sim_contacts_total_CisDis_Trans_number_df, temp_contacts_series], axis=1, sort=False)
    sim_contacts_total_CisDis_Trans_number_df_T = sim_contacts_total_CisDis_Trans_number_df.T
    sim_contacts_total_CisDis_Trans_number_df_T_pkl_path=os.path.join(sim_data_dir,"sim_contacts_total_CisDis_Trans_number_df_T.pkl")
    sim_info.store_variable_from_pikle_file(sim_contacts_total_CisDis_Trans_number_df_T_pkl_path,sim_contacts_total_CisDis_Trans_number_df_T)



def combine_genomic_distance_stratified_cis_trans_for_one_cell_phased(sim_info, raw_data_dir, target_cell_name, cell_name_list,
                                                               each_cell_prop, combine_number, max_bin):
    combine_cell_index_origin = np.array([i for i in range(-(combine_number // 2), combine_number // 2 + 1)])
    cell_number = len(cell_name_list)

    cell_index = cell_name_list.index(target_cell_name)
    combine_cell_index = (combine_cell_index_origin + cell_index) % cell_number
    print("original combine_cell_index", combine_cell_index)
    print("original each_cell_prop", each_cell_prop, each_cell_prop.sum())
    combine_cell_index_filter = []
    each_cell_prop_filter = []
    target_cell_stage_index = sim_info.get_index_of_CRICLET_12_stage_by_cell_index(cell_index)
    for cell_sample_prop_index, one_cell_index in enumerate(combine_cell_index):
        one_cell_stage_index = sim_info.get_index_of_CRICLET_12_stage_by_cell_index(one_cell_index)
        if (target_cell_stage_index == one_cell_stage_index):
            combine_cell_index_filter.append(one_cell_index)
            each_cell_prop_filter.append(each_cell_prop[cell_sample_prop_index])
    combine_cell_index = np.array(combine_cell_index_filter)
    each_cell_prop = np.array(each_cell_prop_filter)
    each_cell_prop = each_cell_prop / np.sum(each_cell_prop)
    print("After filtering the adjacent cells which belong to the other CRICLET stage,")
    print("combine_cell_index", combine_cell_index)
    print("each_cell_prop", each_cell_prop, each_cell_prop.sum())
    all_cell_cis_distance_stratified_df_dic_dic = {}
    all_cell_trans_dic = {}
    for cell_index in combine_cell_index:
        cell_name = cell_name_list[cell_index]
        cis_distance_stratified_dic_pkl_path = os.path.join(raw_data_dir, cell_name, "intermediate_file",
                                                            "cis_distance_stratified_dic.pkl")
        cis_distance_stratified_dic = sim_info.load_variable_from_pikle_file(cis_distance_stratified_dic_pkl_path)
        all_cell_cis_distance_stratified_df_dic_dic[cell_name] = cis_distance_stratified_dic

        trans_df_pkl_path = os.path.join(raw_data_dir, cell_name, "intermediate_file",
                                         "trans_df.pkl")
        trans_df = sim_info.load_variable_from_pikle_file(trans_df_pkl_path)
        all_cell_trans_dic[cell_name] = trans_df
    all_distance_combine_dic = {}
    start = datetime.datetime.now()
    print("start to combine", target_cell_name)
    print("process cis distance")
    for each_distance in np.arange(0, int(max_bin) + 1):
        # print(int(each_distance))

        each_distance_combine_list = []
        for cell_sample_prop_index, one_cell_index in enumerate(combine_cell_index):
            one_cell_name = cell_name_list[one_cell_index]
            # print(cell_sample_prop_index,one_cell_name)
            one_cell_one_distance_cis_df = all_cell_cis_distance_stratified_df_dic_dic[one_cell_name][
                int(each_distance)]
            if (len(one_cell_one_distance_cis_df) != 0):
                one_cell_one_distance_cis_df['prop'] = one_cell_one_distance_cis_df['prop'] * each_cell_prop[
                    cell_sample_prop_index]
            each_distance_combine_list.append(one_cell_one_distance_cis_df)
        each_distance_combine_df = pd.concat(each_distance_combine_list, axis=0, sort=False)
        if (len(each_distance_combine_df) > 0):
            no_dup_each_distance_combine_df = \
            each_distance_combine_df.groupby(['chr1', 'chr2', 'pos1', 'pos2'], as_index=False, sort=False)['prop'].sum()
            no_dup_each_distance_combine_df.reset_index(inplace=True, drop=True)
            no_dup_each_distance_combine_df['count'] = 1
            no_dup_each_distance_combine_df['cell_name'] = target_cell_name
            no_dup_each_distance_combine_df['prop'] = no_dup_each_distance_combine_df['prop'] / \
                                                      no_dup_each_distance_combine_df['prop'].sum()
            all_distance_combine_dic[int(each_distance)] = no_dup_each_distance_combine_df
        else:
            all_distance_combine_dic[int(each_distance)] = each_distance_combine_df
    print("process trans")

    all_trans_combine_list = []
    for cell_sample_prop_index, one_cell_index in enumerate(combine_cell_index):
        one_cell_name = cell_name_list[one_cell_index]
        one_cell_trans_df = all_cell_trans_dic[one_cell_name]
        if (len(one_cell_trans_df) != 0):
            one_cell_trans_df['prop'] = one_cell_trans_df['prop'] * each_cell_prop[cell_sample_prop_index]
        all_trans_combine_list.append(one_cell_trans_df)
    all_trans_combine_df = pd.concat(all_trans_combine_list, axis=0, sort=False)
    if (len(all_trans_combine_df) > 0):
        no_dup_all_trans_combine_df = \
        all_trans_combine_df.groupby(['chr1', 'chr2', 'pos1', 'pos2'], as_index=False, sort=False)['prop'].sum()
        no_dup_all_trans_combine_df.reset_index(inplace=True, drop=True)
        no_dup_all_trans_combine_df['count'] = 1
        no_dup_all_trans_combine_df['cell_name'] = target_cell_name
        no_dup_all_trans_combine_df['prop'] = no_dup_all_trans_combine_df['prop'] / no_dup_all_trans_combine_df[
            'prop'].sum()
        all_trans_combine_df = no_dup_all_trans_combine_df
    end = datetime.datetime.now()
    print("end,using ", end - start)
    print("------------------------")
    return all_distance_combine_dic, all_trans_combine_df,all_cell_cis_distance_stratified_df_dic_dic[target_cell_name]

def combine_genomic_distance_stratified_cis_trans_for_one_cell_phased_by_cell_cell_distance(sim_info, raw_data_dir, target_cell_name, cell_name_list,
                                                               each_cell_prop, combine_number, max_bin,cell_cell_distance_df):
    # combine_cell_index_origin = np.array([i for i in range(-(combine_number // 2), combine_number // 2 + 1)])
    # cell_number = len(cell_name_list)
    #
    # cell_index = cell_name_list.index(target_cell_name)
    # combine_cell_index = (combine_cell_index_origin + cell_index) % cell_number
    # print("original combine_cell_index", combine_cell_index)
    # print("original each_cell_prop", each_cell_prop, each_cell_prop.sum())
    # combine_cell_index_filter = []
    # each_cell_prop_filter = []
    # target_cell_stage_index = sim_info.get_index_of_CRICLET_12_stage_by_cell_index(cell_index)
    # for cell_sample_prop_index, one_cell_index in enumerate(combine_cell_index):
    #     one_cell_stage_index = sim_info.get_index_of_CRICLET_12_stage_by_cell_index(one_cell_index)
    #     if (target_cell_stage_index == one_cell_stage_index):
    #         combine_cell_index_filter.append(one_cell_index)
    #         each_cell_prop_filter.append(each_cell_prop[cell_sample_prop_index])
    # combine_cell_index = np.array(combine_cell_index_filter)
    # each_cell_prop = np.array(each_cell_prop_filter)
    # each_cell_prop = each_cell_prop / np.sum(each_cell_prop)
    # print("After filtering the adjacent cells which belong to the other CRICLET stage,")
    # print("combine_cell_index", combine_cell_index)
    # print("each_cell_prop", each_cell_prop, each_cell_prop.sum())
    all_cell_cis_distance_stratified_df_dic_dic = {}
    all_cell_trans_dic = {}
    # for cell_index in combine_cell_index:
    #     cell_name = cell_name_list[cell_index]
    #     cis_distance_stratified_dic_pkl_path = os.path.join(raw_data_dir, cell_name, "intermediate_file",
    #                                                         "cis_distance_stratified_dic.pkl")
    #     cis_distance_stratified_dic = sim_info.load_variable_from_pikle_file(cis_distance_stratified_dic_pkl_path)
    #     all_cell_cis_distance_stratified_df_dic_dic[cell_name] = cis_distance_stratified_dic
    #
    #     trans_df_pkl_path = os.path.join(raw_data_dir, cell_name, "intermediate_file",
    #                                      "trans_df.pkl")
    #     trans_df = sim_info.load_variable_from_pikle_file(trans_df_pkl_path)
    #     all_cell_trans_dic[cell_name] = trans_df
    combine_cell_name_list=get_adj_cell_name_by_cell_cell_distance(cell_cell_distance_df, target_cell_name, combine_number)
    print(target_cell_name,'neighboring cells:',combine_cell_name_list)
    for cell_name in combine_cell_name_list:
        # cell_name = cell_name_list[cell_index]
        cis_distance_stratified_dic_pkl_path = os.path.join(raw_data_dir, cell_name, "intermediate_file",
                                                            "cis_distance_stratified_dic.pkl")
        cis_distance_stratified_dic = sim_info.load_variable_from_pikle_file(cis_distance_stratified_dic_pkl_path)
        all_cell_cis_distance_stratified_df_dic_dic[cell_name] = cis_distance_stratified_dic

        trans_df_pkl_path = os.path.join(raw_data_dir, cell_name, "intermediate_file",
                                         "trans_df.pkl")
        trans_df = sim_info.load_variable_from_pikle_file(trans_df_pkl_path)
        all_cell_trans_dic[cell_name] = trans_df
    all_distance_combine_dic = {}
    start = datetime.datetime.now()
    print("start to combine", target_cell_name)
    print("process cis distance")
    for each_distance in np.arange(0, int(max_bin) + 1):
        # print(int(each_distance))

        each_distance_combine_list = []
        for cell_sample_prop_index, one_cell_name in enumerate(combine_cell_name_list):
            # one_cell_name = cell_name_list[one_cell_index]
            # print(cell_sample_prop_index,one_cell_name)
            one_cell_one_distance_cis_df = all_cell_cis_distance_stratified_df_dic_dic[one_cell_name][
                int(each_distance)]
            if (len(one_cell_one_distance_cis_df) != 0):
                one_cell_one_distance_cis_df['prop'] = one_cell_one_distance_cis_df['prop'] * each_cell_prop[
                    cell_sample_prop_index]
            each_distance_combine_list.append(one_cell_one_distance_cis_df)
        each_distance_combine_df = pd.concat(each_distance_combine_list, axis=0, sort=False)
        if (len(each_distance_combine_df) > 0):
            no_dup_each_distance_combine_df = \
            each_distance_combine_df.groupby(['chr1', 'chr2', 'pos1', 'pos2'], as_index=False, sort=False)['prop'].sum()
            no_dup_each_distance_combine_df.reset_index(inplace=True, drop=True)
            no_dup_each_distance_combine_df['count'] = 1
            no_dup_each_distance_combine_df['cell_name'] = target_cell_name
            no_dup_each_distance_combine_df['prop'] = no_dup_each_distance_combine_df['prop'] / \
                                                      no_dup_each_distance_combine_df['prop'].sum()
            all_distance_combine_dic[int(each_distance)] = no_dup_each_distance_combine_df
        else:
            all_distance_combine_dic[int(each_distance)] = each_distance_combine_df
    print("process trans")

    all_trans_combine_list = []
    for cell_sample_prop_index, one_cell_name in enumerate(combine_cell_name_list):
        # one_cell_name = cell_name_list[one_cell_index]
        one_cell_trans_df = all_cell_trans_dic[one_cell_name]
        if (len(one_cell_trans_df) != 0):
            one_cell_trans_df['prop'] = one_cell_trans_df['prop'] * each_cell_prop[cell_sample_prop_index]
        all_trans_combine_list.append(one_cell_trans_df)
    all_trans_combine_df = pd.concat(all_trans_combine_list, axis=0, sort=False)
    if (len(all_trans_combine_df) > 0):
        no_dup_all_trans_combine_df = \
        all_trans_combine_df.groupby(['chr1', 'chr2', 'pos1', 'pos2'], as_index=False, sort=False)['prop'].sum()
        no_dup_all_trans_combine_df.reset_index(inplace=True, drop=True)
        no_dup_all_trans_combine_df['count'] = 1
        no_dup_all_trans_combine_df['cell_name'] = target_cell_name
        no_dup_all_trans_combine_df['prop'] = no_dup_all_trans_combine_df['prop'] / no_dup_all_trans_combine_df[
            'prop'].sum()
        all_trans_combine_df = no_dup_all_trans_combine_df
    end = datetime.datetime.now()
    print("end,using ", end - start)
    print("------------------------")
    return all_distance_combine_dic, all_trans_combine_df,all_cell_cis_distance_stratified_df_dic_dic[target_cell_name]


def interval_sample_from_genomic_distance_stratified_cis_df_trans_df_for_one_cell(target_cell_name,target_cell_distance_cis_dic,each_cell_distance_combine_dic,
                                                                                           each_cell_trans_combine_df,sim_contacts_total_CisDis_Trans_number_df_T):
    print("start to genomic distance_stratified sample",target_cell_name)
    start=datetime.datetime.now()

    each_distance_sim_df_list=[]
    for distance,one_distance_target_df in target_cell_distance_cis_dic.items():
        if(len(one_distance_target_df)>0):
            one_distance_combine_df=each_cell_distance_combine_dic[distance]
            one_distance_target_count=sim_contacts_total_CisDis_Trans_number_df_T.loc[target_cell_name][distance]
            # one_distance_target_count=one_distance_target_df.shape[0]
            one_distance_combine_count=one_distance_combine_df.shape[0]

            all_item_number=one_distance_combine_count

            sample_item_number=one_distance_target_count
            sample_item_number=min(all_item_number,sample_item_number)

            propItem=one_distance_combine_df['prop'].values

            sample_index = np.random.choice(all_item_number, sample_item_number, replace=False,
                                                 p=list(propItem))
            sim_df=one_distance_combine_df.iloc[sample_index]
            each_distance_sim_df_list.append(sim_df)

    cell_distance_sim_df=pd.concat(each_distance_sim_df_list,axis=0,sort=False)

    target_cell_trans_count=sim_contacts_total_CisDis_Trans_number_df_T.loc[target_cell_name]['trans_contacts']
    # target_cell_trans_count=target_cell_trans_df.shape[0]
    combine_cell_trans_count=each_cell_trans_combine_df.shape[0]

    all_item_number=combine_cell_trans_count

    sample_item_number=target_cell_trans_count
    sample_item_number = min(all_item_number, sample_item_number)
    propItem=each_cell_trans_combine_df['prop'].values

    sample_index = np.random.choice(all_item_number, sample_item_number, replace=False,
                                         p=list(propItem))
    cell_trans_sim_df=each_cell_trans_combine_df.iloc[sample_index]

    sim_df=pd.concat([cell_distance_sim_df,cell_trans_sim_df],axis=0,sort=False)
    sim_df=sim_df.drop(['prop'],axis=1)
    sim_df.reset_index(inplace=True,drop=True)


    end=datetime.datetime.now()
    print('end, using',end-start)

    return sim_df

def get_adj_cell_name_by_cell_cell_distance(cell_cell_distance_df,cell_name,combine_number):
    one_cell_series = cell_cell_distance_df[cell_name]
    one_cell_sort_series = one_cell_series.sort_values(ascending=True)
    result_list=list(one_cell_sort_series.index)[0:combine_number+1]
    if cell_name not in result_list:
        result_list=[cell_name]+result_list[0:-1]
    return result_list

def one_run_simulation_for_one_cell(sim_info,raw_data_dir,sim_data_dir,cell_name,cell_name_list,each_cell_prop,combine_number,max_bin,sim_contacts_total_CisDis_Trans_number_df_T,replicates_number):

    target_cell_name=cell_name
    each_cell_distance_combine_dic, each_cell_trans_combine_df, target_cell_distance_cis_dic = combine_genomic_distance_stratified_cis_trans_for_one_cell_phased(
        sim_info, raw_data_dir, target_cell_name, cell_name_list,
        each_cell_prop, combine_number, max_bin)
    for replicates_serial_number in range(replicates_number):
        sim_df = interval_sample_from_genomic_distance_stratified_cis_df_trans_df_for_one_cell(target_cell_name,
                                                                                                     target_cell_distance_cis_dic,
                                                                                                     each_cell_distance_combine_dic,
                                                                                                     each_cell_trans_combine_df,
                                                                                                     sim_contacts_total_CisDis_Trans_number_df_T)

        sim_info.write_out_chr_pos_df_with_replicates(sim_df,sim_data_dir,cell_name,replicates_serial_number+1)

def one_run_simulation_for_one_cell_by_cell_cell_distance(sim_info,raw_data_dir,sim_data_dir,cell_name,cell_name_list,each_cell_prop,combine_number,max_bin,sim_contacts_total_CisDis_Trans_number_df_T,replicates_number,cell_cell_distance_df):

    target_cell_name=cell_name
    each_cell_distance_combine_dic, each_cell_trans_combine_df, target_cell_distance_cis_dic = combine_genomic_distance_stratified_cis_trans_for_one_cell_phased_by_cell_cell_distance(
        sim_info, raw_data_dir, target_cell_name, cell_name_list,
        each_cell_prop, combine_number, max_bin,cell_cell_distance_df)
    for replicates_serial_number in range(replicates_number):
        sim_df = interval_sample_from_genomic_distance_stratified_cis_df_trans_df_for_one_cell(target_cell_name,
                                                                                                     target_cell_distance_cis_dic,
                                                                                                     each_cell_distance_combine_dic,
                                                                                                     each_cell_trans_combine_df,
                                                                                                     sim_contacts_total_CisDis_Trans_number_df_T)

        sim_info.write_out_chr_pos_df_with_replicates(sim_df,sim_data_dir,cell_name,replicates_serial_number+1)


def simulation_by_cell_name_list(sim_info,raw_data_dir,sim_data_dir,cell_name_list,combine_number,parallel,kernel_number,each_cell_replicates_number_dic,step):
    sim_contacts_total_CisDis_Trans_number_df_T_pkl_path = os.path.join(sim_data_dir,
                                                                        "sim_contacts_total_CisDis_Trans_number_df_T.pkl")
    sim_contacts_total_CisDis_Trans_number_df_T = sim_info.load_variable_from_pikle_file(
        sim_contacts_total_CisDis_Trans_number_df_T_pkl_path)
    each_cell_prop = np.ones(combine_number + 1)
    each_cell_prop = each_cell_prop / each_cell_prop.sum()


    max_bin=get_max_bin(sim_info,step)

    if(not parallel):
        for cell_name in cell_name_list:
            one_run_simulation_for_one_cell(sim_info, raw_data_dir, sim_data_dir, cell_name, cell_name_list,
                                                  each_cell_prop, combine_number, max_bin,
                                                  sim_contacts_total_CisDis_Trans_number_df_T,each_cell_replicates_number_dic[cell_name])
    else:
        Parallel(n_jobs=kernel_number)(
            delayed(one_run_simulation_for_one_cell)(sim_info, raw_data_dir, sim_data_dir, cell_name, cell_name_list,
                                                     each_cell_prop, combine_number, max_bin,sim_contacts_total_CisDis_Trans_number_df_T,each_cell_replicates_number_dic[cell_name])
            for cell_name in cell_name_list)

def simulation_by_cell_name_list_by_cell_cell_distance(sim_info,raw_data_dir,sim_data_dir,cell_name_list,combine_number,parallel,kernel_number,each_cell_replicates_number_dic,step,cell_cell_distance_df):
    sim_contacts_total_CisDis_Trans_number_df_T_pkl_path = os.path.join(sim_data_dir,
                                                                        "sim_contacts_total_CisDis_Trans_number_df_T.pkl")
    sim_contacts_total_CisDis_Trans_number_df_T = sim_info.load_variable_from_pikle_file(
        sim_contacts_total_CisDis_Trans_number_df_T_pkl_path)
    each_cell_prop = np.ones(combine_number + 1)
    each_cell_prop = each_cell_prop / each_cell_prop.sum()


    max_bin=get_max_bin(sim_info,step)

    if(not parallel):
        for cell_name in cell_name_list:
            one_run_simulation_for_one_cell_by_cell_cell_distance(sim_info, raw_data_dir, sim_data_dir, cell_name, cell_name_list,
                                                  each_cell_prop, combine_number, max_bin,
                                                  sim_contacts_total_CisDis_Trans_number_df_T,each_cell_replicates_number_dic[cell_name],cell_cell_distance_df)
    else:
        Parallel(n_jobs=kernel_number)(
            delayed(one_run_simulation_for_one_cell_by_cell_cell_distance)(sim_info, raw_data_dir, sim_data_dir, cell_name, cell_name_list,
                                                     each_cell_prop, combine_number, max_bin,sim_contacts_total_CisDis_Trans_number_df_T,each_cell_replicates_number_dic[cell_name],cell_cell_distance_df)
            for cell_name in cell_name_list)

class one_run_simulation(object):
    @staticmethod
    def one_run_simulation(sim_info,raw_data_dir,sim_data_dir,cell_name_list,parallel,kernel_number,Bin_interval_number,simSeqTime,combine_number,step,generate_raw_info=True):
        if(generate_raw_info):
            run_generate_statisc_info(sim_info, raw_data_dir, cell_name_list, parallel, kernel_number,step)
            collect_generate_and_store_interval_bin_for_raw(sim_info, raw_data_dir, cell_name_list, kernel_number, parallel,step,
                                                            Bin_interval_number=Bin_interval_number)
        sim_data_dir=os.path.join(sim_data_dir,"SeqDepthTime_"+str(simSeqTime))
        sim_info.recursive_mkdir(sim_data_dir)
        generate_sim_statistic_info(sim_info, raw_data_dir, sim_data_dir, parallel, kernel_number, simSeqTime,step)
        simulation_by_cell_name_list(sim_info, raw_data_dir, sim_data_dir, cell_name_list, combine_number, parallel,
                                     kernel_number,step)

    @staticmethod
    def one_run_simulation_designating_each_cell_fragment_number(sim_info,raw_data_dir,sim_data_dir,cell_name_list,parallel,kernel_number,Bin_interval_number,each_cell_fragment_number_dic,each_cell_replicates_number_dic,combine_number,step,cell_cell_distance_df,generate_raw_info=True):
        if(generate_raw_info):
            run_generate_statisc_info(sim_info, raw_data_dir, cell_name_list, parallel, kernel_number,step)
            collect_generate_and_store_interval_bin_for_raw(sim_info, raw_data_dir, cell_name_list, kernel_number, parallel,step,
                                                            Bin_interval_number=Bin_interval_number)
        # sim_data_dir=os.path.join(sim_data_dir,"SeqDepthTime_"+str(simSeqTime))
        sim_info.recursive_mkdir(sim_data_dir)
        generate_sim_statistic_info_designating_each_cell_fragment_number(sim_info, raw_data_dir, sim_data_dir, parallel, kernel_number, each_cell_fragment_number_dic,step)
        simulation_by_cell_name_list_by_cell_cell_distance(sim_info, raw_data_dir, sim_data_dir, cell_name_list, combine_number, parallel,
                                     kernel_number,each_cell_replicates_number_dic,step,cell_cell_distance_df)
    @staticmethod
    def one_run_simulation_not_designating_each_cell_fragment_number(sim_info,raw_data_dir,sim_data_dir,cell_name_list,parallel,kernel_number,Bin_interval_number,each_cell_seqDepth_time_dic,each_cell_replicates_number_dic,combine_number,step,cell_cell_distance_df,generate_raw_info=True):
        if(generate_raw_info):
            run_generate_statisc_info(sim_info, raw_data_dir, cell_name_list, parallel, kernel_number,step)
            collect_generate_and_store_interval_bin_for_raw(sim_info, raw_data_dir, cell_name_list, kernel_number, parallel,step,
                                                            Bin_interval_number=Bin_interval_number)
        # sim_data_dir=os.path.join(sim_data_dir,"SeqDepthTime_"+str(simSeqTime))
        sim_info.recursive_mkdir(sim_data_dir)
        generate_sim_statistic_info_not_designating_each_cell_fragment_number(sim_info, raw_data_dir, sim_data_dir, parallel, kernel_number, each_cell_seqDepth_time_dic,step)
        simulation_by_cell_name_list_by_cell_cell_distance(sim_info, raw_data_dir, sim_data_dir, cell_name_list, combine_number, parallel,
                                     kernel_number,each_cell_replicates_number_dic,step,cell_cell_distance_df)

def store_all_cell_cis_df_dic_AND_all_cell_cis_distance_stratified_df_dic_dic_AND_all_cell_trans_dic(sim_info,all_cell_cis_df_dic,all_cell_cis_distance_stratified_df_dic_dic,all_cell_trans_dic):
    Cis_CisDistanceStratified_Trans_list=[]
    Cis_CisDistanceStratified_Trans_list.append(all_cell_cis_df_dic)
    Cis_CisDistanceStratified_Trans_list.append(all_cell_cis_distance_stratified_df_dic_dic)
    Cis_CisDistanceStratified_Trans_list.append(all_cell_trans_dic)

    Cis_CisDistanceStratified_Trans_list_path=os.path.join(sim_info.raw_single_cell_dir,"Cis_CisDistanceStratified_Trans_list.pkl")
    sim_info.store_variable_from_pikle_file(Cis_CisDistanceStratified_Trans_list_path,Cis_CisDistanceStratified_Trans_list)


def load_all_cell_cis_df_dic_AND_all_cell_cis_distance_stratified_df_dic_dic_AND_all_cell_trans_dic(sim_info):
    Cis_CisDistanceStratified_Trans_list_path=os.path.join(sim_info.raw_single_cell_dir,"Cis_CisDistanceStratified_Trans_list.pkl")
    Cis_CisDistanceStratified_Trans_list=sim_info.load_variable_from_pikle_file(Cis_CisDistanceStratified_Trans_list_path)
    all_cell_cis_df_dic=Cis_CisDistanceStratified_Trans_list[0]
    all_cell_cis_distance_stratified_df_dic_dic=Cis_CisDistanceStratified_Trans_list[1]
    all_cell_trans_dic=Cis_CisDistanceStratified_Trans_list[2]
    return all_cell_cis_df_dic, all_cell_cis_distance_stratified_df_dic_dic, all_cell_trans_dic

def combine_genomic_distance_stratified_cis_trans_for_one_cell_copy(sim_info,all_cell_cis_distance_stratified_df_dic_dic,all_cell_trans_dic,
                                                               target_cell_name,each_cell_prop,combine_number,max_bin):

    combine_cell_index_origin = np.array([i for i in range(-(combine_number // 2), combine_number // 2 + 1)])
    cell_number = len(sim_info.cell_phase_list[0])

    cell_index = sim_info.cell_phase_list[0].index(target_cell_name)
    combine_cell_index = (combine_cell_index_origin + cell_index) % cell_number
    print("original combine_cell_index",combine_cell_index)
    print("original each_cell_prop",each_cell_prop,each_cell_prop.sum())
    combine_cell_index_filter=[]
    each_cell_prop_filter=[]
    target_cell_stage_index=sim_info.get_index_of_CRICLET_12_stage_by_cell_index(cell_index)
    for cell_sample_prop_index,one_cell_index in enumerate(combine_cell_index):
        one_cell_stage_index=sim_info.get_index_of_CRICLET_12_stage_by_cell_index(one_cell_index)
        if(target_cell_stage_index==one_cell_stage_index):
            combine_cell_index_filter.append(one_cell_index)
            each_cell_prop_filter.append(each_cell_prop[cell_sample_prop_index])
    combine_cell_index=np.array(combine_cell_index_filter)
    each_cell_prop=np.array(each_cell_prop_filter)
    each_cell_prop=each_cell_prop/np.sum(each_cell_prop)
    print("After filtering the adjacent cells which belong to the other CRICLET stage,")
    print("combine_cell_index",combine_cell_index)
    print("each_cell_prop",each_cell_prop,each_cell_prop.sum())

    all_distance_combine_dic={}
    start=datetime.datetime.now()
    print("start to combine",target_cell_name)
    print("process cis distance")
    for each_distance in np.arange(0,int(max_bin)+1):
        #print(int(each_distance))

        each_distance_combine_list=[]
        for cell_sample_prop_index,one_cell_index in enumerate(combine_cell_index):
            one_cell_name = sim_info.cell_phase_list[0][one_cell_index]
            # print(cell_sample_prop_index,one_cell_name)
            one_cell_one_distance_cis_df = copy.deepcopy(all_cell_cis_distance_stratified_df_dic_dic[one_cell_name][int(each_distance)])
            if(len(one_cell_one_distance_cis_df)!=0):
                one_cell_one_distance_cis_df['prop']=one_cell_one_distance_cis_df['prop']*each_cell_prop[cell_sample_prop_index]
            each_distance_combine_list.append(one_cell_one_distance_cis_df)
        each_distance_combine_df=pd.concat(each_distance_combine_list,axis=0,sort=False)
        if(len(each_distance_combine_df)>0):
            no_dup_each_distance_combine_df= each_distance_combine_df.groupby(['chr1','chr2','pos1','pos2'],as_index=False,sort=False)['prop'].sum()
            no_dup_each_distance_combine_df.reset_index(inplace=True,drop=True)
            no_dup_each_distance_combine_df['count']=1
            no_dup_each_distance_combine_df['cell_name']=target_cell_name
            no_dup_each_distance_combine_df['prop']=no_dup_each_distance_combine_df['prop']/no_dup_each_distance_combine_df['prop'].sum()
            all_distance_combine_dic[int(each_distance)]=no_dup_each_distance_combine_df
        else:
            all_distance_combine_dic[int(each_distance)]=each_distance_combine_df
    print("process trans")

    all_trans_combine_list=[]
    for cell_sample_prop_index,one_cell_index in enumerate(combine_cell_index):
        one_cell_name = sim_info.cell_phase_list[0][one_cell_index]
        one_cell_trans_df=copy.deepcopy(all_cell_trans_dic[one_cell_name])
        if(len(one_cell_trans_df)!=0):
            one_cell_trans_df['prop']=one_cell_trans_df['prop']*each_cell_prop[cell_sample_prop_index]
        all_trans_combine_list.append(one_cell_trans_df)
    all_trans_combine_df=pd.concat(all_trans_combine_list,axis=0,sort=False)
    if(len(all_trans_combine_df)>0):
        no_dup_all_trans_combine_df= all_trans_combine_df.groupby(['chr1','chr2','pos1','pos2'],as_index=False,sort=False)['prop'].sum()
        no_dup_all_trans_combine_df.reset_index(inplace=True,drop=True)
        no_dup_all_trans_combine_df['count']=1
        no_dup_all_trans_combine_df['cell_name']=target_cell_name
        no_dup_all_trans_combine_df['prop'] = no_dup_all_trans_combine_df['prop'] / no_dup_all_trans_combine_df['prop'].sum()
        all_trans_combine_df=no_dup_all_trans_combine_df
    end=datetime.datetime.now()
    print("end,using ",end-start)
    print("------------------------")
    return all_distance_combine_dic,all_trans_combine_df
def combine_genomic_distance_stratified_cis_trans_for_one_cell_not_copy(sim_info,all_cell_cis_distance_stratified_df_dic_dic,all_cell_trans_dic,
                                                               target_cell_name,each_cell_prop,combine_number,max_bin):

    combine_cell_index_origin = np.array([i for i in range(-(combine_number // 2), combine_number // 2 + 1)])
    cell_number = len(sim_info.cell_name_list)

    cell_index = sim_info.cell_phase_list[0].index(target_cell_name)
    combine_cell_index = (combine_cell_index_origin + cell_index) % cell_number

    all_distance_combine_dic={}
    start=datetime.datetime.now()
    print("start to combine",target_cell_name)
    print("process cis distance")
    for each_distance in np.arange(0,int(max_bin)+1):
        #print(int(each_distance))

        each_distance_combine_list=[]
        for cell_sample_prop_index,one_cell_index in enumerate(combine_cell_index):
            one_cell_name = sim_info.cell_phase_list[0][one_cell_index]
            one_cell_one_distance_cis_df = all_cell_cis_distance_stratified_df_dic_dic[one_cell_name][int(each_distance)]
            if(len(one_cell_one_distance_cis_df)!=0):
                one_cell_one_distance_cis_df['temp_prop']=one_cell_one_distance_cis_df['prop']*each_cell_prop[cell_sample_prop_index]
            each_distance_combine_list.append(one_cell_one_distance_cis_df)
        each_distance_combine_df=pd.concat(each_distance_combine_list,axis=0,sort=False)
        if(len(each_distance_combine_df)>0):
            each_distance_combine_df.reset_index(inplace=True, drop=True)

            each_distance_combine_df['prop']=each_distance_combine_df['temp_prop']

        if(len(each_distance_combine_df)>0):
            no_dup_each_distance_combine_df= each_distance_combine_df.groupby(['chr1','chr2','pos1','pos2'],as_index=False,sort=False)['prop'].sum()
            no_dup_each_distance_combine_df.reset_index(inplace=True,drop=True)
            no_dup_each_distance_combine_df['count']=1
            no_dup_each_distance_combine_df['cell_name']=target_cell_name
            no_dup_each_distance_combine_df['prop']=no_dup_each_distance_combine_df['prop']/no_dup_each_distance_combine_df['prop'].sum()
            all_distance_combine_dic[int(each_distance)]=no_dup_each_distance_combine_df
        else:
            all_distance_combine_dic[int(each_distance)]=each_distance_combine_df
    print("process trans")

    all_trans_combine_list=[]
    for cell_sample_prop_index,one_cell_index in enumerate(combine_cell_index):
        one_cell_name = sim_info.cell_phase_list[0][one_cell_index]
        one_cell_trans_df=all_cell_trans_dic[one_cell_name]
        if(len(one_cell_trans_df)!=0):
            one_cell_trans_df['temp_prop']=one_cell_trans_df['prop']*each_cell_prop[cell_sample_prop_index]
        all_trans_combine_list.append(one_cell_trans_df)

    all_trans_combine_df=pd.concat(all_trans_combine_list,axis=0,sort=False)
    if(len(all_trans_combine_df)>0):
        all_trans_combine_df.reset_index(inplace=True, drop=True)
        all_trans_combine_df['prop'] = all_trans_combine_df['temp_prop']


    if(len(all_trans_combine_df)>0):
        no_dup_all_trans_combine_df= all_trans_combine_df.groupby(['chr1','chr2','pos1','pos2'],as_index=False,sort=False)['prop'].sum()
        no_dup_all_trans_combine_df.reset_index(inplace=True,drop=True)
        no_dup_all_trans_combine_df['count']=1
        no_dup_all_trans_combine_df['cell_name']=target_cell_name
        no_dup_all_trans_combine_df['prop'] = no_dup_all_trans_combine_df['prop'] / no_dup_all_trans_combine_df['prop'].sum()
        all_trans_combine_df=no_dup_all_trans_combine_df
    end=datetime.datetime.now()
    print("end,using ",end-start)
    print("------------------------")
    return all_distance_combine_dic,all_trans_combine_df


def interval_sample_from_genomic_distance_stratified_cis_df_trans_df_for_one_cell_accord_sample_time(target_cell_name,partial_10_cell_cis_distance_stratified_df_dic_dic,
                                                                                           partial_10_cell_trans_dic,each_cell_distance_combine_dic,
                                                                                           each_cell_trans_combine_df,sample_time,sim_contacts_total_CisDis_Trans_number_df_T):
    print("start to genomic distance_stratified sample",target_cell_name)
    start=datetime.datetime.now()
    sim_df_list=[]
    print("all sample time",sample_time)
    for one_sample_time in range(sample_time):
        print("start",one_sample_time+1,"th sampling")
        target_cell_distance_cis_dic=partial_10_cell_cis_distance_stratified_df_dic_dic[target_cell_name]
        target_cell_trans_df=partial_10_cell_trans_dic[target_cell_name]
        each_distance_sim_df_list=[]
        for distance,one_distance_target_df in target_cell_distance_cis_dic.items():
            if(len(one_distance_target_df)>0):
                one_distance_combine_df=each_cell_distance_combine_dic[distance]
                one_distance_target_count=sim_contacts_total_CisDis_Trans_number_df_T.loc[target_cell_name][distance]
                # one_distance_target_count=one_distance_target_df.shape[0]
                one_distance_combine_count=one_distance_combine_df.shape[0]

                all_item_number=one_distance_combine_count

                sample_item_number=one_distance_target_count
                sample_item_number=min(all_item_number,sample_item_number)

                propItem=one_distance_combine_df['prop'].values

                sample_index = np.random.choice(all_item_number, sample_item_number, replace=False,
                                                     p=list(propItem))
                sim_df=one_distance_combine_df.iloc[sample_index]
                each_distance_sim_df_list.append(sim_df)

        cell_distance_sim_df=pd.concat(each_distance_sim_df_list,axis=0,sort=False)

        target_cell_trans_count=sim_contacts_total_CisDis_Trans_number_df_T.loc[target_cell_name]['trans_contacts']
        # target_cell_trans_count=target_cell_trans_df.shape[0]
        combine_cell_trans_count=each_cell_trans_combine_df.shape[0]

        all_item_number=combine_cell_trans_count

        sample_item_number=target_cell_trans_count
        sample_item_number = min(all_item_number, sample_item_number)
        propItem=each_cell_trans_combine_df['prop'].values

        sample_index = np.random.choice(all_item_number, sample_item_number, replace=False,
                                             p=list(propItem))
        cell_trans_sim_df=each_cell_trans_combine_df.iloc[sample_index]

        sim_df=pd.concat([cell_distance_sim_df,cell_trans_sim_df],axis=0,sort=False)
        sim_df=sim_df.drop(['prop'],axis=1)
        sim_df.reset_index(inplace=True,drop=True)
        sim_df_list.append(sim_df)
    end=datetime.datetime.now()
    print('end, using',end-start)
    return sim_df_list



def sample_from_genomic_distance_stratified_cis_df_trans_df_for_one_cell_accord_sample_time(target_cell_name,partial_10_cell_cis_distance_stratified_df_dic_dic,
                                                                                           partial_10_cell_trans_dic,each_cell_distance_combine_dic,
                                                                                           each_cell_trans_combine_df,sample_time):
    print("start to genomic distance_stratified sample",target_cell_name)
    start=datetime.datetime.now()
    sim_df_list=[]
    print("all sample time",sample_time)
    for one_sample_time in range(sample_time):
        print("start",one_sample_time+1,"th sampling")
        target_cell_distance_cis_dic=partial_10_cell_cis_distance_stratified_df_dic_dic[target_cell_name]
        target_cell_trans_df=partial_10_cell_trans_dic[target_cell_name]
        each_distance_sim_df_list=[]
        for distance,one_distance_target_df in target_cell_distance_cis_dic.items():
            if(len(one_distance_target_df)>0):
                one_distance_combine_df=each_cell_distance_combine_dic[distance]
                one_distance_target_count=one_distance_target_df.shape[0]
                one_distance_combine_count=one_distance_combine_df.shape[0]

                all_item_number=one_distance_combine_count

                sample_item_number=one_distance_target_count

                propItem=one_distance_combine_df['prop'].values

                sample_index = np.random.choice(all_item_number, sample_item_number, replace=False,
                                                     p=list(propItem))
                sim_df=one_distance_combine_df.iloc[sample_index]
                each_distance_sim_df_list.append(sim_df)

        cell_distance_sim_df=pd.concat(each_distance_sim_df_list,axis=0,sort=False)


        target_cell_trans_count=target_cell_trans_df.shape[0]
        combine_cell_trans_count=each_cell_trans_combine_df.shape[0]

        all_item_number=combine_cell_trans_count

        sample_item_number=target_cell_trans_count

        propItem=each_cell_trans_combine_df['prop'].values

        sample_index = np.random.choice(all_item_number, sample_item_number, replace=False,
                                             p=list(propItem))
        cell_trans_sim_df=each_cell_trans_combine_df.iloc[sample_index]

        sim_df=pd.concat([cell_distance_sim_df,cell_trans_sim_df],axis=0,sort=False)
        sim_df=sim_df.drop(['prop'],axis=1)
        sim_df.reset_index(inplace=True,drop=True)
        sim_df_list.append(sim_df)
    end=datetime.datetime.now()
    print('end, using',end-start)
    return sim_df_list




