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
import sys
import shutil
import scipy.sparse as sparse
import math
from scipy.sparse import coo_matrix
from joblib import Parallel, delayed

def get_default_value():
    parameter_name_list = ['python', 'work_dir', 'src', 'cell_base_info', 'raw_data', 'sim_data', 'features',
                           'combineNumber', 'fragment_interaction_number_designating_mode', 'all_cell_seqDepthTime',
                           'each_cell_fragment_interaction_number_designating_mode',
                           'repllicates_number_designating_mode',
                           'all_cell_replicates_number', 'step', 'Bin_interval_number', 'parallel', 'kernel_number']
    default_value_dic = {'python': 'E:\\Users\\scfan\\software\\anaconda3_new\\python.exe',
                         'work_dir': 'E:\\Users\\scfan\\program\\simulation_project\\release_version',
                         'src': 'E:\\Users\\scfan\\program\\simulation_project\\release_version\\src',
                         'cell_base_info': 'E:\\Users\\scfan\\data\\CellCycle\\release_version\\cell_base_info',
                         'raw_data': 'E:\\Users\\scfan\\data\\CellCycle\\release_version\\raw_data',
                         'sim_data': 'E:\\Users\\scfan\\data\\CellCycle\\release_version\\sim_data',
                         'features': 'E:\\Users\\scfan\\data\\CellCycle\\release_version\\features',
                         'combineNumber': '20', 'fragment_interaction_number_designating_mode': 'all_cell',
                         'all_cell_seqDepthTime': '1',
                         'each_cell_fragment_interaction_number_designating_mode': 'sequence_depth_time',
                         'repllicates_number_designating_mode': 'all_cell', 'all_cell_replicates_number': '1',
                         'step': str((math.log(10, 2)) / 80), 'Bin_interval_number': str(200), 'parallel': 'True',
                         'kernel_number': str(24)}
    return parameter_name_list,default_value_dic

def read_in_parameters_df(parmeter_df_dir):
#     parameters_df=pd.read_table("parameters.txt",header=None,index_col=0,comment='#')
    parameters_df=pd.read_table(parmeter_df_dir,header=None,index_col=0,comment='#')
    parameters_df.columns=['value']
    parameters_df.index.name='parameters'
    parameters_df['value']=parameters_df['value'].astype('str')
    return parameters_df
def load_parameters(parameters_df,parameter_name,default_value_dic):
    index_list=list(parameters_df.index)
    if parameter_name not in index_list:
        print("The",parameter_name,"is not designated, the default value", default_value_dic[parameter_name],"will be used.",sep=' ')
        return default_value_dic[parameter_name]
    else:
        print("Loading",parameter_name,": ",parameters_df.loc[parameter_name,'value'])
        return parameters_df.loc[parameter_name,'value']
def read_in_each_cell_SeqDepthTime(cell_base_info_dir):
    each_cell_sequencing_depth_time_file_path=os.path.join(cell_base_info_dir,"each_cell_sequencing_depth_time.txt")
    each_cell_sequencing_depth_df=pd.read_table(each_cell_sequencing_depth_time_file_path,sep='\t',index_col=0,header=0)
    return each_cell_sequencing_depth_df

def read_in_each_cell_fragment_interaction_number(cell_base_info_dir):
    each_cell_fragment_interaction_number_file_path=os.path.join(cell_base_info_dir,"each_cell_fragment_interaction_number.txt")
    each_cell_fragment_interaction_number_df=pd.read_table(each_cell_fragment_interaction_number_file_path,sep='\t',index_col=0,header=0)
    return each_cell_fragment_interaction_number_df
def read_in_each_cell_fragment_replicates_number(cell_base_info_dir):
    each_cell_replicates_number_file_path=os.path.join(cell_base_info_dir,"each_cell_replicates_number.txt")
    each_cell_replicates_number_df=pd.read_table(each_cell_replicates_number_file_path,sep='\t',index_col=0,header=0)
    return each_cell_replicates_number_df


import argparse

def init_argpasre():
    parser = argparse.ArgumentParser(description='pass into parameters')
    #type是要传入的参数的数据类型  help是该参数的提示信息
    parser.add_argument('-p','-par', nargs='?',type=str, help='parameter\'s file name',default='parameters.txt')
    # parser.add_argument('-m', '-merg', nargs='?', type=str, help='merged cells\' name', default=r'data\merge_data\merge_cell_name_list.txt')
    parser.add_argument('-i', '-input', nargs='?', type=str, help='input file', default=r'data\merge_data\chr_pos')
    parser.add_argument('-o', '-output', nargs='?', type=str, help='output file', default=r'data\merge_data\bin_pairs')
    parser.add_argument('-r', '-res', nargs='?', type=int, help='resolution', default=10000)
    return parser

def read_cell_name_list(cell_name_list_path):
    # cell_phase_content
    # 1CDX1_1	phase_1	Diploid_15_16_17_18
    # 1CDX1_2	phase_1	Diploid_15_16_17_18
    # cell_name_list_path=r"E:\Users\scfan\data\CellCycle\release_version\cell_base_info\cell_name_list"

    # cell_name_list_path=os.path.join(cell_base_info_dir,"cell_name_list.txt")
    cell_name_list=[]
    try:
        cell_name_list_df = pd.read_table(cell_name_list_path, header=None)
        cell_name_list = list(cell_name_list_df[0].values)
        return cell_name_list
    except FileNotFoundError:
        print_str = "Can't open file " + cell_name_list_path
        print(print_str)
        return cell_name_list
def read_chr_pos(chr_pos_file):
    # chr_pos_file = os.path.join(raw_data_dir, cell_name, 'chr_pos')
    sc_chr_pos_df = pd.read_table(chr_pos_file, index_col=0)
    return sc_chr_pos_df
if __name__ == '__main__':
    # parmeter_df_dir=r"parameters.txt"
    parser=init_argpasre()
    args = parser.parse_args()
    # print(args.p)
    # print(args.c)
    # parmeter_df_dir = r"parameters_each_cell_fragment_interaction_number.txt"
    parmeter_df_dir=args.p

    # parmeter_df_dir = r"parameters_each_cell_seqDepthTime.txt"
    # parmeter_df_dir = r"parameters_each_cell_replicates_number.txt"
    parameters_df=read_in_parameters_df(parmeter_df_dir)

    # parameter_name_list = ['python', 'work_dir', 'src', 'cell_base_info', 'raw_data', 'sim_data', 'features',
    #                        'combineNumber', 'fragment_interaction_number_designating_mode', 'all_cell_seqDepthTime',
    #                        'each_cell_fragment_interaction_number_designating_mode',
    #                        'repllicates_number_designating_mode',
    #                        'all_cell_replicates_number','step','Bin_interval_number','parallel','kernel_number']
    # default_value_dic = {'python': 'E:\\Users\\scfan\\software\\anaconda3_new\\python.exe',
    #                      'work_dir': 'E:\\Users\\scfan\\program\\simulation_project\\release_version',
    #                      'src': 'E:\\Users\\scfan\\program\\simulation_project\\release_version\\src',
    #                      'cell_base_info': 'E:\\Users\\scfan\\data\\CellCycle\\release_version\\cell_base_info',
    #                      'raw_data': 'E:\\Users\\scfan\\data\\CellCycle\\release_version\\raw_data',
    #                      'sim_data': 'E:\\Users\\scfan\\data\\CellCycle\\release_version\\sim_data',
    #                      'features': 'E:\\Users\\scfan\\data\\CellCycle\\release_version\\features',
    #                      'combineNumber': '20', 'fragment_interaction_number_designating_mode': 'all_cell',
    #                      'all_cell_seqDepthTime': '1',
    #                      'each_cell_fragment_interaction_number_designating_mode': 'sequence_depth_time',
    #                      'repllicates_number_designating_mode': 'all_cell', 'all_cell_replicates_number': '1',
    #                      'step': str((math.log(10, 2)) / 80),'Bin_interval_number':str(200),'parallel':'True',
    #                      'kernel_number':str(24)}
    parameter_name_list,default_value_dic=get_default_value()
    read_in_value_dic = {}
    for parameter_name in parameter_name_list:
        value = load_parameters(parameters_df, parameter_name, default_value_dic)
        read_in_value_dic[parameter_name] = value
    work_dir = read_in_value_dic['work_dir']
    parameters_dir = os.path.join(work_dir, "parameters.txt")
    cell_base_info_dir = read_in_value_dic['cell_base_info']
    raw_data_dir = read_in_value_dic['raw_data']
    sim_data_dir = read_in_value_dic['sim_data']
    features_dir = read_in_value_dic['features']
    python_dir = read_in_value_dic['python']
    combineNumber = int(read_in_value_dic['combineNumber'])
    combine_number=combineNumber
    fragment_interaction_number_designating_mode = read_in_value_dic['fragment_interaction_number_designating_mode']
    all_cell_seqDepthTime = float(read_in_value_dic['all_cell_seqDepthTime'])
    each_cell_fragment_interaction_number_designating_mode = read_in_value_dic[
        'each_cell_fragment_interaction_number_designating_mode']
    repllicates_number_designating_mode = read_in_value_dic['repllicates_number_designating_mode']
    all_cell_replicates_number = read_in_value_dic['all_cell_replicates_number']
    step=float(read_in_value_dic['step'])
    Bin_interval_number=int(read_in_value_dic['Bin_interval_number'])
    parallel=True if read_in_value_dic['parallel']=='True' else False
    kernel_number = int(read_in_value_dic['kernel_number'])
    sys.path.append(work_dir)

    import src.simulation_framework_base_info
    import src.function_prepare_features as features
    import src.function_simulation_method as simulation_method
    importlib.reload(src.simulation_framework_base_info)

    cell_cycle_dir = cell_base_info_dir

    GATC_bed = "GATC.bed"
    GATC_fends = "GATC.fends"

    chr_length = "chr_length"
    fragment_num = "fragment_num"

    GATC_bed_file = os.path.join(cell_cycle_dir, GATC_bed)

    sim_info = src.simulation_framework_base_info.simulation_framework_base_info(cell_cycle_dir, GATC_bed, GATC_fends,
                                                                                 chr_length,
                                                                                 fragment_num)

    sim_info.get_cell_name_list(cell_base_info_dir)
    # sim_info.self_read_GATC_fends(cell_base_info_dir)
    # sim_info.self_get_GATC_bed(cell_base_info_dir)
    sim_info.self_get_chr_length()
    # sim_info.get_cell_stage_info()

    print("Start to convert cells...")


    start=datetime.datetime.now()
    chr_pos_file = args.i
    if (not os.path.exists(chr_pos_file)):
        print(chr_pos_file,"not existing!")
        exit(0)
    else:
        one_df = read_chr_pos(chr_pos_file)
        temp_data=one_df
        resolution = args.r
        temp_data['pos1_bin'] = temp_data['pos1'] // resolution
        temp_data['pos2_bin'] = temp_data['pos2'] // resolution
        new_data = temp_data[['chr1', 'pos1_bin', 'chr2', 'pos2_bin', 'count']]
        new_data.rename(columns={'pos1_bin': 'pos1', 'pos2_bin': 'pos2'}, inplace=True)
        sc_chr_pos = new_data

        chr_pos_1 = sc_chr_pos['chr1'] + '+' + sc_chr_pos['pos1'].astype(str)
        chr_pos_2 = sc_chr_pos['chr2'] + '+' + sc_chr_pos['pos2'].astype(str)
        temp_store_df = pd.DataFrame()
        temp_store_df['chr_pos_1'] = chr_pos_1
        temp_store_df['chr_pos_2'] = chr_pos_2
        temp_store_df['chr_pos_min'] = temp_store_df[['chr_pos_1', 'chr_pos_2']].min(axis=1)
        temp_store_df['chr_pos_max'] = temp_store_df[['chr_pos_1', 'chr_pos_2']].max(axis=1)
        sort_chr_pos_1_df = pd.DataFrame(list(temp_store_df['chr_pos_min'].str.split('+')),
                                         columns=['chr1', 'pos1'])
        sort_chr_pos_2_df = pd.DataFrame(list(temp_store_df['chr_pos_max'].str.split('+')),
                                         columns=['chr2', 'pos2'])
        sort_sc_chr_pos = copy.deepcopy(sc_chr_pos)
        sort_sc_chr_pos[['chr1', 'pos1']] = sort_chr_pos_1_df[['chr1', 'pos1']]
        sort_sc_chr_pos[['chr2', 'pos2']] = sort_chr_pos_2_df[['chr2', 'pos2']]
        sort_sc_chr_pos = sort_sc_chr_pos.astype({'pos1': 'int64', 'pos2': 'int64'})
        bin_df=sort_sc_chr_pos.groupby(['chr1', 'pos1', 'chr2', 'pos2'], as_index=False).sum()
        bin_file=args.o
        bin_df.to_csv(bin_file, sep='\t')
    end=datetime.datetime.now()
    print("Finished, using",end-start)