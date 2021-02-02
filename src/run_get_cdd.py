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
        # print("The",parameter_name,"is not designated, the default value", default_value_dic[parameter_name],"will be used.",sep=' ')
        return default_value_dic[parameter_name]
    else:
        # print("Loading",parameter_name,": ",parameters_df.loc[parameter_name,'value'])
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
    # parser.add_argument('-c', '-comp', nargs='?', type=int, help='components\' number', default=2)

    return parser


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
    sim_info.self_read_GATC_fends(cell_base_info_dir)
    # sim_info.self_get_GATC_bed(cell_base_info_dir)
    sim_info.self_get_chr_length()
    # sim_info.get_cell_stage_info()

    # print("Start to extract cell features...")

    start=time.time()
    raw_acp_df = pd.DataFrame()
    raw_name_list = []
    raw_acp_df_list = []
    i = 0
    raw_acp_df_dic = {}
    for cell_name in sim_info.cell_name_list:
        # print(i, cell_name)
        i = i + 1
        one_df = sim_info.read_chr_pos(raw_data_dir, cell_name)
        raw_name_list.append(cell_name)
        raw_acp_df_list.append(one_df)
        raw_acp_df_dic[cell_name] = one_df
    raw_acp_df = pd.concat(raw_acp_df_list, axis=0, sort=False)
    # raw_acp_df.reset_index(inplace=True)   # raw_acp_df already has column 'cell_name'
    raw_acp_df.reset_index(inplace=True, drop=True)
    raw_cdd_file_path = os.path.join(features_dir, 'cdd_file.txt')
    raw_cdd_df = features.get_CDD_vector(sim_info.cell_name_list, raw_acp_df, raw_cdd_file_path)
    end=time.time()
    print("Finishing extracting CDD, using",end-start)