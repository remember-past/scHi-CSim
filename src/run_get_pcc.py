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
from scipy.sparse import coo_matrix
from joblib import Parallel, delayed

if __name__ == "__main__":
    parameters_dir = sys.argv[1]
    parameters_df = pd.read_table(parameters_dir, header=None, index_col=0)
    parameters_df.columns = ['value']
    parameters_df.index.name = 'parameters'

    work_dir = parameters_df.loc['work_dir', 'value']
    parameters_dir = os.path.join(work_dir, "parameters.txt")
    cell_base_dir = parameters_df.loc['cell_base_info', 'value']
    raw_data_dir = parameters_df.loc['raw_data', 'value']
    sim_data_dir = parameters_df.loc['sim_data', 'value']
    features_dir = parameters_df.loc['features', 'value']

    sys.path.append(work_dir)

    import src.simulation_framework_base_info
    import src.function_prepare_features as features

    cell_cycle_dir = cell_base_dir

    GATC_bed = "GATC.bed"
    GATC_fends = "GATC.fends"

    chr_length = "chr_length"
    fragment_num = "fragment_num"


    GATC_bed_file = os.path.join(cell_cycle_dir, GATC_bed)

    sim_info = src.simulation_framework_base_info.simulation_framework_base_info(cell_cycle_dir, GATC_bed, GATC_fends,
                                                                                 chr_length,
                                                                                 fragment_num)
    sim_info.get_cell_name_list()
    sim_info.self_read_GATC_fends()
    sim_info.self_get_GATC_bed()
    sim_info.self_get_chr_name()
    sim_info.self_get_chr_length()



    raw_acp_df_dic = {}
    for cell_name in sim_info.cell_name_list:

        one_df = sim_info.read_chr_pos(raw_data_dir, cell_name)

        raw_acp_df_dic[cell_name] = one_df


    cell_name_list = sim_info.cell_name_list
    test_data_df_dic = raw_acp_df_dic
    all_pcc_file = os.path.join(features_dir, 'pcc_file.txt')
    pcc_intermediate_file_dir = os.path.join(features_dir, "pcc_intermediate_file")
    sim_info.recursive_mkdir(pcc_intermediate_file_dir)
    threshold = 99.5
    one_or_sum = 'sum'
    max_value=10
    # features.get_PCC_vector_de_novo_MultiProcess_create_intermediate_file_sepcific_threshold_norm(sim_info,
    #                                                                                               test_data_df_dic,
    #                                                                                               cell_name_list,
    #                                                                                               all_pcc_file,
    #                                                                                               pcc_intermediate_file_dir,
    #                                                                                               threshold, one_or_sum,max_value=1)
    # features.get_PCC_vector_de_novo_load_intermediate_file_sepcific_threshold_MultiProcess(sim_info, cell_name_list,
    #                                                                               all_pcc_file,
    #                                                                               pcc_intermediate_file_dir, threshold,
    #                                                                               one_or_sum)
    features.get_PCC_vector_de_novo_NotMultiProcess_save_intermediate_file_sepcific_threshold(sim_info, test_data_df_dic,
                                                                                     cell_name_list, all_pcc_file,
                                                                                     pcc_intermediate_file_dir,
                                                                                     threshold, one_or_sum)