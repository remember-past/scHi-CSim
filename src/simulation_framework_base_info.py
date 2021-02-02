# -*- coding: utf-8 -*-
# @Time    : 2020/8/13 18:10
# @Author  : scfan
# @FileName: simulation_framework_base_info.py
# @Software: PyCharm

# cell_cycle_info: data, chromosome length, cell name,...


import os
import time
import datetime
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
from scipy.stats import zscore
import copy
import pickle
from tqdm import tqdm

import src.function_prepare_features as features

class simulation_framework_base_info(object):
    def __init__(self,cell_cycle_dir,GATC_bed,GATC_fends,chr_length,
                 fragment_num):
        self.cell_cycle_dir=cell_cycle_dir
        self.GATC_bed=GATC_bed
        self.GATC_fends=GATC_fends
        self.chr_length=chr_length
        self.fragment_num=fragment_num

        self.CIRCLET_cell_stage_number_list=[35, 115, 170, 60, 205, 35, 80, 91, 58, 100, 200, 22]

        self.CIRCLET_cell_stage_name_list = ['Post-M', 'G1-1', 'G1-2', 'G1-ES', 'ES', 'ES-MS', 'MS-1', 'MS-2', 'MS-LS', 'LS-G2', 'G2',
                           'Pre-M']
        self.five_region_stage_index_list=[[1,2],[4],[6,7],[8,9],[10]]
        self.five_region_name=['G1','ES','MS','MS-G2','G2']

        self.get_CIRCLET_cell_stage_start_end()
        self.self_get_chr_length()
        self.self_get_chr_length_dic_without_Y()
        self.bin_dic={'10kb':10000,'25kb':25000,'40kb':40000,'100kb':100000,'250kb':250000}
        self.get_CIRCLET_region_start_end()
        self.five_cross_region_stage_name_to_file_name_dic={'G1+ES': '35-320+380-585', 'ES+MS': '380-585+620-791', 'MS+MS-G2': '620-791+791-949', 'MS-G2+G2': '791-949+949-1149', 'G2+G1': '949-1149+35-320'}
        self.five_cross_region_file_name_to_stage_name_dic={'35-320+380-585': 'G1+ES', '380-585+620-791': 'ES+MS', '620-791+791-949': 'MS+MS-G2', '791-949+949-1149': 'MS-G2+G2', '949-1149+35-320': 'G2+G1'}
        self.five_cross_region_file_name_list=['35-320+380-585', '380-585+620-791', '620-791+791-949', '791-949+949-1149', '949-1149+35-320']
    def get_code_dir(self,code_dir):
        self.code_dir=code_dir
    def get_CIRCLET_cell_stage_start_end(self):
        CIRCLET_cell_stage_start_end_list_dic = {}
        CIRCLET_cell_stage_start_end_name_dic={}
        CIRCLET_cell_stage_start_end_list_list = []
        CIRCLET_cell_stage_start_end_name_list=[]
        for stage_index, stage_number in enumerate(self.CIRCLET_cell_stage_number_list):
            stage_name = self.CIRCLET_cell_stage_name_list[stage_index]
            if (stage_index == 0):
                temp_start = 0
                temp_end = temp_start + stage_number
            else:
                temp_start = CIRCLET_cell_stage_start_end_list_list[stage_index - 1][1]
                temp_end = temp_start + stage_number
            CIRCLET_cell_stage_start_end_list_list.append([temp_start, temp_end])
            CIRCLET_cell_stage_start_end_name_list.append(str(temp_start)+'-'+str(temp_end))
            CIRCLET_cell_stage_start_end_list_dic[stage_name] = [temp_start, temp_end]
            CIRCLET_cell_stage_start_end_name_dic[stage_name] = str(temp_start)+'-'+str(temp_end)
        self.CIRCLET_cell_stage_start_end_list_list=CIRCLET_cell_stage_start_end_list_list
        self.CIRCLET_cell_stage_start_end_list_dic=CIRCLET_cell_stage_start_end_list_dic
        self.CIRCLET_cell_stage_start_end_name_list = CIRCLET_cell_stage_start_end_name_list
        self.CIRCLET_cell_stage_start_end_name_dic = CIRCLET_cell_stage_start_end_name_dic
        self.CIRCLET_cell_stage_start_end_name_to_name_dic={}
        for i,start_end_name in enumerate(self.CIRCLET_cell_stage_start_end_name_list):
            self.CIRCLET_cell_stage_start_end_name_to_name_dic[start_end_name]=self.CIRCLET_cell_stage_name_list[i]

    def get_CIRCLET_region_start_end(self):
        five_region_start_end_list_list = []
        five_region_start_end_name_list = []
        for start_end in self.five_region_stage_index_list:
            region_start = self.CIRCLET_cell_stage_start_end_list_list[start_end[0]][0]
            region_end = self.CIRCLET_cell_stage_start_end_list_list[start_end[-1]][1]
            five_region_start_end_name_list.append(str(region_start) + '-' + str(region_end))
            five_region_start_end_list_list.append([region_start, region_end])
        self.five_region_start_end_name_list=five_region_start_end_name_list
        self.five_region_start_end_list_list=five_region_start_end_list_list
        five_region_start_end_name_to_name_dic={}
        for i,start_end in enumerate(five_region_start_end_name_list):
            five_region_start_end_name_to_name_dic[start_end]=self.five_region_name[i]
        self.five_region_start_end_name_to_name_dic=five_region_start_end_name_to_name_dic
    def get_index_of_CRICLET_12_stage_by_cell_index(self,cell_index):
        target_stage_index = 0
        for stage_index, stage_list in enumerate(self.CIRCLET_cell_stage_start_end_list_list):
            if (cell_index >= stage_list[0]) and (cell_index < stage_list[1]):
                target_stage_index = stage_index
                break
        return target_stage_index
    def get_specific_distnaces_cell_list(self,cell_name_list,distances):
        '''

        :param cell_name_list:
        :param distances:
        :return: specific_distances_cell_name={},
                    specific_distances_cell_name[cell_name]=list[specific_distance_cell_one,specific_distance_cell_two]
        '''
        specific_distances_cell_name={}
        for name in cell_name_list:
            specific_distances_cell_name[name]=[]
        for name in specific_distances_cell_name:
            combine_cell_index_origin = np.array([(-distances), distances])
            cell_number = len(self.cell_phase_list[0])
            cell_index = self.cell_phase_list[0].index(name)
            print(name,cell_index)
            combine_cell_index = (combine_cell_index_origin + cell_index) % cell_number
            print(combine_cell_index)
            for i in combine_cell_index:
                specific_distances_cell_name[name].append(self.cell_phase_list[0][i])
        return specific_distances_cell_name

    def get_random_list_from_list(self,raw_list,number):
        sample_number=len(raw_list)
        result_list=[]
        result_index=np.random.choice(sample_number,number,replace=False)
        for i in result_index:
            result_list.append(raw_list[i])
        return result_list
    def get_random_select_cell_dic_list(self,cell_name_list,random_select_number):
        random_dic_list={}
        for cell_name in cell_name_list:
            random_dic_list[cell_name]=[]
            cell_number = len(self.cell_phase_list[0])
            index_list=np.random.choice(cell_number,random_select_number,replace=False)
            for one_index in index_list:
                random_dic_list[cell_name].append(self.cell_phase_list[0][one_index])
        return random_dic_list
    def get_test_set(self,start,combine_num):
        cell_number=len(self.cell_phase_list[0])

        time = int(np.ceil(cell_number / combine_num))

        test_set = []
        for i in range(time):
            index = (start + combine_num * i) % cell_number
            test_set.append(index)
        test_set_name=[]
        for i in test_set:
            test_set_name.append(self.cell_phase_list[0][i])
        return  test_set_name
    def get_cell_stage_info(self):
        '''
        get self.cell_qc_sc_df  self.cell_stage_dic+
        :return:
        '''
        self.cell_qc_sc_df=pd.read_table(os.path.join(self.cell_cycle_dir,"passed_qc_sc_DF_RO.txt"),header=0,index_col=0)
        temp_stage_series=self.cell_qc_sc_df['cond']
        KeyCellName_ValueStage_dic={}
        for cell_name , cell_cond in temp_stage_series.items():
            KeyCellName_ValueStage_dic[cell_name]=cell_cond.lstrip('1CD_')
        self.KeyCellName_ValueStage_dic=KeyCellName_ValueStage_dic
        KeyStage_ValueCellNameList_dic={}
        for cell_stage in set(KeyCellName_ValueStage_dic.values()):
            KeyStage_ValueCellNameList_dic[cell_stage]=[]
        for cell_name in KeyCellName_ValueStage_dic:
            KeyStage_ValueCellNameList_dic[KeyCellName_ValueStage_dic[cell_name]].append(cell_name)
        self.KeyStage_ValueCellNameList_dic=KeyStage_ValueCellNameList_dic
        self.cell_stage_name_FACS=list(set(KeyStage_ValueCellNameList_dic.keys()))
        return 0

    def get_specific_cell_stage_cell_name_list(self,cell_stage,sample_ratio):
        cell_name_list=self.KeyStage_ValueCellNameList_dic[cell_stage]
        all_number=len(cell_name_list)
        sample_number=int(len(cell_name_list)*sample_ratio)
        sample_index=np.random.choice(all_number,sample_number,replace=False)
        result_list=list(np.array(cell_name_list)[sample_index])
        print(cell_stage,len(cell_name_list),sample_number,sample_ratio,sample_index)
        return result_list
    def get_random_cell_dic_list_from_different_stage(self,cell_name_list,random_select_number):
        '''
        the probilities of three stage are equal, 1/3.
        return dic_list
        :param cell_name_list:
        :param random_select_number:
        :return:
        '''
        dic_list={}
        for cell_name in cell_name_list:
            cell_cond=self.KeyCellName_ValueStage_dic[cell_name]
            #first decide the cond
            cell_cond_name=copy.deepcopy(self.cell_stage_name_FACS)
            cell_cond_index=cell_cond_name.index(cell_cond)
            cell_cond_name.pop(cell_cond_index)
            sample_cond_name=cell_cond_name[np.random.choice(3,1)[0]]
            temp_np_array=np.array(self.KeyStage_ValueCellNameList_dic[sample_cond_name])
            sample_cell_name_list=list(temp_np_array[np.random.choice(len(temp_np_array),random_select_number,replace=False)])
            dic_list[cell_name]=sample_cell_name_list
        return dic_list

    def get_cell_phase_info(self,cell_phase_file):
        # cell_phase_content
        # 1CDX1_1	phase_1	Diploid_15_16_17_18
        # 1CDX1_2	phase_1	Diploid_15_16_17_18

        file_handle = None
        cell_phase_list = []
        cell_phase_name = []
        cell_phase_phase_num = []
        cell_phase_exp_batch = []
        try:
            file_handle = open(cell_phase_file, 'r')
            print("Start to read ", cell_phase_file)

            for line in file_handle.readlines():
                temp = line.strip("\n").split("\t")

                cell_phase_name.append(temp[0])
                cell_phase_phase_num.append(temp[1])
                cell_phase_exp_batch.append(temp[2])
        except FileNotFoundError:
            print_str = "Can't open file " + cell_phase_file
            print(print_str)
        finally:
            if file_handle:
                file_handle.close()
        cell_phase_list.append(cell_phase_name)
        cell_phase_list.append(cell_phase_phase_num)
        cell_phase_list.append(cell_phase_exp_batch)

        return cell_phase_list
    def get_cell_name_list(self,cell_base_info_dir):
        # cell_phase_content
        # 1CDX1_1	phase_1	Diploid_15_16_17_18
        # 1CDX1_2	phase_1	Diploid_15_16_17_18
        # cell_name_list_path=r"E:\Users\scfan\data\CellCycle\release_version\cell_base_info\cell_name_list"

        cell_name_list_path=os.path.join(cell_base_info_dir,"cell_name_list.txt")
        try:
            cell_name_list_df = pd.read_table(cell_name_list_path, header=None)
            self.cell_name_list = list(cell_name_list_df[0].values)
        except FileNotFoundError:
            print_str = "Can't open file " + cell_name_list_path
            print(print_str)

        return 0

    def read_adj(self,raw_data_dir,cell_name):
        adj_file = os.path.join(raw_data_dir, cell_name, 'adj')
        adj_df = pd.read_table(adj_file, header=0)
        return adj_df
    def write_out_chr_pos_df(self,sc_chr_pos_df,raw_data_dir,cell_name):
        chr_pos_file = os.path.join(raw_data_dir, cell_name, 'chr_pos')
        sc_chr_pos_df.to_csv(chr_pos_file, sep='\t')
    def write_out_chr_pos_df_with_replicates(self,sc_chr_pos_df,raw_data_dir,cell_name,replicates_serial_number):
        replicates_dir=os.path.join(raw_data_dir, cell_name,'replicates_'+str(replicates_serial_number))
        self.recursive_mkdir(replicates_dir)
        chr_pos_file = os.path.join(replicates_dir, 'chr_pos')
        sc_chr_pos_df.to_csv(chr_pos_file, sep='\t')
    def read_chr_pos(self,raw_data_dir,cell_name):
        chr_pos_file = os.path.join(raw_data_dir, cell_name, 'chr_pos')
        sc_chr_pos_df = pd.read_table(chr_pos_file, index_col=0)
        return sc_chr_pos_df
    def convert_adj_to_chr_pos(self,raw_data_dir,cell_name):

        adj_df=self.read_adj(raw_data_dir,cell_name)

        sc_chr_pos = pd.DataFrame()

        fend_1 = self.GATC_fends_df.loc[adj_df['fend1']]
        fend_1['chr'] = 'chr' + fend_1['chr'].astype(str)

        fend_1.reset_index(drop=True, inplace=True)
        fend_1.columns = ['chr1', 'pos1']

        fend_2 = self.GATC_fends_df.loc[adj_df['fend2']]
        fend_2['chr'] = 'chr' + fend_2['chr'].astype(str)
        fend_2.reset_index(drop=True, inplace=True)
        fend_2.columns = ['chr2', 'pos2']

        sc_chr_pos = pd.concat([sc_chr_pos, fend_1, fend_2], axis=1, sort=False)

        sc_chr_pos['count'] = 1
        sc_chr_pos['cell_name'] = cell_name

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
        sc_chr_pos = sort_sc_chr_pos
        self.write_out_chr_pos_df( sc_chr_pos, raw_data_dir, cell_name)
    def convert_adj_to_chr_pos_by_cell_name_list(self,raw_data_dir,cell_name_list):
        for cell_name in tqdm(cell_name_list):
            print("Start to convert",cell_name)
            self.convert_adj_to_chr_pos(raw_data_dir, cell_name)

    def extract_features(self,features_dir,raw_data_dir,cell_name_list):

        raw_cdd_file_path = os.path.join(features_dir, 'cdd_file.txt')
        raw_cdd_df = features.get_CDD_vector(sim_info.cell_name_list, raw_acp_df, raw_cdd_file_path)
    def get_cell_cell_distance(self,features_dir):
        distance.get_distance(features_dir)
    def simulation_generate_sim_statistic_parameters(self,raw_data_dir,cell_name_list):
        sim.generate_sim_statistic_parameter(self,raw_data_dir,cell_name_list,sim_data_dir)
    def simulation_sample(self,seqDepthTime,combineNumber,raw_data_dir,sim_data_dir):
        sim.sample_by_sim_statistic_parameter(seqDepthTime,combineNumber,raw_data_dir,sim_data_dir)
    def get_pcc_pairs(self,pcc_file):
        pcc_df = pd.read_table(pcc_file, header=0, index_col=0)
        return pcc_df


    def self_get_GATC_bed(self,cell_base_info_dir):
        GATC_bed_file=os.path.join(cell_base_info_dir,"GATC.bed")
        GATC_bed_df = pd.read_table(GATC_bed_file, header=None)
        GATC_bed_df.columns = ["chr", "start", "end", "frag", "score", "strand"]
        self.GATC_bed_df=GATC_bed_df
        return 0


    def get_bed_info(self,bed_file):
        # use fend index, which means fend position, to get chr name, chr position, fragment position
        # return [chr_name,chr_position,chr_media_pos,fragment_position]
        #      fend_chr_name= chr_name[fend_index]

        # Note: For example, the chr_frag_adj is located in phase_1/1CDX1_1/chr_frag_adj_dir/chr_frag_adj
        chr_name = []

        chr_position = []
        chr_media_pos=[]
        fragment_position = []

        GATC_bed_file=bed_file
        # chr1	2	3000536	HIC_chr1_1	0	+
        # chr1	3000536	3000801	HIC_chr1_2	0	+

        try:
            bed_handle = open(GATC_bed_file, 'r')
            print("start to read", GATC_bed_file)
            for line in bed_handle.readlines():
                temp = line.strip('\n').split('\t')
                chr_name.append(temp[0])
                chr_name.append(temp[0])

                chr_position.append(int(temp[1]))
                chr_position.append(int(temp[2]))
                temp_media_pos=int((int(temp[1])+int(temp[2]))/2)
                chr_media_pos.append(temp_media_pos)
                chr_media_pos.append(temp_media_pos)
                temp_fragment_position = temp[3].split('_')[2]
                fragment_position.append(int(temp_fragment_position))
                fragment_position.append(int(temp_fragment_position))
                # print("i=%s"%(i))
                # print("%s %s "%(frag_chr[i-1],frag_start[i-1]))
        except FileNotFoundError:
            print_str = "Can't open file " + GATC_bed_file
            print(print_str)
            return -1
        finally:
            if bed_handle:
                bed_handle.close()
        bed_list=[]
        bed_list.append(chr_name)
        bed_list.append(chr_position)
        bed_list.append(chr_media_pos)
        bed_list.append(fragment_position)
        return bed_list
    def get_chr_length_info(self,chr_length_file):
        chr_length_dic={}
# chr1	197195432
# chr2	181748087
        file_handle=open(chr_length_file,'r')
        for line in file_handle.readlines():
            temp=line.strip('\n').split('\t')
            chr_length_dic[temp[0]]=int(temp[1])
        return chr_length_dic

    def self_get_chr_length(self):
        chr_length_file=os.path.join(self.cell_cycle_dir,self.chr_length)
        self.chr_length_dic=self.get_chr_length_info(chr_length_file)
        return 0

    def self_get_chr_length_dic_without_Y(self):
        chr_length_dic_without_Y={}
        for chr_name,chr_length in self.chr_length_dic.items():
            if(chr_name=='chrY'):
                continue
            chr_length_dic_without_Y[chr_name]=chr_length
        self.chr_length_dic_without_Y=chr_length_dic_without_Y
    def get_fragment_num_info(self, fragment_num_file):
        fragment_num_dic = {}
# chr1	479081
# chr2	441652
        file_handle = open(fragment_num_file, 'r')
        for line in file_handle.readlines():
            temp = line.strip('\n').split('\t')
            fragment_num_dic[temp[0]] = int(temp[1])
        return fragment_num_dic
    def get_chr_name(self,max_chr_num=19):
        chr_name_list=[]
        for i in range(1,max_chr_num+1):
            temp='chr'+str(i)
            chr_name_list.append(temp)
        chr_name_list.append('chrX')

        return chr_name_list
    def self_get_chr_name(self):
        self.chr_name_list = self.get_chr_name()
        return 0
    def get_fends_info(self,fends_file):
        fends_info_df=pd.read_table(fends_file, header=0, index_col=0,dtype={"fend":'int64','chr':'object','coord':'int64'})
        header=fends_info_df.columns
        # for index,row in fends_info_df.iterrows():
        #     print(index,row)
        return fends_info_df
    def get_all_cell_adj_acp(self):
        adj_name='all_raw_adj'
        acp_name='all_raw_abj'#acp adjoin chr position
        adj_file=os.path.join(self.raw_single_cell_dir,adj_name)
        acp_file=os.path.join(self.raw_single_cell_dir,acp_name)
        all_raw_adj = pd.DataFrame()
        all_raw_acp = pd.DataFrame()
        if (not os.path.exists(adj_file)) or (not os.path.exists(acp_file)):

            for i,cell_name in enumerate(self.cell_phase_list[0]):
                # cell_name = self.cell_phase_list[0][0]
                print(i,cell_name)
                cell_phase = self.cell_phase_list[1][i]
                cell_file = os.path.join(self.raw_single_cell_dir, cell_phase, cell_name, "adj")
                sc_df = pd.read_table(cell_file, header=0)

                sc_chr_pos = pd.DataFrame()

                sc_chr1 = sc_df['fend1'].apply(lambda x: self.GATC_fends_df.ix[int(x), 'chr'])
                sc_pos1 = sc_df['fend1'].apply(lambda x: self.GATC_fends_df.ix[int(x), 'coord'])

                sc_chr2 = sc_df['fend2'].apply(lambda x: self.GATC_fends_df.ix[int(x), 'chr'])
                sc_pos2 = sc_df['fend2'].apply(lambda x: self.GATC_fends_df.ix[int(x), 'coord'])

                sc_chr_pos['chr1'] = sc_chr1
                sc_chr_pos['pos1'] = sc_pos1
                sc_chr_pos['chr2'] = sc_chr2
                sc_chr_pos['pos2'] = sc_pos2
                sc_chr_pos['count'] = sc_df['count']
                sc_df['cell_name']=cell_name
                sc_chr_pos['cell_name']=cell_name
                all_raw_adj=pd.concat([all_raw_adj,sc_df],axis=0)
                all_raw_acp=pd.concat([all_raw_acp,sc_chr_pos],axis=0)
            all_raw_adj.to_pickle(adj_file)
            all_raw_acp.to_pickle(acp_file)
            self.all_raw_adj=all_raw_adj
            self.all_raw_acp=all_raw_acp
        else:
            all_raw_adj=pd.read_pickle(adj_file)
            all_raw_acp=pd.read_pickle(acp_file)
            self.all_raw_adj=all_raw_adj
            self.all_raw_acp=all_raw_acp
        self.all_raw_acp['chr1'] = self.all_raw_acp['chr1'].astype(str)
        self.all_raw_acp['chr2'] = self.all_raw_acp['chr2'].astype(str)
        return 0
    def get_all_cell_adj(self):
        adj_name = 'all_raw_adj'
        adj_file = os.path.join(self.raw_single_cell_dir, adj_name)
        all_raw_adj = pd.read_pickle(adj_file)
        self.all_raw_adj = all_raw_adj
        return self.all_raw_adj
    def get_all_cell_acp(self):
        acp_name='all_raw_abj'#acp adjoin chr position
        acp_file=os.path.join(self.raw_single_cell_dir,acp_name)
        all_raw_acp=pd.read_pickle(acp_file)
        self.all_raw_acp=all_raw_acp
        self.all_raw_acp['chr1'] = self.all_raw_acp['chr1'].astype(str)
        self.all_raw_acp['chr2'] = self.all_raw_acp['chr2'].astype(str)
        self.all_raw_acp.reset_index(inplace=True,drop=True)
        return 0
    def get_sort_all_cell_acp(self):
        sort_acp_name='sort_all_raw_acp'
        sort_acp_file=os.path.join(self.raw_single_cell_dir,sort_acp_name)
        if not os.path.exists(sort_acp_file):
            temp_store_df = pd.DataFrame()
            all_raw_acp=self.all_raw_acp
            chr_pos_1 = all_raw_acp['chr1'] + '+' + all_raw_acp['pos1'].astype(str)
            chr_pos_2 = all_raw_acp['chr2'] + '+' + all_raw_acp['pos2'].astype(str)
            temp_store_df['chr_pos_1'] = chr_pos_1
            temp_store_df['chr_pos_2'] = chr_pos_2
            temp_store_df['chr_pos_min'] = temp_store_df[['chr_pos_1', 'chr_pos_2']].min(axis=1)
            temp_store_df['chr_pos_max'] = temp_store_df[['chr_pos_1', 'chr_pos_2']].max(axis=1)
            sort_chr_pos_1_df = pd.DataFrame(list(temp_store_df['chr_pos_min'].str.split('+')),
                                             columns=['chr1', 'pos1'])
            sort_chr_pos_2_df = pd.DataFrame(list(temp_store_df['chr_pos_max'].str.split('+')),
                                             columns=['chr2', 'pos2'])

            sort_all_raw_acp = copy.deepcopy(all_raw_acp)
            sort_all_raw_acp[['chr1', 'pos1']] = sort_chr_pos_1_df[['chr1', 'pos1']]
            sort_all_raw_acp[['chr2', 'pos2']] = sort_chr_pos_2_df[['chr2', 'pos2']]
            sort_all_raw_acp = sort_all_raw_acp.astype({'pos1': 'int64', 'pos2': 'int64'})
            sort_all_raw_acp.to_pickle(sort_acp_file)
        else:
            sort_all_raw_acp=pd.read_pickle(sort_acp_file)
        self.sort_all_raw_acp=sort_all_raw_acp
        return self.sort_all_raw_acp

    def get_all_cell_afj_incorrect(self):
        afj_name="all_raw_afj"
        afj_name_file=os.path.join(self.raw_single_cell_dir,afj_name)

        if not os.path.exists(afj_name_file):
            print("produce all_raw_afj file")
            all_fend_frag = copy.deepcopy(self.all_raw_adj)
            new_fend1 = all_fend_frag[['fend1', 'fend2']].min(axis=1)
            new_fend2 = all_fend_frag[['fend1', 'fend2']].max(axis=1)
            all_fend_frag['fend1'] = new_fend1
            all_fend_frag['fend2'] = new_fend2
            fend1_frag = self.GATC_bed_df['frag'].iloc[all_fend_frag['fend1'] // 2]
            fend2_frag = self.GATC_bed_df['frag'].iloc[all_fend_frag['fend2'] // 2]
            fend1_frag.reset_index(drop=True, inplace=True)
            fend2_frag.reset_index(drop=True, inplace=True)
            #The following two lines introduces errors, all_fend_frag index and fend1_frag index are not compatible
            all_fend_frag['frag1'] = fend1_frag
            all_fend_frag['frag2'] = fend2_frag
            all_fend_frag_order_column = all_fend_frag.iloc[:, [0, 4, 1, 5, 2, 3]]

            all_fend_frag_order_column.to_pickle(afj_name_file)
            self.all_raw_afj=all_fend_frag_order_column

        else:
            self.all_raw_afj=pd.read_pickle(afj_name_file)
        return self.all_raw_afj
    def get_all_cell_afj(self):
        afj_name="all_raw_afj"
        afj_name_file=os.path.join(self.raw_single_cell_dir,afj_name)

        if not os.path.exists(afj_name_file):
            print("produce all_raw_afj file")
            all_fend_frag = copy.deepcopy(self.all_raw_adj)
            new_fend1 = all_fend_frag[['fend1', 'fend2']].min(axis=1)#the type of fend is numberic
            new_fend2 = all_fend_frag[['fend1', 'fend2']].max(axis=1)
            all_fend_frag['fend1'] = new_fend1
            all_fend_frag['fend2'] = new_fend2
            fend1_frag = self.GATC_bed_df.iloc[all_fend_frag['fend1'] // 2]['frag']
            fend2_frag = self.GATC_bed_df.iloc[all_fend_frag['fend2'] // 2]['frag']
            fend1_frag.reset_index(drop=True, inplace=True)
            fend2_frag.reset_index(drop=True, inplace=True)
            all_fend_frag.reset_index(drop=True, inplace=True)
            all_fend_frag['frag1'] = fend1_frag
            all_fend_frag['frag2'] = fend2_frag
            all_fend_frag_order_column = all_fend_frag.iloc[:, [0, 4, 1, 5, 2, 3]]

            all_fend_frag_order_column.to_pickle(afj_name_file)
            self.all_raw_afj=all_fend_frag_order_column

        else:
            self.all_raw_afj=pd.read_pickle(afj_name_file)
        return self.all_raw_afj
    def get_all_cell_afj_dic(self):
        all_raw_cell_df_dic_file = os.path.join(self.raw_single_cell_dir, 'all_raw_cell_afj_df_dic.pkl')
        if not os.path.exists(all_raw_cell_df_dic_file):
            all_raw_afj = self.all_raw_afj.set_index('cell_name')
            all_raw_cell_df_dic = {}
            for cell_name in self.cell_phase_list[0]:
                all_raw_cell_df_dic[cell_name] = all_raw_afj.loc[cell_name]
        else:
            all_raw_cell_df_dic=self.load_variable_from_pikle_file(all_raw_cell_df_dic_file)
        self.all_raw_cell_afj_df_dic=all_raw_cell_df_dic
        return self.all_raw_cell_afj_df_dic
    def get_all_cell_acp_dic(self):
        '''
        all_raw_cell_acp_df_dic for simulation
        1. make sure sort_all_row_acp['chr']='chr1', not '1'
        2. using sort_all_row, make sure chr1 <= chr2, pos1<=pos2
        :return:
        '''

        all_raw_cell_acp_df_dic_file = os.path.join(self.raw_single_cell_dir, 'all_raw_cell_acp_df_dic.pkl')
        if not os.path.exists(all_raw_cell_acp_df_dic_file):
            if (self.sort_all_raw_acp.index.name != 'cell_name'):
                all_raw_acp = self.sort_all_raw_acp.set_index('cell_name')
                temp_df = pd.DataFrame(index=all_raw_acp.index)
                temp_df['chr_name'] = 'chr'
                all_raw_acp['chr1'] = temp_df['chr_name'] + all_raw_acp['chr1']
                all_raw_acp['chr2'] = temp_df['chr_name'] + all_raw_acp['chr2']
            print(all_raw_acp)
            all_raw_cell_acp_df_dic = {}
            for i, cell_name in enumerate(self.cell_phase_list[0]):
                print(i, cell_name)
                all_raw_cell_acp_df_dic[cell_name] = all_raw_acp.loc[cell_name]
            self.store_variable_from_pikle_file(all_raw_cell_acp_df_dic_file, all_raw_cell_acp_df_dic)
        else:
            all_raw_cell_acp_df_dic=self.load_variable_from_pikle_file(all_raw_cell_acp_df_dic_file)
        self.all_raw_cell_acp_df_dic=all_raw_cell_acp_df_dic
        return self.all_raw_cell_acp_df_dic
    def get_all_raw_cell_acp_count_one_df_dic(self):
        all_raw_cell_acp_count_one_df_dic_file = os.path.join(self.raw_single_cell_dir, 'all_raw_cell_acp_count_one_df_dic.pkl')
        if not os.path.exists(all_raw_cell_acp_count_one_df_dic_file):
            all_raw_cell_acp_count_one_df_dic = {}

            for i, j in enumerate(self.all_raw_cell_acp_df_dic.items()):
                print(i, j[0])
                one_df = j[1]
                temp = one_df.groupby(['chr1', 'chr2', 'pos1', 'pos2'], as_index=False, sort=False)['count'].sum()
                temp['count'] = 1
                temp['cell_name'] = j[0]
                temp.set_index('cell_name', inplace=True, drop=True)
                all_raw_cell_acp_count_one_df_dic[j[0]] = temp
            self.store_variable_from_pikle_file(all_raw_cell_acp_count_one_df_dic_file, all_raw_cell_acp_count_one_df_dic)
        else:
            all_raw_cell_acp_count_one_df_dic=self.load_variable_from_pikle_file(all_raw_cell_acp_count_one_df_dic_file)
        self.all_raw_cell_acp_count_one_df_dic=all_raw_cell_acp_count_one_df_dic
        return self.all_raw_cell_acp_count_one_df_dic
    def get_all_raw_chr_pos_fend_frag_count_one_df_dic(self):

        # design all_raw_chr_pos_fend_frag_count_one_df_dic[cell_name].columns=['fend1', 'frag1', 'fend2', 'frag2', 'count', 'chr1', 'start1', 'end1','chr2', 'start2', 'end2', 'chr1_frag_index', 'chr2_frag_index']
        all_raw_chr_pos_fend_frag_count_one_df_dic_file_path = os.path.join(self.raw_single_cell_dir,
                                                                            "all_raw_chr_pos_fend_frag_count_one_df_dic.pkl")
        if not os.path.exists(all_raw_chr_pos_fend_frag_count_one_df_dic_file_path):
            print("creating file")
            all_raw_chr_pos_fend_frag_count_one_df_dic = {}
            GATC_bed_df = self.GATC_bed_df.set_index('frag')
            i = 0
            for cell_name, target_df in self.all_raw_cell_afj_df_dic.items():
                print(i, cell_name)
                i = i + 1
                target_df.reset_index(inplace=True, drop=True)
                frag1_bed_df = GATC_bed_df.loc[target_df['frag1']]
                frag2_bed_df = GATC_bed_df.loc[target_df['frag2']]
                frag1_bed_df.reset_index(drop=True, inplace=True)
                frag2_bed_df.reset_index(drop=True, inplace=True)
                frag1_index_df = pd.DataFrame(list(target_df['frag1'].str.split('_')),
                                              columns=['HIC', 'chr', 'chr_index'])
                frag2_index_df = pd.DataFrame(list(target_df['frag2'].str.split('_')),
                                              columns=['HIC', 'chr', 'chr_index'])
                chr_pos_fend_frag_df = copy.deepcopy(target_df)
                chr_pos_fend_frag_df[['chr1', 'start1', 'end1']] = frag1_bed_df[['chr', 'start', 'end']]
                chr_pos_fend_frag_df[['chr2', 'start2', 'end2']] = frag2_bed_df[['chr', 'start', 'end']]
                chr_pos_fend_frag_df['chr1_frag_index'] = frag1_index_df['chr_index'].astype('int')
                chr_pos_fend_frag_df['chr2_frag_index'] = frag2_index_df['chr_index'].astype('int')
                chr_pos_fend_frag_df['count'] = 1
                all_raw_chr_pos_fend_frag_count_one_df_dic[cell_name] = chr_pos_fend_frag_df
            self.store_variable_from_pikle_file(all_raw_chr_pos_fend_frag_count_one_df_dic_file_path,
                                                    all_raw_chr_pos_fend_frag_count_one_df_dic)
        else:
            print("loading ",all_raw_chr_pos_fend_frag_count_one_df_dic_file_path)
            all_raw_chr_pos_fend_frag_count_one_df_dic=self.load_variable_from_pikle_file(all_raw_chr_pos_fend_frag_count_one_df_dic_file_path)
        self.all_raw_chr_pos_fend_frag_count_one_df_dic=all_raw_chr_pos_fend_frag_count_one_df_dic
        return self.all_raw_chr_pos_fend_frag_count_one_df_dic

    def store_each_cell_for_all_raw_chr_pos_fend_frag_count_one_df_dic(self):
        store_dir = os.path.join(self.cell_cycle_dir,
                                 r"raw\scHiC2_process_author\single_cell\chr_pos_fend_frag_count_one")
        i = 0
        for cell_name, one_df in self.all_raw_chr_pos_fend_frag_count_one_df_dic.items():
            print(i, cell_name)
            i = i + 1
            one_cell_file_path = os.path.join(store_dir, cell_name + ".pkl")
            if (not os.path.exists(one_cell_file_path)):
                self.store_variable_from_pikle_file(one_cell_file_path, one_df)

    def get_all_cell_bin_acp(self,bin_name,bin_dic):
        bin_name_file=os.path.join(self.raw_single_cell_dir,"bin_all_raw_acp_"+bin_name+".pkl")
        if not os.path.exists(bin_name_file):
            print(bin_name_file)
            resolution = bin_dic[bin_name]
            bin_1_name = bin_name + "1"
            bin_2_name = bin_name + "2"
            bin_df=pd.DataFrame()

            bin1 = self.sort_all_raw_acp['pos1'] // resolution
            bin2 = self.sort_all_raw_acp['pos2'] // resolution
            bin_df[bin_1_name] = bin1
            bin_df[bin_2_name] = bin2

            bin1 = bin1.astype(str)
            bin2 = bin2.astype(str)
            chr_bin1_name = 'chr+' + bin_1_name
            chr_bin2_name = 'chr+' + bin_2_name

            chr_bin1 = self.sort_all_raw_acp['chr1'] + '+' + bin1
            chr_bin2 = self.sort_all_raw_acp['chr2'] + '+' + bin2
            bin_df[chr_bin1_name] = chr_bin1
            bin_df[chr_bin2_name] = chr_bin2
            bin_df.to_pickle(bin_name_file)
        else:
            bin_df=pd.read_pickle(bin_name_file)
        return bin_df
    def produce_all_cell_bin_acp(self,bin_name,bin_dic):
        bin_name_file=os.path.join(self.raw_single_cell_dir,"bin_all_raw_acp_"+bin_name+".pkl")
        if not os.path.exists(bin_name_file):
            print(bin_name_file)
            resolution = bin_dic[bin_name]
            bin_1_name = bin_name + "1"
            bin_2_name = bin_name + "2"
            bin_df=pd.DataFrame()

            bin1 = self.sort_all_raw_acp['pos1'] // resolution
            bin2 = self.sort_all_raw_acp['pos2'] // resolution
            bin_df[bin_1_name] = bin1
            bin_df[bin_2_name] = bin2

            bin1 = bin1.astype(str)
            bin2 = bin2.astype(str)
            chr_bin1_name = 'chr+' + bin_1_name
            chr_bin2_name = 'chr+' + bin_2_name

            chr_bin1 = self.sort_all_raw_acp['chr1'] + '+' + bin1
            chr_bin2 = self.sort_all_raw_acp['chr2'] + '+' + bin2
            bin_df[chr_bin1_name] = chr_bin1
            bin_df[chr_bin2_name] = chr_bin2
            bin_df.to_pickle(bin_name_file)
        print("pass out",bin_name_file)
        return 0
    def convert_afj_df_dic_to_acj_df_dic(self,afj_df_dic):

        GATC_frag_index = self.GATC_bed_df.set_index('frag')
        sim_acp_df_dic = {}
        for cell_name in sim_df_dic:
            print(cell_name)
            one_combine_df = afj_df_dic[cell_name]

            pos1 = GATC_frag_index['start'].loc[one_combine_df['frag1']]
            pos2 = GATC_frag_index['start'].loc[one_combine_df['frag2']]
            chr1 = GATC_frag_index['chr'].loc[one_combine_df['frag1']]
            chr2 = GATC_frag_index['chr'].loc[one_combine_df['frag2']]
            count = one_combine_df['count']
            count.reset_index(drop=True, inplace=True)

            contact_df = pd.DataFrame()
            pos1.reset_index(drop=True, inplace=True)
            pos2.reset_index(drop=True, inplace=True)
            chr1.reset_index(drop=True, inplace=True)
            chr2.reset_index(drop=True, inplace=True)
            contact_df['chr1'] = chr1
            contact_df['pos1'] = pos1
            contact_df['chr2'] = chr2
            contact_df['pos2'] = pos2
            contact_df['count'] = count
            sim_acp_df_dic[cell_name] = contact_df
        return sim_acp_df_dic
    def self_set_index_as_name_for_all_raw_afj(self):
        self.all_raw_afj.set_index("cell_name", inplace=True)
        return self.all_raw_afj
    def get_combine_cell_df_as_dic(self,all_raw_afj,combine_number):
    # all_raw_afj has been processed to make sure the index is cell_name
    # all_raw_afj=self.all_raw_afj.set_index('cell_name')
        combine_cell_dic = {}

        self.cell_phase_list[0]

        combine_cell_index_origin = np.array([i for i in range(-(combine_number // 2), combine_number // 2 + 1)])
        cell_number = len(self.cell_phase_list[0])
        for cell_index, raw_single_name in enumerate(self.cell_phase_list[0]):
            combine_cell_index = (combine_cell_index_origin + cell_index) % cell_number

            cell_name_df = pd.DataFrame(self.cell_phase_list[0], columns=["name"])
            combine_cell_name = cell_name_df.loc[combine_cell_index]

            combine_cell = all_raw_afj.loc[combine_cell_name['name']]
            combine_cell_dic[raw_single_name] = combine_cell
            print(cell_index, raw_single_name)
        return  combine_cell_dic
    def get_cell_df_from_afj_as_dic(self,cell_name_list,all_raw_afj):
        raw_single_cell_df_dic = {}
        for cell_name in cell_name_list:
            raw_single_cell_df_dic[cell_name] = all_raw_afj.loc[cell_name]
        return raw_single_cell_df_dic
    def get_cell_df_from_bin_acp_as_dic(self,cell_name_list,all_raw_bin_acp):
        raw_single_cell_df_dic = {}
        for cell_name in cell_name_list:
            raw_single_cell_df_dic[cell_name] = all_raw_bin_acp.loc[cell_name]
        return raw_single_cell_df_dic
    def get_combine_cell_df_for_specific_name_as_dic(self,all_raw_afj,name_list,combine_number):
        '''
        all_raw_afj.index==cell_name
        if above state is not true, codes will correct it automatically
        :param all_raw_afj: dataframe(index="integer from 0 to length of dataframe",columns=[fend1,frag1,fend2,frag2,count])
        :param name_list: get one cell name ,combine neighboor cells for this cell
        :param combine_number:
        :return:combine_cell_dic["cell_name"]=dataframe(index="combine cell name",columns=[fend1,frag1,fend2,frag2,count])
        '''
    # all_raw_afj has been processed to make sure the index is cell_name
    # all_raw_afj=self.all_raw_afj.set_index('cell_name')

        combine_cell_dic = {}
        if(str(all_raw_afj.index.name)!="cell_name"):
            all_raw_afj.set_index('cell_name',inplace=True)#process all_raw_afj to make sure the index is cell_name
        combine_cell_index_origin = np.array([i for i in range(-(combine_number // 2), combine_number // 2 + 1)])
        cell_number = len(self.cell_phase_list[0])
        print("Start to combine cell,","combine num is",combine_number)
        for process_num, raw_single_name in enumerate(name_list):
            start_time=time.time()
            print(process_num,raw_single_name)
            cell_index=self.cell_phase_list[0].index(raw_single_name)

            print(raw_single_name,"position is",cell_index)
            combine_cell_index = (combine_cell_index_origin + cell_index) % cell_number
            print("ready to combine cell index :",combine_cell_index)

            cell_name_df = pd.DataFrame(self.cell_phase_list[0], columns=["name"])
            combine_cell_name = cell_name_df.loc[combine_cell_index]

            combine_cell = all_raw_afj.loc[combine_cell_name['name']]
            combine_cell_dic[raw_single_name] = combine_cell
            end_time=time.time()
            print("combine using time ",int(end_time-start_time),"(s)")
            print("-------------------------------")
        print("================================")
        return  combine_cell_dic
    def write_out_combine_cell_df_dic(self,combine_df_dic,distance_measure_mode,combine_number):
        combine_name = "combine" + str(combine_number)
        all_start_time=datetime.datetime.now()
        for process_number, name in enumerate(combine_df_dic):
            start_time=datetime.datetime.now()
            cell_index=self.cell_phase_list[0].index(name)
            phase = self.cell_phase_list[1][cell_index]
            file_name_dir = os.path.join(self.raw_combine_cell_dir, phase, name, distance_measure_mode, combine_name)
            file_name = os.path.join(file_name_dir, "afj.pickle")
            print("Start to save",process_number, name)
            self.recursive_mkdir(file_name_dir)
            if not os.path.exists(file_name):
                print(file_name,"not exists,","start to write out")
                combine_df_dic[name].to_pickle(file_name)
            else:
                print(file_name,"exists,","don't write out")
            end_time=datetime.datetime.now()
            print("Save using time:",end_time-start_time)
            print("------------------------")

        all_end_time=datetime.datetime.now()
        print("All save using time:", all_end_time - all_start_time)
        print("=================================")
        return 0
    def get_random_combine_cell_df_as_dic(self,all_raw_afj,cell_name_list,combine_num):
        combine_cell_dic = {}
        cell_number = len(self.cell_phase_list[0])

        for process_num, raw_single_name in enumerate(cell_name_list):

            fragProp = []
            cell_index = self.cell_phase_list[0].index(raw_single_name)
            prop = 1 / (cell_number - 1)
            for i in range(cell_number):
                fragProp.append(prop)
            fragProp[cell_index] = 0
            random_index = np.random.choice(cell_number, size=combine_num, replace=False,
                                            p=list(fragProp))
            random_index_array = np.array(random_index)

            combine_cell_index = random_index_array

            cell_name_df = pd.DataFrame(self.cell_phase_list[0], columns=["name"])
            combine_cell_name = cell_name_df.loc[combine_cell_index]

            combine_cell = all_raw_afj.loc[combine_cell_name['name']]
            combine_cell_dic[raw_single_name] = combine_cell
            print(process_num, raw_single_name)
        return combine_cell_dic

    def calculate_compartment_from_dic_df(self,dic_df,compartment_prefix='traj\\combine20\\enhancement\\only_combine\\read_1\\sim_1\\batch_1\\compartment'):
        '''
        calculate compartment
        :param dic_df: dic_df['cell_name']=dataframe(columns=['chr1','pos1','chr2','pos2','count','cell_name'])
        :param prefix:
        :return:
        '''
        for cell_name in dic_df:
            all_start_time = datetime.datetime.now()
            print(cell_name)
            temp_df = dic_df[cell_name]
            temp_df=temp_df.astype({'count': 'int64'})
            cis_contact = temp_df[temp_df['chr1'] == temp_df['chr2']]
            new_pos1 = cis_contact[['pos1', 'pos2']].min(axis=1)
            new_pos2 = cis_contact[['pos1', 'pos2']].max(axis=1)
            cis_contact['pos1'] = new_pos1
            cis_contact['pos2'] = new_pos2
            new_temp_df = cis_contact.groupby(['chr1', 'pos1', 'pos2'], as_index=False)['count'].sum()
            out_summary_chr_dic = {}
            for chr_name in self.chr_name_list:
                out_summary_chr_dic[chr_name] = []
            process_num = 0
            print("start to calculate out_str")
            for row in new_temp_df.itertuples():
                chr_name = getattr(row, 'chr1')
                if "chr" not in chr_name:
                    chr_name="chr"+chr_name
                if (chr_name == "chrY"):
                    continue
                chr_pos1 = getattr(row, 'pos1')
                chr_pos2 = getattr(row, 'pos2')
                count = getattr(row, 'count')

                for j in range(count):
                    out_str = str(process_num) + '_' + str(j) + '\t' + chr_name + '\t' + str(
                        chr_pos1) + '\t+\t' + chr_name + '\t' + str(chr_pos2) + '\t-\n'
                    out_summary_chr_dic[chr_name].append(out_str)
                process_num = process_num + 1
            print("start to process compartment")

            cell_index = self.cell_phase_list[0].index(cell_name)
            phase = self.cell_phase_list[1][cell_index]
            prefix = os.path.join(self.raw_combine_cell_dir, phase, cell_name, compartment_prefix)
            for chr_name in out_summary_chr_dic:
                print(chr_name)
                print("write out summary")
                out_file = os.path.join(prefix, chr_name + '_summary')
                self.recursive_mkdir(prefix)
                with open(out_file, 'w') as f:
                    for out_str in out_summary_chr_dic[chr_name]:
                        f.writelines(out_str)
                command_cstool_str = os.path.join(self.compartment_tool_dir, self.compartment_tool)
                chr_bed_dir = os.path.join(self.compartment_tool_dir, "chr_bed")
                chr_bed_file = os.path.join(chr_bed_dir, chr_name + "_50kb.bed")

                os.chdir(prefix)
                space_str = ' '
                all_command_str = space_str.join(
                    [command_cstool_str, chr_bed_file, out_file, chr_name, "24", "1000000"])
                print("start", all_command_str)
                os.system(all_command_str)
            all_end_time = datetime.datetime.now()
            print("Process end", cell_name)
            print("All process one cell using time:", all_end_time - all_start_time)
        return 0

    def self_read_cell_phase_info(self):
        self.cell_phase_list = self.get_cell_phase_info(os.path.join(self.cell_cycle_dir, self.cell_phase))
        return 0
    def self_read_GATC_fends(self,cell_base_info_dir):
        #
        self.GATC_fends_df = self.get_fends_info(os.path.join(cell_base_info_dir, "GATC.fends"))
        return 0
    def recursive_mkdir(self,dir):
        if not os.path.exists(dir):
            os.makedirs(dir)
    def store_variable_from_pikle_file(self,file_name,variable):
        with open(file_name,'wb') as f:
            pickle.dump(variable,f,protocol=-1)
    def load_variable_from_pikle_file(self,file_name):
        with open(file_name,'rb') as f:
            variable=pickle.load(f)
        return variable
    def get_chr_length_bin(self,resolution):
        chr_length_bin={}
        for chr_name in self.chr_name_list:
            chr_length = self.chr_length_dic[chr_name]
            if chr_length % resolution == 0:
                chr_length_bin[chr_name] = self.chr_length_dic[chr_name] // resolution
            else:
                chr_length_bin[chr_name] = self.chr_length_dic[chr_name] // resolution + 1
        return chr_length_bin
    def get_chr_length_bin_without_Y(self,resolution):
        chr_length_bin={}
        for chr_name,chr_length in self.chr_length_dic_without_Y.items():
            chr_length_bin[chr_name]=chr_length//resolution
        return chr_length_bin
    def get_index_content(self,index,a):
        return a[index]
    def get_neighbor_index(self,target_cell_name,combine_number):
        combine_cell_index_origin = np.array([i for i in range(-(combine_number // 2), combine_number // 2 + 1)])
        cell_number = len(self.cell_phase_list[0])
        cell_index = self.cell_phase_list[0].index(target_cell_name)
        combine_cell_index = (combine_cell_index_origin + cell_index) % cell_number
        combine_cell_name_list=[]
        for i in combine_cell_index:
            combine_cell_name_list.append(self.cell_phase_list[0][i])
        return combine_cell_index,combine_cell_name_list
    def get_CDD_vector(self,cell_name_list,acp_df,all_cdd_file):
        print("Start to get CDD")
        print("delete 649")
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
        # all_cdd_file = os.path.join(self.raw_single_cell_dir, r"fan_raw_cdd.txt")
        all_cdd.to_csv(all_cdd_file, sep='\t')
        return all_cdd

    def get_PCC_vector(self, cell_name_list,acp_df, all_pcc_file, fan_test=0):

        # all_pcc_file = os.path.join(self.raw_single_cell_dir, 'fan_raw_pcc.txt')

        temp_pair = self.pcc_df.columns
        temp_pair = temp_pair.to_series().reset_index(drop=True)

        def temp_fun_pcc(x):
            b = x.split('+')
            res = int(float(b[0]) * 1000)
            chr = b[1].strip('chr')
            bin1 = int(b[2])
            bin2 = int(b[3])
            return pd.Series([res, chr, bin1, bin2], index=['res', 'chr', 'bin1', 'bin2'])

        target_pair_df = temp_pair.apply(temp_fun_pcc)
        resolution = [100000, 500000, 1000000]
        target_pair_df_100kb = target_pair_df[target_pair_df['res'] == resolution[0]]
        target_pair_df_500kb = target_pair_df[target_pair_df['res'] == resolution[1]]
        target_pair_df_1000kb = target_pair_df[target_pair_df['res'] == resolution[2]]

        # cis_contact_acp = self.all_raw_acp[self.all_raw_acp['chr1'] == self.all_raw_acp['chr2']]

        cis_contact_acp = acp_df[acp_df['chr1'] == acp_df['chr2']]
        chr_name = []
        for i in range(1, 20):
            chr_name.append(str(i))
        chr_name.append('X')
        chr_cis_dic = {}
        for c in chr_name:
            each_chr_cis_dic = {}
            chr_cis = cis_contact_acp[cis_contact_acp['chr1'] == c]
            chr_cis_pos1_100kb = chr_cis['pos1'] // resolution[0]
            chr_cis_pos2_100kb = chr_cis['pos2'] // resolution[0]
            chr_cis_pos1_500kb = chr_cis['pos1'] // resolution[1]
            chr_cis_pos2_500kb = chr_cis['pos2'] // resolution[1]
            chr_cis_pos1_1000kb = chr_cis['pos1'] // resolution[2]
            chr_cis_pos2_1000kb = chr_cis['pos2'] // resolution[2]
            each_chr_cis_dic['chr_cis'] = chr_cis
            each_chr_cis_dic['chr_cis_pos1_100kb'] = chr_cis_pos1_100kb
            each_chr_cis_dic['chr_cis_pos2_100kb'] = chr_cis_pos2_100kb
            each_chr_cis_dic['chr_cis_pos1_500kb'] = chr_cis_pos1_500kb
            each_chr_cis_dic['chr_cis_pos2_500kb'] = chr_cis_pos2_500kb
            each_chr_cis_dic['chr_cis_pos1_1000kb'] = chr_cis_pos1_1000kb
            each_chr_cis_dic['chr_cis_pos2_1000kb'] = chr_cis_pos2_1000kb
            chr_cis_dic[c] = each_chr_cis_dic

        # fan_test = 0
        all_pcc = pd.DataFrame(index=cell_name_list)
        for index, row in target_pair_df_100kb.iterrows():
            #     print(index,row)
            print(index)
            temp_res = row['res']
            temp_chr = row['chr']
            temp_bin1 = row['bin1']
            temp_bin2 = row['bin2']

            chr_cis = chr_cis_dic[temp_chr]['chr_cis']
            chr_cis_pos1_100kb = chr_cis_dic[temp_chr]['chr_cis_pos1_100kb']
            chr_cis_pos2_100kb = chr_cis_dic[temp_chr]['chr_cis_pos2_100kb']

            result = chr_cis[((chr_cis_pos1_100kb == temp_bin1) & (chr_cis_pos2_100kb == temp_bin2)) | (
                    (chr_cis_pos1_100kb == temp_bin2) & (chr_cis_pos2_100kb == temp_bin1))]
            chr_cis_name_count = result.groupby('cell_name')['count'].sum()
            pair_name = '+'.join([str(float(temp_res / 1000)), 'chr' + temp_chr, str(temp_bin1), str(temp_bin2)])
            print(pair_name)
            all_pcc[pair_name] = chr_cis_name_count.apply(float)
            if (fan_test):
                break
        for index, row in target_pair_df_500kb.iterrows():
            #     print(index,row)
            print(index)
            temp_res = row['res']
            temp_chr = row['chr']
            temp_bin1 = row['bin1']
            temp_bin2 = row['bin2']

            chr_cis = chr_cis_dic[temp_chr]['chr_cis']
            chr_cis_pos1_500kb = chr_cis_dic[temp_chr]['chr_cis_pos1_500kb']
            chr_cis_pos2_500kb = chr_cis_dic[temp_chr]['chr_cis_pos2_500kb']

            result = chr_cis[((chr_cis_pos1_500kb == temp_bin1) & (chr_cis_pos2_500kb == temp_bin2)) | (
                    (chr_cis_pos1_500kb == temp_bin2) & (chr_cis_pos2_500kb == temp_bin1))]
            chr_cis_name_count = result.groupby('cell_name')['count'].sum()
            pair_name = '+'.join([str(float(temp_res / 1000)), 'chr' + temp_chr, str(temp_bin1), str(temp_bin2)])
            print(pair_name)
            all_pcc[pair_name] = chr_cis_name_count.apply(float)
            if (fan_test):
                break
        for index, row in target_pair_df_1000kb.iterrows():
            #     print(index,row)
            print(index)
            temp_res = row['res']
            temp_chr = row['chr']
            temp_bin1 = row['bin1']
            temp_bin2 = row['bin2']

            chr_cis = chr_cis_dic[temp_chr]['chr_cis']
            chr_cis_pos1_1000kb = chr_cis_dic[temp_chr]['chr_cis_pos1_1000kb']
            chr_cis_pos2_1000kb = chr_cis_dic[temp_chr]['chr_cis_pos2_1000kb']

            result = chr_cis[((chr_cis_pos1_1000kb == temp_bin1) & (chr_cis_pos2_1000kb == temp_bin2)) | (
                    (chr_cis_pos1_1000kb == temp_bin2) & (chr_cis_pos2_1000kb == temp_bin1))]
            chr_cis_name_count = result.groupby('cell_name')['count'].sum()
            pair_name = '+'.join([str(float(temp_res / 1000)), 'chr' + temp_chr, str(temp_bin1), str(temp_bin2)])
            print(pair_name)
            all_pcc[pair_name] = chr_cis_name_count.apply(float)
            if (fan_test):
                break
        all_pcc = all_pcc.fillna(0)

        temp_name = pd.Series(all_pcc.index, name='cell_nm')

        all_pcc.index = temp_name
        if not os.path.exists(all_pcc_file):
            all_pcc.to_csv(all_pcc_file, sep='\t')

        return all_pcc
    def get_MCM_vector(self,name_list,all_mcm_file):
        return 0
    def get_fan_raw_mcm(self):
        mcm_file = os.path.join(self.cell_cycle_dir, "MCM.txt")
        if not os.path.exists(mcm_file):
            mcm_df = pd.read_table(mcm_file, header=None, index_col=0)
            all_mcm = pd.DataFrame(index=self.cell_phase_list[0])
            all_mcm[1] = mcm_df[1]
            for i in range(len(mcm_df.columns)):
                col = i + 1
                all_mcm[col] = mcm_df[col]
            all_mcm_file = os.path.join(self.raw_single_cell_dir, "fan_raw_mcm.txt")
            all_mcm.to_csv(all_mcm_file, sep='\t', header=None)
        else:
            all_mcm=pd.read_table(mcm_file,header=None)
        return all_mcm
    def get_new_adj(self,raw_content):

        GATC_chr = self.GATC_fends_df['chr']
        GATC_pos = self.GATC_fends_df['coord']

        raw_df = pd.DataFrame(raw_content, columns=['fend1', 'fend2', 'count'])
        fend1_s = raw_df['fend1'].apply(int)
        fend2_s = raw_df['fend2'].apply(int)
        fend1_chr_name = list(fend1_s.apply(self.get_index_content, args=(GATC_chr,)))
        fend1_pos = list(fend1_s.apply(self.get_index_content, args=(GATC_pos,)))
        fend2_chr_name = list(fend2_s.apply(self.get_index_content, args=(GATC_chr,)))
        fend2_pos = list(fend2_s.apply(self.get_index_content, args=(GATC_pos,)))
        fend_pair_count = list(raw_df['count'].apply(int))
        end2 = time.time()
        print("time", end2 - start2)

        # fend1_chr_name=list(GATC_chr[int(temp[0]) for temp in raw_content])

        raw_dic = {'chrom1': fend1_chr_name, 'start1': fend1_pos, 'chrom2': fend2_chr_name, 'start2': fend2_pos,
                   'count': fend_pair_count}
        sc_contacts = pd.DataFrame(raw_dic)
        return  sc_contacts
    def adj_to_chr_position(self,adj_content):
        #return dic
        #{'chr1':[[pos1,pos2,count],...],...}
        chr_position={}
        for each_chr_name in self.chr_name_list:
            chr_position[each_chr_name]=[]

        GATC_chr_name=self.GATC_bed_list[0]
        # GATC_chr_pos=self.GATC_bed_list[1] to be compatiable with sample methods,use GATC_chr_pos
        GATC_chr_media_pos=self.GATC_bed_list[2]
        for each_pair in adj_content:
            pair_end1_chr=GATC_chr_name[int(each_pair[0])]
            pair_end2_chr = GATC_chr_name[int(each_pair[1])]
            pair_end1_pos=int(GATC_chr_media_pos[int(each_pair[0])])
            pair_end2_pos = int(GATC_chr_media_pos[int(each_pair[1])])
            if pair_end1_chr==pair_end2_chr:
                each_chr_position_count=[]
                each_chr_position_count.append(pair_end1_pos)
                each_chr_position_count.append(pair_end2_pos)
                each_chr_position_count.append(int(each_pair[2]))
                chr_position[pair_end1_chr].append(each_chr_position_count)
        for key in chr_position:
            temp=chr_position[key]
            i=0
            for i in range(len(temp)):
                line=temp[i]
                if int(line[0])>=int(line[1]):
                    temp_temp=line[1]
                    line[1]=line[0]
                    line[0]=temp_temp
                temp[i]=line
            chr_position[key]=sorted(temp,key=lambda s:(s[0],s[1]))
        return chr_position
    def convert_to_summary(raw_dir, GATC_bed):

        gatc_handle = open(GATC_bed, 'r')
        # chr1	2	3000536	HIC_chr1_1	0	+
        # chr1	3000536	3000801	HIC_chr1_2	0	+
        bed_to_position = {}
        for i in gatc_handle.readlines():
            temp = i.strip("\n").split("\t")
            index = temp[0] + "_" + temp[3].split("_")[2]

            bed_to_position[index] = int(int(temp[2]) - int(temp[1]) / 2)

        in_file = raw_dir + '/combine_adj'
        in_file_handle = open(in_file, 'r')

        first_chr = []
        second_chr = []
        pos1_list = []
        pos2_list = []
        count_list = []
        for line in in_file_handle.readlines():
            temp = line.strip('\n').split('\t')
            pos1_index = temp[0] + '_' + temp[1]
            pos2_index = temp[2] + '_' + temp[3]
            pos1 = bed_to_position[pos1_index]
            pos2 = bed_to_position[pos2_index]
            first_chr.append(temp[0])
            second_chr.append(temp[2])
            pos1_list.append(pos1)
            pos2_list.append(pos2)
            count_list.append(int(temp[4]))
        out_file = raw_dir + '/adj.summary'
        out_file_handle = open(out_file, 'w')
        for i in range(len(first_chr)):
            for j in range(count_list[i]):
                out_str = str(i) + '_' + str(j) + '\t' + first_chr[i] + '\t' + str(pos1_list[i]) + '\t+\t' + second_chr[
                    i] + '\t' + str(pos2_list[i]) + '\t-\n'
                out_file_handle.writelines(out_str)
        out_file_handle.close()



if __name__ == "__main__":

    cell_cycle_dir=r"E:\Users\scfan\data\CellCycle"
    compartment_tool_dir = r"E:\Users\scfan\program\2020_05_09\compartment"
    compartment_tool="CscoreTools"
    cell_phase="cell_phase"
    GATC_bed="GATC.bed"
    GATC_fends="GATC.fends"
    raw_single_cell_dir=os.path.join(cell_cycle_dir,"raw","scHiC2_process_author","single_cell")
    raw_combine_cell_dir=os.path.join(cell_cycle_dir,"raw","scHiC2_process_author","combine_cell")
    chr_length="chr_length"
    fragment_num="fragment_num"
    combine_mode="fixed_num"
    combine_num=20
    cell_phase_file=os.path.join(cell_cycle_dir,cell_phase)
    GATC_bed_file=os.path.join(cell_cycle_dir,GATC_bed)
    sim_single_reads_time=1
    sim_combine_reads_time=1

    sim1=simulation_framework_base_info(cell_cycle_dir,raw_single_cell_dir,
                             raw_combine_cell_dir,compartment_tool_dir,compartment_tool,cell_phase,GATC_bed,GATC_fends,chr_length,
                             fragment_num,combine_mode,combine_num,sim_single_reads_time,
                             sim_combine_reads_time)
    sim1.self_read_cell_phase_info()
    # sim1.test_get_PCC()
    sim1.simulate_process()
    # a=sim1.cell_phase_list

    # cell_phase_list=get_cell_phase_info(cell_phase_file)
    # traverse_cell_phase_for_converting_adj2chr_frag_adj(raw_single_cell_dir, cell_phase_list, GATC_bed_file)
    # traverse_cell_phase_for_combine_raw_chr_frag_adj(raw_single_cell_dir, raw_combine_cell_dir, cell_phase_list,
    #                                                  combine_mode, combine_num)
    #traverse_cell_phase_for_simulating_raw_chr_frag_adj(raw_combine_cell_dir,cell_phase_list,combine_mode,combine_num)

    print("fan")
    code_dir = r"E:\Users\scfan\program\simulation_project"
    sim_info.get_code_dir(code_dir)