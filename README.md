# Hi-CSim: a flexible simulator that generates high-fidelity single-cell Hi-C data for benchmarking

## Overview of Hi-CSim
Hi-CSim is a single-cell Hi-C simulator for generating high-fidelity data. As for the sparseness and heterogeneity of single-cell data, Hi-CSim merges neighboring cells to overcome the sparseness, samples interactions in distance-stratified chromosomes to maintain the heterogeneity of the single cells and estimates the empirical distribution of restriction fragments to generate simulated data. We verify that Hi-CSim generates high-fidelity data by comparing the performance of single-cell clustering and detection of chromosomal high-order structures with raw data. Furthermore, Hi-CSim is flexible to change the sequencing depth and the number of simulated replicates. Hi-CSim requires real single-cell Hi-C sequencing data (fragment-interaction format) as input along with user-defined simulation parameters.

**Hi-CSim workflow：**

![Hi-CSim pipeline diagram](/fig/01_workflow.png)

## Hi-CSim Usage
### 1. Preparation

    git clone https://github.com/zhanglabtools/Hi-CSim/

After git clone the repository completely, Hi-CSim is installed successfully. The repository includes an example in the data folder, consisting of  20 mouse embryonic stem cells(The cells are available at [https://github.com/tanaylab/schic2](https://github.com/tanaylab/schic2)). The default setting of Hi-CSim is ready to run the example once the environmental requirements meet. The requirements of Hi-CSim are as follows,
-   python(>= 3.7.4)
-   pandas(>= 0.25.1)
-   numpy(>= 1.16.5)
-   scipy(>= 1.3.1)
-   tqdm(>= 4.36.1)
-   seaborn(>= 0.9.0)
-   joblib(>=0.13.2)

The environment can also be quickly installed through python-requirements.txt file by running

		pip install -r python-requirements.txt

After installing the necessary environment, the instance consisting of 20 cells placed in data folder is easy to run by the following guidelines.



### 2. Setting the parameters in the "parameters.txt" file
#### 2.1 Parameters set for general folders

    1. python        : Path to the python interpreter(E.g., C:\Program Files\anaconda3\python.exe is a popular alternative path).
    1. work_dir      : Path to the Hi-CSim repository.
    2. src           : Path to the Hi-CSim src folder(E.g., work_dir\src).
    3. cell_base_info: Path to the folder contaning basic infomation of all cells, such as cell name, chromosome length and so on(E.g., work_dir\data\cell_base_info).
    4. raw_data      : Path to the folder contaning raw data of all cells(E.g., work_dir\data\raw_data).
    5. sim_data      : Path to the folder contaning simulated data of all cells(E.g., work_dir\data\sim_data).
    6. features      : Path to the folder contaning feature sets of raw data(E.g., work_dir\data\features).

#### 2.2 Parameters set for simulation: the numbers of fragmnet interactions and replicates
![flow_chart_designating_fragment_interaction_number](/fig/flow_chart_designating_fragment_interaction_number.png)
Flow chart showing the designation of fragment-interaction number.

    7. fragment_interaction_number_designating_mode: Two-value parameter(all_cell or each_cell) to determine how to designate the number of fragment interactions per cell.
       When assigned as all_cell, the fragment-interaction numbers of all cells will be specified together
       by parameter all_cell_seqDepthTime.
       each_cell_seqDepthTime is equal to all_cell_seqDepthTime.
       The basic fragment-interaction number of each simulated cell will be generated
       by stratified sampling according to the raw data distribution of library size.
       The final fragment-interaction number of each simulated cell is equal to
       the basic fragment-interaction number multiply all_cell_seqDepthTime.
       When assigned as each_cell, see the parameter each_cell_fragment_interaction_designating_mode description.
       The default value is all_cell.
    8. all_cell_seqDepthTime: Multiples of sequencing depth(E.g., 0.1,0.5,1 or 2 times sequencing depth.).
       When fragment_interaction_number_designating_mode is assigned as all_cell, all_cell_seqDepthTime will work. The default value is 1.
    9. each_cell_fragment_interaction_number_designating_mode: Two-value parameter(sequence_depth_time or fragment_interaction_number) to determine how to designate fragment-interaction number of each cell.
       When assigned as sequence_depth_time, a tab-separated file, named "each_cell_sequencing_depth_time.txt", should be provided in "cell_base_info" folder.
       each_cell_seqDepthTime will be read from "each_cell_sequencing_depth_time.txt".  
       The basic fragment-interaction number of each simulated cell will be generated
       by stratified sampling according to library size's raw data distribution.
       The final fragment-interaction number of each simulated cell is equal to the basic fragment-interaction number multiply each_cell_seqDepthTime.
       When assigned as fragment_interaction_number, a tab-separated file, named "each_cell_fragment_interaction_number.txt",
       should be provided in "cell_base_info" folder.
       each_cell_fragment_interaction_number will be read from "each_cell_fragment_interaction_number.txt".
       When fragment_interaction_number_designating_mode is assigned as each_cell, each_cell_fragment_interaction_number_designating_mode will work.
       The default value is sequence_depth_time.

![flow_chart_designating_replicates_number](/fig/flow_chart_designating_replicates_number.png)
Flow chart showing the designation of replicates' number.

    10. replicates_number_designating_mode: Two-value parameter(all_cell or each_cell) to determine how to designate the number of replicates per cell.
        When assigned as all_cell, the replicates numbers of all cells will be specified together by parameter all_cell_replicates_number.
        each_cell_replicates_number is equal to all_cell_replicates_number.
        When assigned as each_cell, a tab-separated file, named "each_cell_replicates_number.txt", should be provided in "cell_base_info" folder.
        each_cell_replicates_number will be read from "each_cell_replicates_number.txt".
        The default value is all_cell.
    11. all_cell_replicates_number: Number of replicates(E.g., 1, 2, 3 or 4 replicates per cell.).  
        When replicates_number_designating_mode is assigned as all_cell, all_cell_replicates_number will work.
        The default value is 1.

#### 2.3 Parameters set for simulation: Number of merged cells and others

    12. combineNumber: The number of merged cells. The default value is 20.
    13. step: The step size used when dividing chromosomes into different distances. The default value is 0.04.
    14. Bin_interval_number: Number of intervals when stratified sampling. The default value is 200.
    15. parallel: Two-value parameter(True or False) to determine whether to simulate in parallel. The default value is True.
    16. kernel_number: The number of GPU kernel used in simulating. When parallel is assigned as True, kernel_number will work. The default value is 24.
    17. filter_distance： The threshold of chromosomal distance used for filtering noisy signals. The part greater than the threshold will be denoised. The default value is 1000000.
    18. filter_value_percential: The percential of values used for filtering noisy signals. The fragment interactions whose chromosomal distance is more than _filter_distance_ and count number is less than _filter_value_percential_ will be filtered by controlling the simulated sequencing depth. The default value is 20.
### 3. Pre-processing
#### 3.1 Input-file format
A tab('\t') separated file, named _chr_pos_, that contains, on each line

`<chr1> <pos1> <chr2> <pos2> <count> <cell_name>`

* chr = chromosome (must be a chromosome in the genome)
* pos = position, the specific position of corresponding restriction fragment
* count =restriction fragment-interaction number
* cell_name= cell name corresponding to current file

#### 3.2 File conversion(optional)
Convert the _adj_ (fends-fends interaction) file to the _chr_pos_ file by running the following script:

    python convert_adj_to_chr_pos.py -p parameters.txt -f GATC.fends

GATC.fends is the projection file conveting fragment end(fend) to chromosome(chr) and coordinates(coord), placed in _cell_base_info_ directory (Due to the file size limitation of github, the file has been compressed into rar format, you need to decompress it before use). scHiC2 provides scripts and guidelines to generate _adj_ file([https://github.com/tanaylab/schic2](https://github.com/tanaylab/schic2)). The website also supplies Hi-C contact maps with processed _adj_ files.


#### 3.3 Extracting features
Construct features sets, PCC and CDD, by running the following script:

    python extract_features.py -p parameters.txt -b 10

The `-b` flag indicates the lower bound number of contacts used in extracting PCC. The bin will be filtered if the bin's total number of all cells is less than the lower bound. If "ValueError" occurs， please decrease the bound value.
The recommended value for the bound value under different cell numbers is as follows. Besides, if the cells' number is more than 2000, it is strongly recommended that the user should turn off the parallel by setting _parallel_ "False" to avoid memory overflow.

| cell number   | bound value   |
| ------------- |:-------------:|
|    20-100     |      1-10     |
|    100-500    |      10-30    |
|    500-1000   |      30-50    |
|    >1000      |      50       |



#### 3.4 Calculating the cell-cell distances
Calculate the cell-cell distances by using PCC and CDD, as following

    python calculate_cell_cell_distance.py -p parameters.txt -c 2

The script will use PCA(principal component analysis) to reduce the dimension of features' sets, PCC and CDD. The `-c` flag indicates the number of principal components used in calculating cell-cell distances. Then the cell-cell distances file, _cell_cell_distance.txt_, is generated and placed in _features_ directory. The user can also provide user-defined cell-to-cell distances, put it in the _features_ folder and name it _cell_cell_distance.txt_. For circular cell trajectories, it is recommended to use CIRCLET([https://github.com/zhanglabtools/CIRCLET](https://github.com/zhanglabtools/CIRCLET)) to construct the distance relationship between cells.
### 4. Simulating
Simulate cells according to _cell_name_list.txt_, as following

    python simulating.py -p parameters.txt

The simulated cells, named _chr_pos_, are placed in _sim_data_ folder.
### 5. Post-processing
#### 5.1 Merging the simulated cells

    python merge_cell.py -p parameters.txt -m data\merge_data\merge_cell_name_list.txt -i data\sim_data -o data\merge_data


#### 5.2 Converting  the chr_pos to bin_pairs file


    python convert_chr_pos_to_bin.py -p parameters.txt -i combine_data\chr_pos -o combine_data\bin_pairs  -r resolution

### Run time and Complexity
* The consumption of simulating the instance consisting of 20 cells is expected to less than 10 minutes with 1 cores CPU in a normal PC or server. The usage of time and memory for reference with a distinct number of cores are exhibited as follows. If _n_ represents the total number of fragment interactions, the simulation can be performed with _O(n)_.


![usage_of_time_and_memory](/fig/time_00.png)


![usage_of_time_and_memory](/fig/time_01.png)


![usage_of_time_and_memory](/fig/time_02.png)

### 6. Creating simulation with your own data
1. Follow section 1 "Preparation" to download Hi-CSim repository and install essential modules.
2. Download or prepare your own single-cell Hi-C sequencing data (chr_pos format). Put the single-cell Hi-C data in _raw_data_ folder determined in "parameter.txt" file. Each cell should be placed in a separate folder under _raw_data_. Then, as in the example, put the basic information of the data under the _cell_base_info_ folder, which contains _cell_name_list.txt_, _chr_length_, _fragment_num_. If you choose to independently specify the simulation information of each cell in "parameter.txt" file, you also need to provide an additional corresponding files, such as _each_cell_fragment_interaction_number.txt_, _each_cell_replicates_number.txt_ and _each_cell_replicates_number.txt_. All the above files are tab-separated.
3. Generate the _cell_cell_distance.txt_ file according to our pre-process guidelines and put it under the _features_ folder.
4. Designate folder's name of _sim_data_ in "parameter.txt" file for this run. After running _simulating.py_ script, the simualted cell will be placed in _sim_data_ folder. Note that the name of _sim_data_ should be different in order to avode the overlaps of results.
5. Create the _merger_cell_name_list.txt_ upder _merge_data_ folder. Then run _merge_cell.py_ script to generate merged Hi-C files for downstream analysis. We also supply a script, convert_chr_pos_to_bin.py, to convert the chr_pos file to bin_pars file.
