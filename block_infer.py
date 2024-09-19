import numpy as np
import pysam
import math
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from collections import defaultdict
import hdbscan
from multiprocess import Pool
import time
import warnings
warnings.filterwarnings("ignore")

np.seterr(divide='ignore',invalid="ignore")
pass
#%%
def read_bcf_file(vcf_file,vcf_index=None):
    if vcf_index != None:
        vcf = pysam.VariantFile(vcf_file,index_filename=vcf_index,mode="rb")
    else:
        vcf = pysam.VariantFile(vcf_file,mode="rb")
    return vcf

def break_contig(vcf_data,contig_name,block_size=100000,shift=100000):
    """
    Generator to return chunks of the VCF data from a single contig
    of a specified size. block_size is the size of each block, shift
    is how much we move over the starting point of each subsequent block
    
    shift must be <= block_size
    """
    cur_start = 0
    done = False
    
    full_data = list(vcf_data.fetch(contig_name))
    
    starting_index = 0
    
    while not done:        
            
        data_list = []

        for i in range(starting_index,len(full_data)):
            record = full_data[i]
            if record.pos >= cur_start and record.pos < cur_start+block_size:
                data_list.append(record)
                
                if record.pos < cur_start+shift:
                    starting_index += 1
            else:
                break
        cur_start += shift
        
        if starting_index >= len(full_data):
            done = True

        yield data_list

def gt_sum(gt_tuple):
    """
    Helper function to convert a variant call into a nice sum
    used in cleanup_block
    """
    
    if gt_tuple[0] == None or gt_tuple[1] == None:
        return np.nan
    else:
        return sum(gt_tuple)
    
def cleanup_block(block_list,min_frequency=0.1):
    """
    Turn a list of variant records site data into
    a list of site positions and a matrix of data
    """
    
    if len(block_list) == 0:
        return (np.array([]),np.array([]))
    
    cleaned_positions = []
    cleaned_list = []
    
    samples = block_list[0].samples
    
    for row in block_list:
            
        allele_freq = row.info.get("AF")[0]
        if allele_freq > 0 and allele_freq < 1 and allele_freq >= min_frequency:
            
            cleaned_positions.append(row.pos)
            
            row_vals = []
            
            for sample in samples:
                row_vals.append(gt_sum(row.samples.get(sample).get("GT")))
            
            cleaned_list.append(row_vals)
    return (cleaned_positions,np.ascontiguousarray(np.array(cleaned_list).transpose()))

def calc_distance(first_row,second_row,missing_penalty):
    """
    Calculate the L1 distance between two rows
    """
    diff_row = np.abs(first_row-second_row)
    abs_diff = np.abs(diff_row)
    abs_diff_sum = np.nansum(abs_diff)
    
    if missing_penalty == "None":
        return abs_diff_sum
    
    nan_penalty = 0
    
    first_nan = np.where(np.logical_and(np.isnan(first_row),~np.isnan(second_row)))[0]
    second_nan = np.where(np.logical_and(np.isnan(second_row),~np.isnan(first_row)))[0]
    both_nan = np.where(np.logical_and(np.isnan(first_row),np.isnan(second_row)))[0]
    
    nan_penalty += 2*len(both_nan)
    
    first_nan_pairs = first_row[second_nan]
    second_nan_pairs = second_row[first_nan]
    
    first_num_one = np.count_nonzero(first_nan_pairs == 1)
    second_num_one = np.count_nonzero(second_nan_pairs == 1)
    
    nan_penalty += 2*len(first_nan_pairs)-first_num_one
    nan_penalty += 2*len(second_nan_pairs)-second_num_one
    
    return abs_diff_sum+nan_penalty
    

def calc_distance_row(row,data_matrix,start_point,missing_penalty):
    """
    Calculate the distance between one row and all rows in a data 
    matrix starting from a given start_point index
    """
    
    num_samples = len(data_matrix)

    row_vals = [0] * start_point
    
    for i in range(start_point,num_samples):
        row_vals.append(calc_distance(row,data_matrix[i],missing_penalty))
    
    return row_vals
    
def generate_distance_matrix(genotype_array,missing_penalty="None"):
    
    num_samples = genotype_array.shape[0]
    
    genotype_copy = genotype_array.copy()

    
    processing_pool = Pool(processes=8)
    
    
    dist_matrix = []
    
    dist_matrix = processing_pool.starmap(
        lambda x,y : calc_distance_row(x,genotype_copy,y,missing_penalty),
        zip(genotype_array,range(num_samples)))
    
    #Convert to array and fill up lower diagonal
    dist_matrix = np.array(dist_matrix)
    dist_matrix = dist_matrix+dist_matrix.T-np.diag(dist_matrix.diagonal())

    del(processing_pool)
    
    return dist_matrix

#%%
def get_heterozygosity(genotype):
    """
    Calculate the heterozygosity of a genotype
    """
    num_sites = len(genotype)
    num_hetero = np.count_nonzero(genotype == 1)
    
    return 100*num_hetero/num_sites

def hap_distance(hap_one,hap_two):
    """
    Absolute distance between two haplotypes calculated
    via L1 metric
    """
    return np.sum(np.abs(hap_one-hap_two))

def hap_divergence(hap_one,hap_two):
    """
    Percentage of sites in two haplotypes that differ
    by allele
    """
    return 100*hap_distance(hap_one,hap_two)/len(hap_one)

def size_l1(vec):
    """
    Absolute distance between a vector and zero
    """
    return np.nansum(np.abs(vec))

def magnitude_percentage(vec):
    """
    A percentage measure of how far away a diploid is from zero in the 
    L1 metric compared to a vector of 2s. Can be bigger than 100%
    """
    return 100*size_l1(vec)/(2*np.count_nonzero(~np.isnan(vec)))

def perc_wrong(array):
    """
    Get the percentage of sites that are wrong (not between 0 and 1)
    for a haplotype
    """

    num_wrong = np.count_nonzero(np.logical_or(np.array(array) < 0,np.array(array) > 1))
    
    num_tot = len(array)
    
    return 100*num_wrong/num_tot

def fix_hap(haplotype):
    """
    Fix a haplotype by removing negative values and
    values bigger than 1 (setting them to 0 and 1
    respectively)
    
    haplotype must be a numpy float array
    """
    haplotype[haplotype < 0] = 0.0
    haplotype[haplotype > 1] = 1.0

    return haplotype

def hdbscan_cluster(dist_matrix,
                    min_cluster_size=2,
                    min_samples=1,
                    cluster_selection_method="eom",
                    alpha=1):
    def cluster_details(clustering,cut_distance,min_cluster_size):
        new_clustering = clustering.dbscan_clustering(cut_distance=cut_distance,min_cluster_size=min_cluster_size)

        new_labels = np.array(new_clustering)
        #new_probabilities = np.array(new_clustering.probabilities_)
        
        new_clusters = set(new_labels)
        new_clusters.discard(-1)
        
        
        #print(f"EPS: {cut_distance :.2f}, NUM OUTLIERS: {new_outlier_count}, NUM CLUSTERS: {new_num_clusters}")
        
        return new_labels


    #Create clustering object from sklearn
    base_clustering = hdbscan.HDBSCAN(metric="precomputed",
                                      min_cluster_size=min_cluster_size,
                                      min_samples=min_samples,
                                      cluster_selection_method=cluster_selection_method,
                                      alpha=alpha)
    
    
    #Fit data to clustering
    base_clustering.fit(dist_matrix)
    
    #Plot the condensed tree
    # color_palette = sns.color_palette('Paired', 20)
    # base_clustering.condensed_tree_.plot(select_clusters=True,selection_palette=color_palette)
    # plt.show()
    
    initial_labels = np.array(base_clustering.labels_)
    initial_probabilities = np.array(base_clustering.probabilities_)
    
    #Get outliers
    outliers = np.where(initial_labels == -1)[0]
    #print(f"Outliers: {outliers}")
    num_outliers = np.count_nonzero(initial_labels == -1)
    
    all_clusters = set(initial_labels)
    all_clusters.discard(-1)
    
    initial_num_clusters = len(all_clusters)
    
    return [initial_labels,initial_probabilities,base_clustering]

def fix_cluster_labels(c_labels):
    """
    Convert jumpbled up cluster labels into ones following
    ascending order as we move across the list of labels
    """
    seen_old = set([])
    new_mapping = {}
    cur_number = 0
    
    for i in range(len(c_labels)):
        test = c_labels[i]
        if test not in seen_old:
            seen_old.add(test)
            new_mapping[test] = cur_number
            cur_number += 1
    
    new_labels = list(map(lambda x: new_mapping[x],c_labels))
    
    return new_labels

def fill_nan(data_vals,clusters,cluster_probs=np.array([]),prob_cutoff=0.8,ambig_dist = 0.2):
    """
    Get representatives for each cluster as well as fill up NaN values in our clustering
    """
    
    if len(cluster_probs) == 0:
        cluster_probs = np.array([1]*data_vals.shape[0])

    cluster_names = set(clusters)
    cluster_representatives = {}
    raw_clusters = {}
    filled_clusters = {}
    
    
    #List which will store the data with NaN values filled in
    filled_data = [[]]*len(data_vals)
    
    #Remove the outliers
    cluster_names.discard(-1)
    
    
    cluster_array = np.array(clusters)
    
    data_array = np.array(data_vals)
    
    for cluster in cluster_names:
        
        #Find those indices which strongly map to the current cluster
        indices = np.where(np.logical_and(cluster_array == cluster,cluster_probs >= prob_cutoff))[0]

        #Pick out these indices and get the data for the cluster
        cluster_data = data_array[indices,:]
    
        
        #Calculate average value at each site for the samples which fall into the cluster
        site_sum = np.nansum(cluster_data,axis=0)
        site_counts = np.count_nonzero(~np.isnan(cluster_data),axis=0)
        site_avg = site_sum/site_counts
        np.nan_to_num(site_avg,copy=False)
        
        #Get representative haplotype for cluster, replacing nan values by 0
        site_rounded = np.round(site_avg)
        np.nan_to_num(site_rounded,copy=False)
        
        #Add representative for cluster to dictionary as well as add data for cluster to dictionary
        cluster_representatives[cluster] = site_rounded
        raw_clusters[cluster] = cluster_data
        
        #Find location of all NaN values in the cluster data
        missing = np.where(np.isnan(cluster_data))
        
        #Create a copy of the data for the cluster with NaN values filled in by the representative value for that site
        #Add this data to the filled clusters
        cluster_filled = np.copy(cluster_data)
        cluster_filled[missing] = np.take(site_rounded,missing[1])
        filled_clusters[cluster] = cluster_filled
        
        #Populate the filled data portion corresponding to this cluster
        for i in range(len(indices)):
            filled_data[indices[i]] = cluster_filled[i]
            

    #Fill in those rows of filled_data which weren't put into any cluster
    for i in range(len(filled_data)):
        if len(filled_data[i]) == 0:
            filled_data[i] = np.nan_to_num(data_vals[i],nan=0)

    #Return the results
    return (cluster_representatives,np.array(filled_data),raw_clusters,filled_clusters)

def combine_haplotypes(initial_haps,new_candidate_haps,unique_cutoff=5):
    """
    Takes two lists of haplotypes and creates a new dictionary of
    haplotypes containing all the first ones as well as those
    from the second which are at least unique_cutoff percent
    different from all of the ones in the first list/any of 
    those in the second list already chosen.
    
    Returns a dictionary detailing the new set of haplotypes
    as well as a dictionary showing where all the haps in the 
    second list map to in the new thing.

    """
    cur_haps = {}
    haps_len = len(initial_haps[0])
    new_haps_mapping = {-1:-1}
    
    cutoff = math.ceil(unique_cutoff*haps_len/100)
    i = 0
    j = 0
    
    for hap in initial_haps:
        cur_haps[i] = hap
        i += 1
    
    for hap in new_candidate_haps:
        add = True
        for k in range(len(cur_haps)):
            compare = cur_haps[k]
            num_differences = len(np.where(hap != compare)[0])
            if num_differences < cutoff:
                add = False
                new_haps_mapping[j] = k
                j += 1
                break
        if add:
            cur_haps[i] = hap
            new_haps_mapping[j] = i
            i += 1
            j += 1
    
    return (cur_haps,new_haps_mapping)

def get_initial_haps(genotype_array,
                     het_cutoff_start=10,
                     het_excess_add=6,
                     het_max_cutoff=20,
                     deeper_analysis=False,
                     deeper_tolerance=5,
                     make_pca=False,
                     verbose=False):
    """
    Get our initial haplotypes by finding high homozygosity samples
    """
    het_vals = np.array([get_heterozygosity(genotype_array[i]) for i in range(len(genotype_array))])
    
    found_homs = False
    cur_het_cutoff = het_cutoff_start
    
    while not found_homs:
        
        if cur_het_cutoff > het_max_cutoff:
            if verbose:
                print("Unable to find samples with high homozygosity in region")
            return ([],[])
        homs_where = np.where(het_vals <= cur_het_cutoff)[0]
    
        homs_array = genotype_array[homs_where]
        
        homs_array[homs_array == 1] = np.nan
    
        homs_array = homs_array/2
    
    
        dist_submatrix = generate_distance_matrix(
            homs_array,
            missing_penalty="None")
    
        initial_clusters = hdbscan_cluster(
                            dist_submatrix,
                            min_cluster_size=2,
                            min_samples=1,
                            cluster_selection_method="eom",
                            alpha=1.0)
        num_clusters = 1+np.max(initial_clusters[0])
        
        #If we don't find any clusters the increase the het threshold and repeat
        if num_clusters < 1:
            cur_het_cutoff += het_excess_add
            continue
        else:
            found_homs = True
        
        num_outliers = np.count_nonzero(initial_clusters[0] == -1)
    
    (representatives,filled_data,raw_clusters,filled_clusters) = fill_nan(
        homs_array,initial_clusters[0])
    
    pca_labels = initial_clusters[0]
    
    if deeper_analysis:
        second_clustering = hdbscan.HDBSCAN(metric="l1",
                            cluster_selection_method="eom",
                            min_cluster_size=2,
                            min_samples=1,
                            alpha=1.0,
                            allow_single_cluster=False)
        second_clustering.fit(filled_data)
        second_labels = np.array(second_clustering.labels_)

        (new_representatives,filled_data,new_raw,new_filled) = fill_nan(
            filled_data,second_labels)
        
        color_palette = sns.color_palette('Paired', 20)
        second_clustering.condensed_tree_.plot(select_clusters=True,selection_palette=color_palette)
        plt.show()
        
        initial_representatives = np.array(list(representatives.values()))
        second_representatives = np.array(list(new_representatives.values()))
        
        (final_representatives,label_mappings) = combine_haplotypes(
            initial_representatives,second_representatives,
            unique_cutoff=deeper_tolerance)
        
        final_labels = list(map(lambda x:label_mappings[x],second_labels))
        pca_labels = final_labels
    
    if make_pca:
        
        pca_plotting = PCA(n_components=2)
        
        standardised = StandardScaler().fit_transform(filled_data)


        pca_plot_proj = pca_plotting.fit_transform(standardised)
        
        print(pca_plot_proj.shape)

        color_palette = sns.color_palette('Paired', 20)
        cluster_colors = [(color_palette[x]) if x >= 0
                          else (0.5, 0.5, 0.5)
                          for x in pca_labels]
        
        fig,ax = plt.subplots()
        plt.scatter(*pca_plot_proj.T,c=cluster_colors)
        plt.title(f"PCA initial haps, Block size: {block_size}")
        fig.set_size_inches(8,6)
        plt.show()
        
    
    
    if deeper_analysis:
        use_reps =  final_representatives
    else:
        use_reps =  representatives
        
    return use_reps

def generate_further_haps(genotype_array,
                          initial_haps,
                          wrongness_cutoff=10,
                          uniqueness_threshold=10,
                          make_pca = False):
    """
    Given a genotype array and a set of initial haplotypes
    which are present in some of the samples of the array
    calculates other new haplotypes which are also present
    het_cutoff is the maximum percentage of sites which are not 0,1
    for a candidate hap to consider it further as a new haplotype,
    uniqueness_threshold is a percentage lower bound of how different
    a new candidate hap has to be from all the initial haps to be considered
    further.
    """
    candidate_haps = []
    initial_list = np.array(list(initial_haps.values()))
    
    for geno in genotype_array:
        for init_hap in initial_list:
            diff = geno-init_hap
            

            wrongness = perc_wrong(diff)
            
            if wrongness <= wrongness_cutoff:
                fixed_diff = fix_hap(diff)
                
                add = True
                for comp_hap in initial_list:
                    
                    
                    perc_diff = hap_divergence(fixed_diff,comp_hap)
                    if perc_diff < uniqueness_threshold:
                        add = False
                        break
                
                if add:
                    candidate_haps.append(fixed_diff)
    
    candidate_haps = np.array(candidate_haps)
    dist_submatrix = generate_distance_matrix(
        candidate_haps,
        missing_penalty="None")

    initial_clusters = hdbscan_cluster(
                        dist_submatrix,
                        min_cluster_size=len(initial_haps)+1,
                        min_samples=1,
                        cluster_selection_method="eom",
                        alpha=1.0)
    
    (representatives,filled_data,raw_clusters,filled_clusters) = fill_nan(
        candidate_haps,initial_clusters[0])
    
    new_reps_list = np.array(list(representatives.values()))
    
    pca_labels = initial_clusters[0]
    
    if make_pca:
        
        pca_plotting = PCA(n_components=2)
        
        standardised = StandardScaler().fit_transform(filled_data)


        pca_plot_proj = pca_plotting.fit_transform(standardised)

        color_palette = sns.color_palette('Paired', 20)
        cluster_colors = [(color_palette[x]) if x >= 0
                          else (0.5, 0.5, 0.5)
                          for x in pca_labels]
        
        fig,ax = plt.subplots()
        plt.scatter(*pca_plot_proj.T,c=cluster_colors)
        plt.title(f"PCA extra haps, Block size: {block_size}")
        fig.set_size_inches(8,6)
        plt.show()
        
    final_haps = combine_haplotypes(initial_list,new_reps_list,unique_cutoff=uniqueness_threshold)[0]
    
    
    
    return final_haps

def match_best(haps_dict,diploids):
    """
    Find the best matches of a pair of haploids for each diploid in the diploid list
    """
    
    dips_matches = []
    haps_usage = {}
    errs = []
    
    for i in range(len(diploids)):
        cur_best = (None,None)
        cur_div = np.inf
        
        for j in haps_dict.keys():
            for k in haps_dict.keys():
                difference = diploids[i]-haps_dict[j]-haps_dict[k]
                div = magnitude_percentage(difference)
                
                if div < cur_div:
                    cur_div = div
                    cur_best = (j,k)
        for index in cur_best:
            if index not in haps_usage.keys():
                haps_usage[index] = 0
            haps_usage[index] += 1
        errs.append(cur_div)
        dips_matches.append((cur_best,cur_div))
    
    return (dips_matches,haps_usage,np.array(errs))

#%%

def generate_haplotypes_all(chromosome_data):
    haps = []
    
    for i in range(len(chromosome_data)):
        print(i,len(chromosome_data))
        
        haps.append(generate_haplotypes_block(chromosome_data[i]))
    
    return haps
#%%
bcf = read_bcf_file("./fish_vcf/AsAc.AulStuGenome.biallelic.bcf.gz")
contigs = bcf.header.contigs
names = bcf.header.samples
#%%
block_size = 100000
chr1 = list(break_contig(bcf,"chr1",block_size=block_size,shift=block_size))

#%%
def generate_haplotypes_block(block_data,
                              error_reduction_cutoff = 0.98,
                              max_cutoff_error_increase = 1.05,
                              max_hapfind_iter=5,
                              make_pca=False,
                              deeper_analysis_initial=False):
    """
    Given a block of sample data generates the haplotypes that make up
    the samples present
    
    """
    
    (positions,genotype_array) = cleanup_block(block_data)
    dist_matrix = generate_distance_matrix(
            genotype_array,
            missing_penalty="None")
    
    initial_haps = get_initial_haps(genotype_array,make_pca=make_pca,deeper_analysis=deeper_analysis_initial)
    initial_matches = match_best(initial_haps,genotype_array)
    initial_error = np.mean(initial_matches[2])
    
    matches_history = [initial_matches]
    errors_history = [initial_error]
    haps_history = [initial_haps]
    
    all_found = False
    cur_haps = initial_haps
    
    print(initial_error)
    
    while not all_found:
        cur_haps = generate_further_haps(genotype_array,cur_haps)
        cur_matches = match_best(cur_haps,genotype_array)
        cur_error = np.mean(cur_matches[2])
        
        # matches_history.append(cur_matches)
        # errors_history.append(cur_error)
        # haps_history.append(cur_haps)
        
        print(cur_error,len(cur_haps))
        
        
        if cur_error/errors_history[-1] >= error_reduction_cutoff and len(errors_history) > 2:
            all_found = True
            break
        if len(errors_history) > max_hapfind_iter+1:
            all_found = True
            
        matches_history.append(cur_matches)
        errors_history.append(cur_error)
        haps_history.append(cur_haps)
    
    print(errors_history)
    print("Pre truncation len",len(cur_haps))
    
    truncated_haps = truncate_haps(haps_history[-1],matches_history[-1],genotype_array,
                                   max_cutoff_error_increase=max_cutoff_error_increase)
        
        
    return truncated_haps

def get_removal_statistics(candidate_haps,candidate_matches,removal_value,genotype_array):
    """
    Remove one hap from our list of candidate haps and see how much worse
    of a total fit we get.
    """
    truncated_haps = candidate_haps.copy()
    truncated_haps.pop(removal_value)
    truncated_matches = match_best(truncated_haps,genotype_array)
    
    truncated_mean = np.mean(truncated_matches[2])
    truncated_max = np.max(truncated_matches[2])
    truncated_std = np.std(truncated_matches[2])
    
    old_mean = np.mean(candidate_matches[2])
    old_max = np.max(candidate_matches[2])
    old_std = np.std(candidate_matches[2])
    
    return (truncated_mean,truncated_max,truncated_std,truncated_matches)

def truncate_haps(candidate_haps,candidate_matches,genotype_array,
                  max_cutoff_error_increase=1.1):
    """
    Truncate a list of haplotypes so that only the necessary ones remain
    """
    processing_pool = Pool(processes=8)
    
    cand_copy = candidate_haps.copy()
    cand_matches = candidate_matches
    used_haps = cand_matches[1].keys()
    
    starting_error = np.mean(cand_matches[2])
    starting_error_max = np.max(cand_matches[2])
    
    
    for hap in list(cand_copy.keys()):
        if hap not in used_haps:
            cand_copy.pop(hap)
            
    haps_names = list(cand_copy.keys())
    
    errors_limit = max_cutoff_error_increase*starting_error
    
    truncation_complete = False
    
    while not truncation_complete:
        removal_indicators = processing_pool.starmap(lambda x:
                                                 get_removal_statistics(cand_copy,
                                                                        cand_matches,x,
                                                                        genotype_array),
                                                 zip(haps_names))
        
        smallest_value = min(removal_indicators,key=lambda x:x[0])
        smallest_index = removal_indicators.index(smallest_value)
        
        if removal_indicators[smallest_index][0] > errors_limit:
            truncation_complete = True
        else:
            hap_name = haps_names[smallest_index]
            cand_copy.pop(hap_name)
            cand_matches = smallest_value[3]
            errors_limit = max_cutoff_error_increase*smallest_value[0]
            haps_names = list(cand_copy.keys())
            
    final_haps = {}
    i = 0
    for j in cand_copy.keys():
        final_haps[i] = cand_copy[j]
        i += 1
        
    final_matches = match_best(final_haps,genotype_array)
    
    print(np.mean(final_matches[2]))
    
    return final_haps
    
#%%
st= time.time()
test = chr1[73]
test_haps = generate_haplotypes_block(test)
print(time.time()-st)
#%%