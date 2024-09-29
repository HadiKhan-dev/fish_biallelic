import numpy as np
import pysam
import math
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
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
        
        if starting_index >= len(full_data):
            done = True

        yield (data_list,(cur_start,cur_start+block_size))
        
        cur_start += shift

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
    
    keep_flags = []
    cleaned_positions = []
    cleaned_list = []
    
    samples = block_list[0].samples
    
    for row in block_list:
            
        allele_freq = row.info.get("AF")[0]
        
        if allele_freq >= min_frequency and allele_freq <= 1-min_frequency:
            keep_flags.append(1)
        else:
            keep_flags.append(0)
        
        
        cleaned_positions.append(row.pos)
            
        row_vals = []
            
        for sample in samples:
            row_vals.append(gt_sum(row.samples.get(sample).get("GT")))
            
        cleaned_list.append(row_vals)
    return (cleaned_positions,np.array(keep_flags),np.ascontiguousarray(np.array(cleaned_list).transpose()))

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
    
def generate_distance_matrix(genotype_array,
                             keep_flags=None,
                             missing_penalty="None"):
    
    if keep_flags is None:
        keep_flags = np.array([True for _ in range(genotype_array.shape[1])])
    
    if keep_flags.dtype != bool:
        keep_flags = np.array(keep_flags,dtype=bool)
          
    genotype_array = genotype_array[:,keep_flags]
    
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
def get_heterozygosity(genotype,keep_flags=None):
    """
    Calculate the heterozygosity of a genotype
    """
    if keep_flags is None:
        keep_flags = np.array([True for _ in range(len(genotype))])
        
    if keep_flags.dtype != bool:
        keep_flags = np.array(keep_flags,dtype=bool)
        
    
    gt_use = np.array(genotype)
    gt_use = gt_use[keep_flags]
    
    num_sites = len(gt_use)
    num_hetero = np.count_nonzero(gt_use == 1)
    
    return 100*num_hetero/num_sites

def hap_distance(hap_one,hap_two,keep_flags=None):
    """
    Absolute distance between two haplotypes calculated
    via L1 metric, keep_flags is a boolean mask of site
    to calculate the distance over
    """
    if keep_flags is None:
        keep_flags = np.array([True for _ in range(len(hap_one))])
        
    if keep_flags.dtype != bool:
        keep_flags = np.array(keep_flags,dtype=bool)
        
    one_keep = hap_one[keep_flags]
    two_keep = hap_two[keep_flags]
        
    return np.sum(np.abs(one_keep-two_keep))

def hap_divergence(hap_one,hap_two,keep_flags=None):
    """
    Percentage of sites in two haplotypes that differ
    by allele
    """
    if keep_flags is None:
        keep_flags = np.array([True for _ in range(len(hap_one))])
        
    if keep_flags.dtype != bool:
        keep_flags = np.array(keep_flags,dtype=bool)
        
    num_keep = np.count_nonzero(keep_flags)
        
    return 100*hap_distance(hap_one,hap_two,keep_flags=keep_flags)/num_keep

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
    if np.count_nonzero(~np.isnan(vec)) == 0:
        return 0
    return 100*size_l1(vec)/(2*np.count_nonzero(~np.isnan(vec)))

def perc_wrong(haplotype,keep_flags=None):
    """
    Get the percentage of sites that are wrong (not between 0 and 1)
    for a haplotype
    """
    if keep_flags is None:
        keep_flags = np.array([True for _ in range(len(haplotype))])
        
    if keep_flags.dtype != bool:
        keep_flags = np.array(keep_flags,dtype=bool)
        
    hap_use = haplotype[keep_flags]

    num_wrong = np.count_nonzero(np.logical_or(np.array(hap_use) < 0,np.array(hap_use) > 1))
    
    num_tot = len(hap_use)
    
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
                    alpha=1,
                    allow_single_cluster=False):

    #Create clustering object from sklearn
    base_clustering = hdbscan.HDBSCAN(metric="precomputed",
                                      min_cluster_size=min_cluster_size,
                                      min_samples=min_samples,
                                      cluster_selection_method=cluster_selection_method,
                                      alpha=alpha,allow_single_cluster=allow_single_cluster)
    
    
    #Fit data to clustering
    base_clustering.fit(dist_matrix)
    
    #Plot the condensed tree
    # color_palette = sns.color_palette('Paired', 20)
    # base_clustering.condensed_tree_.plot(select_clusters=True,selection_palette=color_palette)
    # plt.show()
    
    initial_labels = np.array(base_clustering.labels_)
    initial_probabilities = np.array(base_clustering.probabilities_)
    
    all_clusters = set(initial_labels)
    all_clusters.discard(-1)
    
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

def match_best(haps_dict,diploids,keep_flags=None):
    """
    Find the best matches of a pair of haploids for each diploid in the diploid list
    """
    
    dip_length = len(diploids[0])
    
    if keep_flags is None:
        keep_flags = np.array([True for _ in range(dip_length)])
    
    if keep_flags.dtype != bool:
        keep_flags = np.array(keep_flags,dtype=bool)
        
    haps_dict = {x:haps_dict[x][keep_flags] for x in haps_dict.keys()}
    diploids = diploids[:,keep_flags]
    
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

def combine_haplotypes(initial_haps,
                       new_candidate_haps,
                       keep_flags=None,
                       usages=None,
                       unique_cutoff=5,
                       max_hap_add=1000):
    """
    Takes two dictionaries of haplotypes and a list of new potential
    haptotype and creates a new dictionary of haplotypes containing
    all the first ones as well as those from the second which are at
    least unique_cutoff percent different from all of the
    ones in the first list/any of those in the second list already chosen.
    
    usages is an optional parameter which is an indicator of 
    how often a new candidate haplotype is used in the generated
    haplotype list. It works with max_hap_add to ensure that if
    we have a limit on the maximum number of added haplotypes 
    we preferentially add the haplotypes that have highest usage
    
    Returns a dictionary detailing the new set of haplotypes
    as well as a dictionary showing where all the haps in the 
    second list map to in the new thing.

    """
    
    
    for x in initial_haps.keys(): #Get the legth of a haplotype
        haplotype_length = len(initial_haps[x])
        break
    
    if keep_flags is None:
        keep_flags = np.array([True for _ in range(haplotype_length)])
    
    if keep_flags.dtype != bool:
        keep_flags = np.array(keep_flags,dtype=bool)
    
    keep_length = np.count_nonzero(keep_flags)
    
    new_haps_mapping = {-1:-1}
    
    if usages == None:
        usages = {x:1 for x in new_candidate_haps.keys()}
    usages = {k: v for k, v in sorted(usages.items(), key=lambda item: item[1])} #Get a sorted version of the usage dictionary
    
    combined_dict = {x:(new_candidate_haps[x],usages[x]) for x in usages.keys()}
    
    cutoff = math.ceil(unique_cutoff*keep_length/100)
    
    i = 0
    j = 0
    num_added = 0
    
    cur_haps = {}
    for idx in initial_haps.keys():
        cur_haps[i] = initial_haps[idx]
        i += 1
    
    for identifier in combined_dict.keys():
        (hap,hap_usage) = combined_dict[identifier]
        hap_keep = hap[keep_flags]
        add = True
        for k in range(len(cur_haps)):
            compare = cur_haps[k]
            compare_keep = compare[keep_flags]
            num_differences = len(np.where(hap_keep != compare_keep)[0])
            if num_differences < cutoff:
                add = False
                new_haps_mapping[j] = k
                j += 1
                break
        if add and num_added < max_hap_add:
            cur_haps[i] = hap
            new_haps_mapping[j] = i
            i += 1
            j += 1
            num_added += 1
        if num_added >= max_hap_add:
            break
    
    return (cur_haps,new_haps_mapping)

def get_addition_statistics(starting_haps,
                            candidate_haps,
                            addition_index,
                            genotype_array,
                            keep_flags = None):
    """
    Add one hap from our list of candidate haps and see how much worse
    of a total fit we get.
    """
    
    for x in initial_haps.keys():
        orig_hap_len = len(starting_haps[x])
        break
    
    if keep_flags is None:
        keep_flags = np.array([True for _ in range(orig_hap_len)])
    
    if keep_flags.dtype != bool:
        keep_flags = np.array(keep_flags,dtype=bool)
        
    added_haps = starting_haps.copy()
    adding_name = max(added_haps.keys())+1
    
    added_haps[adding_name] = candidate_haps[addition_index]
    
    added_matches = match_best(added_haps,genotype_array,keep_flags=keep_flags)
    
    added_mean = np.mean(added_matches[2])
    added_max = np.max(added_matches[2])
    added_std = np.std(added_matches[2])
    
    return (added_mean,added_max,added_std,added_matches)


def combine_haplotypes_smart(initial_haps,
                        new_candidate_haps,
                        genotype_array,
                        keep_flags=None,
                        loss_reduction_cutoff_ratio =0.98):
    """
    Takes two lists of haplotypes and creates a new dictionary of
    haplotypes containing all the first ones as well as those
    from the second which are at least unique_cutoff percent
    different from all of the ones in the first list/any of 
    those in the second list already chosen.
    
    Returns a dictionary detailing the new set of haplotypes
    as well as a dictionary showing where all the haps in the 
    second list map to in the new thing.
    
    Alternate method to combine_haplotype that smartly looks at 
    which candidate hap will reduce the mean error the most, adds
    that to the list and continues until the reduction is too small
    to matter
    
    Unlike combine_haplotype not all candidate haplotypes get mapped,
    so there is no new_haps_mapping being returned

    """
    
    processing_pool = Pool(processes=8)
    
    for x in initial_haps.keys():
        orig_hap_len = len(initial_haps[x])
        break
    
    if keep_flags is None:
        keep_flags = np.array([True for _ in range(orig_hap_len)])
    
    if keep_flags.dtype != bool:
        keep_flags = np.array(keep_flags,dtype=bool)
    
    i = 0
    cur_haps = {}
    for idx in initial_haps.keys():
        cur_haps[i] = initial_haps[idx]
        i += 1
        
    cur_matches = match_best(cur_haps,genotype_array,keep_flags=keep_flags)
    cur_error = np.mean(cur_matches[2])
    
    
    candidate_haps = new_candidate_haps.copy()
    
    addition_complete = False
    
    while not addition_complete:
        cand_keys = list(candidate_haps.keys())
        addition_indicators = processing_pool.starmap(lambda x:
                            get_addition_statistics(cur_haps,
                            candidate_haps,x,
                            genotype_array,keep_flags=keep_flags),
                            zip(cand_keys))            
        
        smallest_result = min(addition_indicators,key=lambda x:x[0])
        smallest_index = addition_indicators.index(smallest_result)
        smallest_name = cand_keys[smallest_index]
        smallest_value = smallest_result[0]
        
        if smallest_value/cur_error < loss_reduction_cutoff_ratio:
            new_index = max(cur_haps.keys())+1
            cur_haps[new_index] = candidate_haps[smallest_name]
            candidate_haps.pop(smallest_name)
            
            cur_matches = match_best(cur_haps,genotype_array)
            cur_error = np.mean(cur_matches[2])
            
            
        else:
            addition_complete = True
        
        if len(candidate_haps) == 0:
            addition_complete = True
            
    final_haps = {}
    i = 0
    for idx in cur_haps.keys():
        final_haps[i] = cur_haps[idx]
        i += 1
    
    return final_haps

def get_initial_haps(genotype_array,
                     keep_flags=None,
                     het_cutoff_start=10,
                     het_excess_add=2,
                     het_max_cutoff=20,
                     deeper_analysis=False,
                     uniqueness_tolerance=5,
                     make_pca=False,
                     verbose=False):
    """
    Get our initial haplotypes by finding high homozygosity samples
    
    keep_flags is a optional boolean array which is 1 at those sites which
    we wish to consider for purposes of the analysis and 0 elsewhere,
    if not provided we use all sites
    """
    if keep_flags is None:
        keep_flags = np.array([True for _ in range(genotype_array.shape[1])])
    
    if keep_flags.dtype != bool:
        keep_flags = np.array(keep_flags,dtype=bool)

    het_vals = np.array([get_heterozygosity(genotype_array[i],keep_flags) for i in range(len(genotype_array))])
    
    found_homs = False
    cur_het_cutoff = het_cutoff_start
    
    accept_singleton = (het_cutoff_start+het_max_cutoff)/2
    
    while not found_homs:
        
        if cur_het_cutoff > het_max_cutoff:
            if verbose:
                print("Unable to find samples with high homozygosity in region")
            return {}
        homs_where = np.where(het_vals <= cur_het_cutoff)[0]
    
        homs_array = genotype_array[homs_where]
        
        if len(homs_array) < 5:
            cur_het_cutoff += het_excess_add
            continue
        
        homs_array[homs_array == 1] = np.nan
    
        homs_array = homs_array/2
        
    
        dist_submatrix = generate_distance_matrix(
            homs_array,keep_flags=keep_flags,
            missing_penalty="None")
        
        #First do clustering looking for at least 2 clusters, if that fails rerun allowing single clusters
        try:
            initial_clusters = hdbscan_cluster(
                                dist_submatrix,
                                min_cluster_size=2,
                                min_samples=1,
                                cluster_selection_method="eom",
                                alpha=1.0)
            num_clusters = 1+np.max(initial_clusters[0])
            
            if num_clusters == 0:
                assert False
        except:
            if cur_het_cutoff < accept_singleton:
                cur_het_cutoff += het_excess_add
                continue
            initial_clusters = hdbscan_cluster(
                                dist_submatrix,
                                min_cluster_size=2,
                                min_samples=1,
                                cluster_selection_method="eom",
                                alpha=1.0,
                                allow_single_cluster=True)
            num_clusters = 1+np.max(initial_clusters[0])

        
        #If we don't find any clusters the increase the het threshold and repeat
        if num_clusters < 1:
            cur_het_cutoff += het_excess_add
            continue
        else:
            found_homs = True
    
    (representatives,
     filled_data,
     raw_clusters,
     filled_clusters) = fill_nan(
        homs_array,initial_clusters[0])
    
    
    #Remove any haps that are too close to each other
    (representatives,
     label_mappings) = combine_haplotypes(
             {},representatives,keep_flags=keep_flags,
             unique_cutoff=uniqueness_tolerance)
    
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

        (new_representatives,filled_data,
         new_raw,new_filled) = fill_nan(
            filled_data,second_labels)
        
        (final_representatives,
         label_mappings) = combine_haplotypes(
            representatives,new_representatives,
            keep_flags=keep_flags,
            unique_cutoff=uniqueness_tolerance)
        
        final_labels = list(map(lambda x:label_mappings[x],second_labels))
        pca_labels = final_labels
    
    
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
                          keep_flags=None,
                          wrongness_cutoff=10,
                          uniqueness_threshold=5,
                          max_hap_add = 1000,
                          min_perc_usage = 5,
                          make_pca = False,
                          verbose=False):
    """
    Given a genotype array and a set of initial haplotypes
    which are present in some of the samples of the array
    calculates other new haplotypes which are also present.
    
    het_cutoff is the maximum percentage of sites which are not 0,1
    for a candidate hap to consider it further as a new haplotype
    
    uniqueness_threshold is a percentage lower bound of how different
    a new candidate hap has to be from all the initial haps to be considered
    further.
    
    max_hap_add is the maximum number of additional haplotypes to add
    
    min_perc_usage is the minimum percentage of derived samples that
    must fall into a cluster for its representative to be considered
    as a potential new haplotype
    """
    if keep_flags is None:
        keep_flags = np.array([True for _ in range(genotype_array.shape[1])])
    
    if keep_flags.dtype != bool:
        keep_flags = np.array(keep_flags,dtype=bool)
        
    candidate_haps = []
    initial_list = np.array(list(initial_haps.values()))
    
    for geno in genotype_array:
        for init_hap in initial_list:
            diff = geno-init_hap

            wrongness = perc_wrong(diff,keep_flags=keep_flags)
            
            if wrongness <= wrongness_cutoff:
                fixed_diff = fix_hap(diff)
                
                add = True
                for comp_hap in initial_list:
                    
                    
                    perc_diff = hap_divergence(fixed_diff,comp_hap,keep_flags=keep_flags)
                    if perc_diff < uniqueness_threshold:
                        add = False
                        break
                
                if add:
                    candidate_haps.append(fixed_diff)
    
    candidate_haps = np.array(candidate_haps)
    
    if len(candidate_haps) == 0:
        if verbose:
            print("Unable to find candidate haplotypes when generating further haps")
        return initial_haps
    
    dist_submatrix = generate_distance_matrix(
        candidate_haps,keep_flags=keep_flags,
        missing_penalty="None")

    initial_clusters = hdbscan_cluster(
                        dist_submatrix,
                        min_cluster_size=len(initial_haps)+1,
                        min_samples=1,
                        cluster_selection_method="eom",
                        alpha=1.0)
    
    (representatives,
     filled_data,
     raw_clusters,
     filled_clusters) = fill_nan(
        candidate_haps,initial_clusters[0])
    
    
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
        

    final_haps = combine_haplotypes_smart(initial_haps,
                representatives,genotype_array,
                keep_flags=keep_flags)
    
    return final_haps

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
    
    for hap in list(cand_copy.keys()):
        if hap not in used_haps:
            cand_copy.pop(hap)
            
    haps_names = list(cand_copy.keys())
    
    errors_limit = max_cutoff_error_increase*starting_error
    
    truncation_complete = False
    
    while not truncation_complete:
        removal_indicators = processing_pool.starmap(lambda x:
                            get_removal_statistics(cand_copy,
                            cand_matches,x,genotype_array),
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
    
    return final_haps

def generate_haplotypes_block(block_data,
                              error_reduction_cutoff = 0.98,
                              max_cutoff_error_increase = 1.02,
                              max_hapfind_iter=5,
                              make_pca=False,
                              deeper_analysis_initial=False,
                              min_num_haps=0):
    """
    Given a block of sample data generates the haplotypes that make up
    the samples present
    
    min_num_haps is a (soft) minimum value for the number of haplotypes,
    if we have fewer than that many haps we iterate further to get more 
    haps.
    
    
    """
    
    (positions,keep_flags,genotype_array) = cleanup_block(block_data)
    
    initial_haps = get_initial_haps(genotype_array,keep_flags,
                                    make_pca=make_pca,
                                    deeper_analysis=deeper_analysis_initial)
    
    initial_matches = match_best(initial_haps,genotype_array)
    initial_error = np.mean(initial_matches[2])
    
    matches_history = [initial_matches]
    errors_history = [initial_error]
    haps_history = [initial_haps]
    
    all_found = False
    cur_haps = initial_haps
    
    minimum_strikes = 0 #Counter that increases every time we get fewer than the required minimum number of haplotypes, if it hits 3 we break out of our loop to find further haps
    striking_up = False
    
    uniqueness_threshold=5
    wrongness_cutoff = 10
    
    while not all_found:
        cur_haps = generate_further_haps(genotype_array,
                    cur_haps,uniqueness_threshold=uniqueness_threshold,
                    wrongness_cutoff=wrongness_cutoff,
                    min_perc_usage=0.0)
        cur_matches = match_best(cur_haps,genotype_array)
        cur_error = np.mean(cur_matches[2])
        
        # matches_history.append(cur_matches)
        # errors_history.append(cur_error)
        # haps_history.append(cur_haps)
        
        if cur_error/errors_history[-1] >= error_reduction_cutoff and len(errors_history) >= 2:
            if len(cur_haps) >= min_num_haps or minimum_strikes >= 3:
                all_found = True
                break
            else:
                minimum_strikes += 1
                uniqueness_threshold -= 1
                wrongness_cutoff += 2
                striking_up = True
                
        if len(cur_haps) == len(haps_history[-1]) and not striking_up: #Break if in the last iteration we didn't find a single new hap
            all_found = True
            break
            
        if len(errors_history) > max_hapfind_iter+1:
            all_found = True
            
        matches_history.append(cur_matches)
        errors_history.append(cur_error)
        haps_history.append(cur_haps)
        
        striking_up = False
    
    truncated_haps = truncate_haps(haps_history[-1],matches_history[-1],genotype_array,
                                   max_cutoff_error_increase=max_cutoff_error_increase)
        
    return truncated_haps

#%%

def generate_haplotypes_all(chromosome_data):
    haps = []
    
    for i in range(len(chromosome_data)):
        print(i,chromosome_data[i][1])
        
        found_haps = generate_haplotypes_block(chromosome_data[i][0])
        print(len(found_haps))
        print()
        haps.append(found_haps)
    
    return haps
#%%
bcf = read_bcf_file("./fish_vcf/AsAc.AulStuGenome.biallelic.bcf.gz")
contigs = bcf.header.contigs
names = bcf.header.samples
#%%
block_size = 100000
shift_size = 50000
chr1 = list(break_contig(bcf,"chr1",block_size=block_size,shift=shift_size))

#%%
full_haps = generate_haplotypes_all(chr1)    
#%%
st= time.time()
test = chr1[189][0]
test_haps = generate_haplotypes_block(test,
            deeper_analysis_initial=False,
            min_num_haps=6,
            max_hapfind_iter=10)
print("Number haps found:",len(test_haps))
print("Time:",time.time()-st)
#%%
test = chr1[189][0]
(positions,keep_flags,genotype_array) = cleanup_block(test,min_frequency=0.1)

dist_matrix = generate_distance_matrix(
        genotype_array,keep_flags=keep_flags,
        missing_penalty="None")

initial_haps = get_initial_haps(genotype_array,
                keep_flags=keep_flags,
                make_pca=True,deeper_analysis=True)
#%%

initial_matches = match_best(initial_haps,genotype_array)
h2 = generate_further_haps(genotype_array,
            initial_haps,uniqueness_threshold=5,
            wrongness_cutoff=10)

h3 = generate_further_haps(genotype_array,
            h2,uniqueness_threshold=5,
            wrongness_cutoff=10)
h4 = generate_further_haps(genotype_array,
            h3,uniqueness_threshold=4,
            wrongness_cutoff=12)
h5 = generate_further_haps(genotype_array,
            h3,uniqueness_threshold=3,
            wrongness_cutoff=14)
h6 = generate_further_haps(genotype_array,
            h3,uniqueness_threshold=2,
            wrongness_cutoff=16)

#%%
for x in h4.keys():
    for y in h4.keys():
        print(x,y,len(np.where(h4[x] != h4[y])[0]))
    print()