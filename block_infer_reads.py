import numpy as np
import pysam
import math
import copy
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import hdbscan
from multiprocess import Pool
import time
import warnings
import networkx as nx
import pandas as pd
import seaborn as sns
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

def break_contig(vcf_data,contig_name,block_size=100000,shift=50000):
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

def get_vcf_subset(vcf_data,contig_name,start_index,end_index):
    """
    Simple function to extract records from a portion of a contig
    between two positions
    """
    
    data_list = []
    
    full_data = list(vcf_data.fetch(contig_name))
    
    for i in range(len(full_data)):
        record = full_data[i]
        if record.pos < start_index:
            continue
        if record.pos >= end_index:
            break
        
        data_list.append(record)
    
    return data_list
    
    
def cleanup_block_reads(block_list,min_frequency=0.0,
                        read_error_prob=0.02,min_total_reads=5):
    """
    Turn a list of variant records site data into
    a list of site positions and a 3d matrix of the 
    number of reads for ref/alt for that sample at
    that site
    
    Also returns a boolean array of those sites which had enough
    total reads mapped to them to be reliable (control this through
    changing read_error_prob and min_total_reads)
    """
    
    if len(block_list) == 0:
        return (np.array([]),np.array([]))
    
    early_keep_flags = []
    cleaned_positions = []
    cleaned_list = []
    
    samples = block_list[0].samples
    num_samples = len(samples)
    
    for row in block_list:
            
        allele_freq = row.info.get("AF")[0]
        
        if allele_freq >= min_frequency and allele_freq <= 1-min_frequency:
            early_keep_flags.append(1)
        else:
            early_keep_flags.append(0)
        
        
        cleaned_positions.append(row.pos)
            
        row_vals = []
            
        for sample in samples:
            allele_depth = row.samples.get(sample).get("AD")
            allele_depth = allele_depth[:2]
            row_vals.append(list(allele_depth))
            
        cleaned_list.append(row_vals)
    
    reads_array = np.ascontiguousarray(np.array(cleaned_list).swapaxes(0,1))

    total_read_pos = np.sum(reads_array,axis=(0,2))
    
    late_keep_flags = (total_read_pos >= max(min_total_reads,read_error_prob*num_samples)).astype(int)
    
    keep_flags = np.bitwise_and(early_keep_flags,late_keep_flags).astype(int)
    
    return (cleaned_positions,np.array(keep_flags),reads_array)

def resample_reads_array(reads_array,resample_depth):
    """
    Resample the reads array to a reduced read depth
    """
    array_shape = reads_array.shape
    starting_depth = np.sum(reads_array,axis=None)/(array_shape[0]*array_shape[1])
    
    print("Initial Depth was:",starting_depth)
    
    cutoff = resample_depth/starting_depth
    
    if cutoff > 1:
        print("Trying to resample to higher than original depth, exiting")
        assert False
        
    resampled_array = np.random.binomial(reads_array,cutoff)
    return resampled_array
def log_fac(x):
    """
    Returns log(x!) exactly if x < 150, otherwise returns
    Stirling's approximation to it
    """
    if x < 150:
        return math.log(math.factorial(x))
    else:
        return (x+0.5)*math.log(x)-x+math.log(6.283185)
def log_binomial(n,k):
    """
    Returns log(nCk) exactly if n,k < 150, otherwise caclulates
    the consituent factorials through Stirling's approximation
    """
    if n < 150:
        return math.log(math.comb(n,k))
    else:
        return log_fac(n)-log_fac(k)-log_fac(n-k)
    
def make_upper_triangular(matrix):
    """
    Make a matrix upper triangular by adding the value at 
    index (j,i) with i < j to the value at index (i,j)
    
    matrix must be a numpy array
    """
    
    return np.triu(matrix+matrix.T-np.diag(matrix.diagonal()))

def reads_to_probabilities(reads_array,read_error_prob = 0.02,min_total_reads=5):
    """
    Convert a reads array to a probability of the underlying
    genotype being 0, 1 or 2
    
    min_total_reads is a minimum number of reads each site must have for it to be considered
    a valid alternate site (this is to reduce the chance of us considering something which is a variant site only because of errors as a real site)
    """
    reads_sum = np.sum(reads_array,axis=0)
    
    num_samples = len(reads_array)
    
    site_ratios = []
    for i in range(len(reads_sum)):
        
        if sum(reads_sum[i]) >= max(min_total_reads,read_error_prob*num_samples):
            site_ratios.append((1+reads_sum[i][1])/(2+reads_sum[i][0]+reads_sum[i][1]))
        else:
            site_ratios.append(read_error_prob)
    
    site_priors = []
    for i in range(len(site_ratios)):
        singleton = site_ratios[i]
        site_priors.append([(1-singleton)**2,2*singleton*(1-singleton),singleton**2])
    site_priors = np.array(site_priors)
    
    num_samples = reads_array.shape[0]
    num_sites = reads_array.shape[1]
    
    new_array = []
    for i in range(num_sites):
        prior_vals = site_priors[i]
        log_priors = np.array(np.log(prior_vals))
        new_array.append([])
        for j in range(num_samples):
            
            zeros = reads_array[j][i][0]
            ones = reads_array[j][i][1]
            total = zeros+ones
            
            #log_likelihood_00 = np.log(math.comb(total,ones)/1.0)+zeros*np.log(1-read_error_prob)+ones*np.log(read_error_prob)
            #log_likelihood_11 = np.log(math.comb(total,zeros)/1.0)+zeros*np.log(read_error_prob)+ones*np.log(1-read_error_prob)
            #log_likelihood_01 = np.log(math.comb(total,ones)/1.0)+total*np.log(1/2)
            
            log_likelihood_00 = log_binomial(total,ones)+zeros*np.log(1-read_error_prob)+ones*np.log(read_error_prob)
            log_likelihood_11 = log_binomial(total,zeros)+zeros*np.log(read_error_prob)+ones*np.log(1-read_error_prob)
            log_likelihood_01 = log_binomial(total,ones)+total*np.log(1/2)
            
            log_likli = np.array([log_likelihood_00,log_likelihood_01,log_likelihood_11])
            
            nonnorm_log_postri = log_priors + log_likli
            
            nonnorm_log_postri -= np.mean(nonnorm_log_postri)
            
            nonnorm_post = np.exp(nonnorm_log_postri)
            posterior = nonnorm_post/sum(nonnorm_post)
            
            new_array[-1].append(posterior)
            
    new_array = np.array(new_array)
    new_array = np.ascontiguousarray(new_array.swapaxes(0,1))
   
    return (site_priors,new_array)

            
def calc_distance(first_row,second_row,calc_type="diploid"):
    """
    Calculate the probabalistic distance between two rows
    """

    if calc_type == "diploid":
        distances = [[0,1,2],[1,0,1],[2,1,0]]
    else:
        distances = [[0,1],[1,0]]
        
    #print(first_row[:10],second_row[:10])
    
    #print(np.where(np.isnan(second_row)))
        
    ens = np.einsum("ij,ik->ijk",first_row,second_row)
    ensd = ens * distances
    
    return np.sum(ensd,axis=None)

def calc_perc_difference(first_row,second_row,calc_type="diploid"):
    """
    Calculate the probabalistic percentage difference between two rows
    """
    row_len = len(first_row)
    
    return 100*calc_distance(first_row,second_row,calc_type=calc_type)/row_len

def calc_distance_row(row,data_matrix,start_point,calc_type="diploid"):
    """
    Calculate the distance between one row and all rows in a data 
    matrix starting from a given start_point index
    """
    
    num_samples = len(data_matrix)

    row_vals = [0] * start_point
    
    for i in range(start_point,num_samples):
        row_vals.append(calc_distance(row,data_matrix[i],calc_type=calc_type))
    
    return row_vals
    
def generate_distance_matrix(probs_array,
                             keep_flags=None,
                             calc_type="diploid"):
    
    if keep_flags is None:
        keep_flags = np.array([True for _ in range(probs_array.shape[1])])
    
    if keep_flags.dtype != bool:
        keep_flags = np.array(keep_flags,dtype=bool)
          
    probs_array = probs_array[:,keep_flags]
    
    num_samples = probs_array.shape[0]
    
    probs_copy = probs_array.copy()

    
    processing_pool = Pool(processes=8)    
    
    dist_matrix = []
    
    dist_matrix = processing_pool.starmap(
        lambda x,y : calc_distance_row(x,probs_copy,y,calc_type=calc_type),
        zip(probs_array,range(num_samples)))
    
    #Convert to array and fill up lower diagonal
    dist_matrix = np.array(dist_matrix)
    dist_matrix = dist_matrix+dist_matrix.T-np.diag(dist_matrix.diagonal())

    del(processing_pool)
    
    return dist_matrix

#%%
def get_heterozygosity(probabalistic_genotype,keep_flags=None):
    """
    Calculate the heterozygosity of a genotype
    """
    if keep_flags is None:
        keep_flags = np.array([True for _ in range(len(probabalistic_genotype))])
        
    if keep_flags.dtype != bool:
        keep_flags = np.array(keep_flags,dtype=bool)
        
    
    gt_use = np.array(probabalistic_genotype)
    gt_use = gt_use[keep_flags]
    
    num_sites = len(gt_use)
    num_hetero = np.sum(gt_use[:,1])
    
    return 100*num_hetero/num_sites

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

def get_representatives_reads(site_priors,
                        reads_array,
                        cluster_labels,
                        cluster_probs=np.array([]),
                        prob_cutoff=0.8,
                        read_error_prob=0.02):
    """
    Get representatives for each of the clusters we have,
    cluster_labels must be a list of length len(reads_array) showing
    which cluster each sample maps to
    
    This version works by taking as input an array of read counts
    for each sample for each site
    """
    
    singleton_priors = np.array([np.sqrt(site_priors[:,0]),np.sqrt(site_priors[:,2])]).T
    reads_array = np.array(reads_array)
    
    if len(cluster_probs) == 0:
        cluster_probs = np.array([1]*reads_array.shape[0])

    cluster_names = set(cluster_labels)
    cluster_representatives = {}
    
    #Remove the outliers
    cluster_names.discard(-1)
    
    
    
    for cluster in cluster_names:
        
        #Find those indices which strongly map to the current cluster
        indices = np.where(np.logical_and(cluster_labels == cluster,cluster_probs >= prob_cutoff))[0]

        #Pick out these indices and get the data for the cluster
        cluster_data = reads_array[indices,:]
        
        
        cluster_representative = []
    
        #Calculate total count of reads at each site for the samples which fall into the cluster
        site_sum = np.nansum(cluster_data,axis=0)
        
        for i in range(len(site_sum)):
            
            
            priors = singleton_priors[i]
            log_priors = np.log(priors)
            
            zeros = site_sum[i][0]
            ones = site_sum[i][1]
            total = zeros+ones
        
            # log_likelihood_0 = np.log(float(math.comb(total,ones)))+zeros*math.log(1-read_error_prob)+ones*math.log(read_error_prob)
            # log_likelihood_1 = np.log(float(math.comb(total,zeros)))+zeros*math.log(read_error_prob)+ones*math.log(1-read_error_prob)
            
            log_likelihood_0 = log_binomial(total,ones)+zeros*math.log(1-read_error_prob)+ones*math.log(read_error_prob)
            log_likelihood_1 = log_binomial(total,zeros)+zeros*math.log(read_error_prob)+ones*math.log(1-read_error_prob)
            
            
            log_likli = np.array([log_likelihood_0,log_likelihood_1])
            
            nonnorm_log_postri = log_priors + log_likli
            nonnorm_log_postri -= np.mean(nonnorm_log_postri)
            
            #Numerical correction when we have a very wide range in loglikihoods to prevent np.inf
            while np.max(nonnorm_log_postri) > 200:
                nonnorm_log_postri -= 100
            
            
            nonnorm_post = np.exp(nonnorm_log_postri)
            posterior = nonnorm_post/sum(nonnorm_post)
            
            cluster_representative.append(posterior)
        
        #Add the representative for the cluster to the dictionary
        cluster_representatives[cluster] = np.array(cluster_representative)
        

    #Return the results
    return cluster_representatives

def get_representatives_probs(site_priors,
                        probs_array,
                        cluster_labels,
                        cluster_probs=np.array([]),
                        prob_cutoff=0.8,
                        site_max_singleton_sureness=0.98,
                        read_error_prob=0.02):
    """
    Get representatives for each of the clusters we have,
    cluster_labels must be a list of length len(reads_array) showing
    which cluster each sample maps to
    
    This version works by taking as input an array of ref/alt probabilities
    for each sample/site
    
    """
    
    singleton_priors = np.array([np.sqrt(site_priors[:,0]),np.sqrt(site_priors[:,2])]).T
    probs_array = np.array(probs_array)
    
    if len(cluster_probs) == 0:
        cluster_probs = np.array([1]*probs_array.shape[0])

    cluster_names = set(cluster_labels)
    cluster_representatives = {}
    
    #Remove the outliers
    cluster_names.discard(-1)
    
    for cluster in cluster_names:
        
        #Find those indices which strongly map to the current cluster
        indices = np.where(
            np.logical_and(
                cluster_labels == cluster,
                cluster_probs >= prob_cutoff))[0]

        #Pick out these indices and get the data for the cluster
        cluster_data = probs_array[indices,:]
        
        cluster_representative = []
    
        num_sites = cluster_data.shape[1]
        
        for i in range(num_sites):
            
            priors = singleton_priors[i]
            log_priors = np.log(priors)
            
            site_data = cluster_data[:,i,:].copy()
            
            site_data[site_data > site_max_singleton_sureness] = site_max_singleton_sureness
            site_data[site_data < 1-site_max_singleton_sureness] = 1-site_max_singleton_sureness
            
            zero_logli = 0
            one_logli = 0
            
            for info in site_data:
                zero_logli += np.log((1-read_error_prob)*info[0]+read_error_prob*info[1])
                one_logli += np.log(read_error_prob*info[0]+(1-read_error_prob)*info[1])
            
            logli_array = np.array([zero_logli,one_logli])
            
            nonorm_log_post = log_priors+logli_array
            nonorm_log_post -= np.mean(nonorm_log_post)
            
            nonorm_post = np.exp(nonorm_log_post)
            
            posterior = nonorm_post/np.sum(nonorm_post)
            
            cluster_representative.append(posterior)
        
        #Add the representative for the cluster to the dictionary
        cluster_representatives[cluster] = np.array(cluster_representative)
        

    #Return the results
    return cluster_representatives

def combine_haploids(hap_one,hap_two):
    """
    Combine together two haploids to get a diploid
    """
        
    ens = np.einsum("ij,ik->ijk",hap_one,hap_two)
    vals_00 = ens[:,0,0]
    vals_01 = ens[:,0,1]+ens[:,1,0]
    vals_11 = ens[:,1,1]
    
    together = np.ascontiguousarray(np.array([vals_00,vals_01,vals_11]).T)
    
    return together
    
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
    
    combined_haps = {}
    
    for i in haps_dict.keys():
        for j in haps_dict.keys():
            comb = combine_haploids(haps_dict[i],haps_dict[j])
            if (j,i) not in combined_haps.keys():
                combined_haps[(i,j)] = comb
    
    for i in range(len(diploids)):
        cur_best = (None,None)
        cur_div = np.inf
        
        for combination_index in combined_haps.keys():
                combi = combined_haps[combination_index]
                div = 100*calc_distance(diploids[i],combi)/dip_length
                if div < cur_div:
                    cur_div = div
                    cur_best = combination_index
        
        for index in cur_best:
            if index not in haps_usage.keys():
                haps_usage[index] = 0
            haps_usage[index] += 1
            
        errs.append(cur_div)
        dips_matches.append((cur_best,cur_div))
    
    return (dips_matches,haps_usage,np.array(errs))

def add_distinct_haplotypes(initial_haps,
                       new_candidate_haps,
                       keep_flags=None,
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

    cutoff = math.ceil(unique_cutoff*keep_length/100)
    
    i = 0
    j = 0
    num_added = 0
    
    cur_haps = {}
    for idx in initial_haps.keys():
        cur_haps[i] = initial_haps[idx]
        i += 1
    
    for identifier in new_candidate_haps.keys():
        hap = new_candidate_haps[identifier]
        hap_keep = hap[keep_flags]
        add = True
        for k in range(len(cur_haps)):
            compare = cur_haps[k]
            compare_keep = compare[keep_flags]
            
            distance = calc_distance(hap_keep,compare_keep,calc_type="haploid")
            
            if distance < cutoff:
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
                            probs_array,
                            keep_flags = None):
    """
    Add one hap from our list of candidate haps and see how much worse
    of a total fit we get.
    """
    
    for x in starting_haps.keys():
        orig_hap_len = len(starting_haps[x])
        break
    
    if keep_flags is None:
        keep_flags = np.array([True for _ in range(orig_hap_len)])
    
    if keep_flags.dtype != bool:
        keep_flags = np.array(keep_flags,dtype=bool)
        
    added_haps = starting_haps.copy()
    adding_name = max(added_haps.keys())+1
    
    added_haps[adding_name] = candidate_haps[addition_index]
    
    added_matches = match_best(added_haps,probs_array,keep_flags=keep_flags)
    
    added_mean = np.mean(added_matches[2])
    added_max = np.max(added_matches[2])
    added_std = np.std(added_matches[2])
    
    return (added_mean,added_max,added_std,added_matches)


def add_distinct_haplotypes_smart(initial_haps,
                        new_candidate_haps,
                        probs_array,
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
        
    if len(new_candidate_haps) == 0:
        return initial_haps
    
    i = 0
    cur_haps = {}
    for idx in initial_haps.keys():
        cur_haps[i] = initial_haps[idx]
        i += 1
        
    cur_matches = match_best(cur_haps,probs_array,keep_flags=keep_flags)
    cur_error = np.mean(cur_matches[2])
    
    
    
    candidate_haps = new_candidate_haps.copy()
    
    addition_complete = False
    
    while not addition_complete:
        cand_keys = list(candidate_haps.keys())
        addition_indicators = processing_pool.starmap(lambda x:
                            get_addition_statistics(cur_haps,
                            candidate_haps,x,
                            probs_array,keep_flags=keep_flags),
                            zip(cand_keys))            
        
        smallest_result = min(addition_indicators,key=lambda x:x[0])
        smallest_index = addition_indicators.index(smallest_result)
        smallest_name = cand_keys[smallest_index]
        smallest_value = smallest_result[0]
        
        if smallest_value/cur_error < loss_reduction_cutoff_ratio:
            new_index = max(cur_haps.keys())+1
            cur_haps[new_index] = candidate_haps[smallest_name]
            candidate_haps.pop(smallest_name)
            
            cur_matches = match_best(cur_haps,probs_array,keep_flags=keep_flags)
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

def get_initial_haps(site_priors,
                     probs_array,
                     reads_array,
                     keep_flags=None,
                     het_cutoff_start=10,
                     het_excess_add=2,
                     het_max_cutoff=20,
                     deeper_analysis=False,
                     uniqueness_tolerance=5,
                     read_error_prob = 0.02,
                     verbose=False):
    """
    Get our initial haplotypes by finding high homozygosity samples
    
    keep_flags is a optional boolean array which is 1 at those sites which
    we wish to consider for purposes of the analysis and 0 elsewhere,
    if not provided we use all sites
    """
    if keep_flags is None:
        keep_flags = np.array([True for _ in range(probs_array.shape[1])])
    
    if keep_flags.dtype != bool:
        keep_flags = np.array(keep_flags,dtype=bool)

    het_vals = np.array([get_heterozygosity(probs_array[i],keep_flags) for i in range(len(probs_array))])
    
    found_homs = False
    cur_het_cutoff = het_cutoff_start
    
    accept_singleton = (het_cutoff_start+het_max_cutoff)/2
    
    while not found_homs:
        
        if cur_het_cutoff > het_max_cutoff:
            if verbose:
                print("Unable to find samples with high homozygosity in region")
            
            base_probs = []
            for i in range(len(site_priors)):
                ref_prob = math.sqrt(site_priors[i,0])
                alt_prob = 1-ref_prob
                base_probs.append([ref_prob,alt_prob])
            base_probs = np.array(base_probs)
            
            return {0:base_probs}
        
        homs_where = np.where(het_vals <= cur_het_cutoff)[0]
    
        homs_array = probs_array[homs_where]
        corresp_reads_array = reads_array[homs_where]
        
        if len(homs_array) < 5:
            cur_het_cutoff += het_excess_add
            continue
        
        
        homs_array[:,:,0] += 0.5*homs_array[:,:,1]
        homs_array[:,:,2] += 0.5*homs_array[:,:,1]
        
        homs_array = homs_array[:,:,[0,2]]
    
        dist_submatrix = generate_distance_matrix(
            homs_array,keep_flags=keep_flags,
            calc_type="haploid")
        
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
    
    representatives = get_representatives_reads(site_priors,
        corresp_reads_array,initial_clusters[0],
        read_error_prob=read_error_prob)
    
    #Hacky way to remove any haps that are too close to each other
    (representatives,
      label_mappings) = add_distinct_haplotypes(
              {},representatives,keep_flags=keep_flags,
              unique_cutoff=uniqueness_tolerance)
        
    return representatives

def get_diff_wrongness(diploid,haploid,keep_flags=None):
    """
    Get the probabalistic difference created by subtracting a 
    haploid from a diploid. Also gets a wrongness metric for
    what probabalistic proportion of the sites where keep_flags=True
    were wrong (not 0 or 1)
    """
    
    if keep_flags is None:
        keep_flags = np.array([True for _ in range(len(diploid))])
    
    if keep_flags.dtype != bool:
        keep_flags = np.array(keep_flags,dtype=bool)
    
    ens = np.einsum("ij,ik->ijk",haploid,diploid)
    
    zeros = ens[:,0,0]+ens[:,1,0]+ens[:,1,1]
    ones = ens[:,0,1]+ens[:,0,2]+ens[:,1,2]
    wrong = ens[:,1,0]+ens[:,0,2]
    
    wrong = wrong[keep_flags]
    
    
    
    difference = np.ascontiguousarray(np.array([zeros,ones]).T)
    
    
    perc_wrong = 100*np.sum(wrong)/len(wrong)
    
    return (difference,perc_wrong)

def generate_further_haps(site_priors,
                          probs_array,
                          initial_haps,
                          keep_flags=None,
                          wrongness_cutoff=10,
                          uniqueness_threshold=5,
                          max_hap_add = 1000,
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
    
    """
    if keep_flags is None:
        keep_flags = np.array([True for _ in range(probs_array.shape[1])])
    
    if keep_flags.dtype != bool:
        keep_flags = np.array(keep_flags,dtype=bool)
        
    candidate_haps = []
    initial_list = np.array(list(initial_haps.values()))
    
    for geno in probs_array:
        for init_hap in initial_list:
            
            (fixed_diff,wrongness) = get_diff_wrongness(geno,init_hap,keep_flags=keep_flags)
            
            if wrongness <= wrongness_cutoff:

                add = True
                for comp_hap in initial_list:
                    
                    fixed_keep = fixed_diff[keep_flags]
                    comp_keep = comp_hap[keep_flags]
                    
                    perc_diff = 100*calc_distance(fixed_keep,comp_keep,calc_type="haploid")/len(comp_keep)
                    
                    
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
        calc_type="haploid")

    initial_clusters = hdbscan_cluster(
                        dist_submatrix,
                        min_cluster_size=len(initial_haps)+1,
                        min_samples=1,
                        cluster_selection_method="eom",
                        alpha=1.0)
    
    
    representatives = get_representatives_probs(
        site_priors,candidate_haps,initial_clusters[0])
    

    final_haps = add_distinct_haplotypes_smart(initial_haps,
                representatives,probs_array,
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
    
    
    (positions,keep_flags,reads_array) = cleanup_block_reads(block_data,min_frequency=0)
    
    #reads_array = resample_reads_array(reads_array,1)
    
    (site_priors,probs_array) = reads_to_probabilities(reads_array)
    
    
    initial_haps = get_initial_haps(site_priors,probs_array,
        reads_array,keep_flags=keep_flags)
    
    initial_matches = match_best(initial_haps,probs_array,keep_flags=keep_flags)
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
        cur_haps = generate_further_haps(site_priors,probs_array,
                    cur_haps,keep_flags=keep_flags,uniqueness_threshold=uniqueness_threshold,
                    wrongness_cutoff=wrongness_cutoff)
        cur_matches = match_best(cur_haps,probs_array)
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
    
    final_haps = haps_history[-1]
        
    return (positions,keep_flags,reads_array,final_haps)
    
    

def generate_haplotypes_all(chromosome_data):
    """
    Generate a list of block haplotypes which make up each element 
    of the list of blocks of VCF data
    """
    
    # total_positions = []
    # total_keep_flags = []
    # reads_arrays = []
    # haps = []
    
    overall_haplotypes = []
    
    for i in range(len(chromosome_data)):
        print(i,chromosome_data[i][1])
        
        
        overall_haplotypes.append(generate_haplotypes_block(
            chromosome_data[i][0]))

    
    return overall_haplotypes

def greatest_likelihood_hap(hap):
    """
    Convert a probabalistic hap to a deterministic one by
    choosing the highest probability allele at each site
    """
    return np.argmax(hap,axis=1)

def match_haplotypes_by_overlap(block_level_haps,
                     hap_cutoff_autoinclude=2,
                     hap_cutoff_noninclude=5,):
    """
    Takes as input a list of positions and block level haplotypes and
    finds which haps from which block 
    
    hap_cutoff_autoinclude is an upper bound for how different two 
    overlapping portions from neighbouring haps can be for
    us to always have a link between them
    
    hap_cutoff_noninclude is an lower bound such that whenever two
    overlapping portions are at least this different we never have 
    a link between them
    
    """        
    
    next_starting = []
    matches = []
    
    block_haps_names = []
    block_counts = {}
    
    #Find overlap starting points
    for i in range(1,len(block_level_haps)):
        start_position = block_level_haps[i][0][0]
        insertion_point = np.searchsorted(block_level_haps[i-1][0],start_position)
        next_starting.append(insertion_point)
        
    #Create list of unique names for the haps in each of the blocks    
    for i in range(len(block_level_haps)):
        block_haps_names.append([])
        block_counts[i] = len(block_level_haps[i][3])
        for name in block_level_haps[i][3].keys():
            block_haps_names[-1].append((i,name))

    #Iterate over all the blocks
    for i in range(len(block_level_haps)-1):
        start_point = next_starting[i]
        overlap_length = len(block_level_haps[i][0])-start_point   
        
        cur_ends = {k:block_level_haps[i][3][k][start_point:] for k in block_level_haps[i][3].keys()}
        next_ends = {k:block_level_haps[i+1][3][k][:overlap_length] for k in block_level_haps[i+1][3].keys()}

        cur_keep_flags = np.array(block_level_haps[i][1][start_point:],dtype=bool)
        next_keep_flags = np.array(block_level_haps[i+1][1][:overlap_length],dtype=bool)
        
        assert (cur_keep_flags == next_keep_flags).all(),"Keep flags don't match up"
        
        matches.append([])
        
        min_expected_connections = max(block_counts[i],block_counts[i+1])
        if block_counts[i] == 0 or block_counts[i+1] == 0:
            min_expected_connections = 0
            
        amount_added = 0
        all_edges_consideration = {}
        
        for first_name in cur_ends.keys(): 
            dist_values = {}

            for second_name in next_ends.keys():
                
                first_new_hap = cur_ends[first_name][cur_keep_flags]
                second_new_hap = next_ends[second_name][next_keep_flags]
                
                common_size = len(first_new_hap)
                
                if common_size > 0:
                    haps_dist = 100*calc_distance(first_new_hap,
                                      second_new_hap,
                                      calc_type="haploid")/common_size
                else:
                    haps_dist = 0
                    
                dist_values[(first_name,second_name)] = haps_dist

            #Add autoinclude edges and remove from consideration those which are too different to ever be added
            removals = []
            for k in dist_values.keys():
                if dist_values[k] <= hap_cutoff_autoinclude:
                    matches[-1].append(((i,k[0]),(i+1,k[1])))
                    amount_added += 1
                    removals.append(k)
                if dist_values[k] >= hap_cutoff_noninclude:
                    removals.append(k)
            for k in removals:
                dist_values.pop(k)
            all_edges_consideration.update(dist_values)
        
        all_edges_consideration = {k:v for k, v in sorted(all_edges_consideration.items(), key=lambda item: item[1])}
        removals = []
        
        for key in all_edges_consideration:
            if amount_added < min_expected_connections:
                matches[-1].append(((i,key[0]),(i+1,key[1])))
                amount_added += 1
                continue
            
    return (block_haps_names,matches)

def match_haplotypes_by_overlap_probabalistic(block_level_haps):
    """
    Probabalistic version of match_haplotypes_by_overlap which
    instead of returning a list of edges it returns a likelihood of 
    an edge for each pair of nodes in neighbouring layers
    
    Takes as input a list of positions and block level haplotypes and
    finds which haps from which block 
    """
    next_starting = []
    matches = []
    
    block_haps_names = []
    block_counts = {}
    
    #Find overlap starting points
    for i in range(1,len(block_level_haps)):
        start_position = block_level_haps[i][0][0]
        insertion_point = np.searchsorted(block_level_haps[i-1][0],start_position)
        next_starting.append(insertion_point)
        
    #Create list of unique names for the haps in each of the blocks    
    for i in range(len(block_level_haps)):
        block_haps_names.append([])
        block_counts[i] = len(block_level_haps[i][3])
        for name in block_level_haps[i][3].keys():
            block_haps_names[-1].append((i,name))

    #Iterate over all the blocks
    for i in range(len(block_level_haps)-1):
        start_point = next_starting[i]
        overlap_length = len(block_level_haps[i][0])-start_point   
        
        cur_ends = {k:block_level_haps[i][3][k][start_point:] for k in block_level_haps[i][3].keys()}
        next_ends = {k:block_level_haps[i+1][3][k][:overlap_length] for k in block_level_haps[i+1][3].keys()}
           
        cur_keep_flags = np.array(block_level_haps[i][1][start_point:],dtype=bool)
        next_keep_flags = np.array(block_level_haps[i+1][1][:overlap_length],dtype=bool)
        
        assert (cur_keep_flags == next_keep_flags).all(),"Keep flags don't match up"
        
        similarities = {}
        for first_name in cur_ends.keys(): 

            for second_name in next_ends.keys():
                
                first_new_hap = cur_ends[first_name][cur_keep_flags]
                second_new_hap = next_ends[second_name][next_keep_flags]
                
                common_size = len(first_new_hap)
                
                if common_size > 0:
                    haps_dist = 100*calc_distance(first_new_hap,
                                      second_new_hap,
                                      calc_type="haploid")/common_size
                else:
                    haps_dist = 0
                
                if haps_dist > 50:
                    similarity = 0
                else:
                    similarity = 2*(50-haps_dist)
                
                #print(first_name,second_name,haps_dist,similarity)
                similarities[((i,first_name),(i+1,second_name))] = similarity
            
        #Scale and transform the similarities into a score
        transform_similarities = {}
            
        for item in similarities.keys():
            val = similarities[item]/100
                
            transformed = 100*(val**2)
            transform_similarities[item] = transformed
            
        matches.append(transform_similarities)
            
    return (block_haps_names,matches)

def match_usage_find(first_hap,first_matches,second_matches):
    """
    Counts the relative usage of haps in second_matches for those
    samples which include first_hap in first_matches
    
    first_matches and second_matches must correspond to the same 
    samples in order (so the first element of both are from the same 
    sample etc. etc.)
    """
    use_indices = []
    
    for sample in range(len(first_matches[0])):
        if first_matches[0][sample][0][0] == first_hap:
            use_indices.append(sample)
        if first_matches[0][sample][0][1] == first_hap:
            use_indices.append(sample)
    
    second_usages = {}
    
    for sample in use_indices:
        dat = second_matches[0][sample][0]
        
        for s in dat:
            if s not in second_usages.keys():
                second_usages[s] = 0
            second_usages[s] += 1
    
    second_usages = {k: v for k, v in sorted(second_usages.items(), key=lambda item: item[1])}
    
    return second_usages

def hap_matching_from_haps(haps_data):
    """
    Takes haps data for a single block and calculates 
    the best matches for the haps for that block
    """
    keep_flags = haps_data[1]
    reads_array = haps_data[2]
    haps = haps_data[3]
    (site_priors,probs_array) = reads_to_probabilities(reads_array)
    
    matches = match_best(haps,probs_array,keep_flags=keep_flags)
    
    return matches

def get_best_matches_all_blocks(haps_data):
    """
    Multithreaded function to calculate the best matches for 
    each block in haps data
    """
    
    processing_pool = Pool(8)
    
    processing_results = processing_pool.starmap(hap_matching_from_haps,
                                                 zip(haps_data))
    
    return processing_results

def hap_matching_comparison(haps_data,matches_data,first_block_index,second_block_index):
    """
    For each hap at the first_block_index block this fn. compares
    where the samples which use that hap end up for the block at
    second_block_index and converts these numbers into percentage
    usages for each hap at index first_block_index
        
    It also then scales these scores and returns the scaled version of 
    these scores back to us as a dictionary
    """
        
    forward_scores = {}
    backward_scores = {}
        
    first_haps_data = haps_data[first_block_index][3]
    second_haps_data = haps_data[second_block_index][3]
        
    first_matches_data = matches_data[first_block_index]
    second_matches_data = matches_data[second_block_index]
        
    for hap in first_haps_data.keys():
        hap_usages = match_usage_find(hap,first_matches_data,second_matches_data)
        total_matches = sum(hap_usages.values())
        hap_percs = {x:100*hap_usages[x]/total_matches for x in hap_usages.keys()}
            
        scaled_scores = {}
            
        for other_hap in second_haps_data.keys():
            if other_hap in hap_percs.keys():
                scaled_val = 100*(min(1,2*hap_percs[other_hap]/100))**2
                    
                scaled_scores[((first_block_index,hap),
                               (second_block_index,other_hap))] = scaled_val
            elif other_hap not in hap_percs.keys():
                scaled_scores[((first_block_index,hap),
                               (second_block_index,other_hap))] = 0
        forward_scores.update(scaled_scores)
            
    for hap in second_haps_data.keys():
        hap_usages = match_usage_find(hap,second_matches_data,first_matches_data)
        total_matches = sum(hap_usages.values())
        hap_percs = {x:100*hap_usages[x]/total_matches for x in hap_usages.keys()}
            
        scaled_scores = {}
            
        for other_hap in first_haps_data.keys():
            if other_hap in hap_percs.keys():
                scaled_val = 100*(min(1,2*hap_percs[other_hap]/100))**2
                    
                scaled_scores[((first_block_index,other_hap),
                               (second_block_index,hap))] = scaled_val
            elif other_hap not in hap_percs.keys():
                scaled_scores[((first_block_index,other_hap),
                               (second_block_index,hap))] = 0
        backward_scores.update(scaled_scores)
        
    return (forward_scores,backward_scores)
    
def match_haplotypes_by_samples(full_haps_data):
    """
    Alternate method of matching haplotypes in nearby blocks together
    by matching hap A with hap B if the samples which use hap A at its location
    disproportionately use hap B at its location
    
    full_haps_data is the return from a run of generate_haplotypes_all,
    containing info about the positions, read count array and haplotypes
    for each block.
    """
    
    
    #Controls the threshold for which a score higher than results in an edge and how low other scores relative to the highest can be for an edge to be added
    auto_add_val = 70
    max_reduction_include = 0.8
    
    num_blocks = len(full_haps_data)
    
    all_matches = []
    
    block_haps_names = []
    for i in range(len(full_haps_data)):
        block_haps_names.append([])
        
        for nm in full_haps_data[i][3].keys():
            block_haps_names[-1].append((i,nm))
    
    match_best_results = []
    
    processing_pool = Pool(8)
    
    match_best_results = processing_pool.starmap(hap_matching_from_haps,
                                                 zip(full_haps_data))
    
    neighbouring_usages = processing_pool.starmap(lambda x,y:
                        hap_matching_comparison(full_haps_data,match_best_results,x,y),
                        zip(list(range(num_blocks-1)),list(range(1,num_blocks))))
    
        
    forward_matches = []
    backward_matches = []
    
    for x in range(len(block_haps_names)-1):
        first_names = block_haps_names[x]
        second_names = block_haps_names[x+1]
        
        forward_edges_add = []
        for first_hap in first_names:
            highest_score_found = -1
            for second_hap in second_names:
                sim_val = neighbouring_usages[x][0][(first_hap,second_hap)]
                if sim_val > highest_score_found:
                    highest_score_found = sim_val

            for second_hap in second_names:
                sim_val = neighbouring_usages[x][0][(first_hap,second_hap)]
                
                if sim_val >= auto_add_val or sim_val > highest_score_found*max_reduction_include:
                    forward_edges_add.append((first_hap,second_hap))
        
        forward_matches.append(forward_edges_add)
        
        backward_edges_add = []
        for second_hap in second_names:
            highest_score_found = -1
            for first_hap in first_names:
                sim_val = neighbouring_usages[x][1][(first_hap,second_hap)]
                if sim_val > highest_score_found:
                    highest_score_found = sim_val
                    
            for first_hap in first_names:
                sim_val = neighbouring_usages[x][1][(first_hap,second_hap)]
                
                if sim_val >= auto_add_val or sim_val > highest_score_found*max_reduction_include:
                    backward_edges_add.append((first_hap,second_hap))
        
        backward_matches.append(backward_edges_add)

    for i in range(len(forward_matches)):
        commons = []
        for x in forward_matches[i]:
            if x in backward_matches[i]:
                commons.append(x)
        all_matches.append(commons)
        
    return [(block_haps_names,forward_matches),(block_haps_names,backward_matches),(block_haps_names,all_matches)]
        
def match_haplotypes_by_samples_probabalistic(full_haps_data):
    """
    Probabalistic version of match_haplotypes_by_samples that gives 
    a likelihood for each possible edge between neighbouring layers
    
    full_haps_data is the return from a run of generate_haplotypes_all,
    containing info about the positions, read count array and haplotypes
    for each block.
    
    Returns a list of node labels and a list of dictionaries with scaled
    scores for how strong of an edge there is between the first element and the 
    second element of the key for each key in each dictionary
    """
        
    num_blocks = len(full_haps_data)
    
    block_haps_names = []
    for i in range(len(full_haps_data)):
        block_haps_names.append([])
        
        for nm in full_haps_data[i][3].keys():
            block_haps_names[-1].append((i,nm))
    
    match_best_results = []
    
    processing_pool = Pool(8)
    
    match_best_results = processing_pool.starmap(hap_matching_from_haps,
                                                 zip(full_haps_data))
    
    neighbouring_usages = processing_pool.starmap(lambda x,y:
                        hap_matching_comparison(full_haps_data,match_best_results,x,y),
                        zip(list(range(num_blocks-1)),list(range(1,num_blocks))))
    
    forward_match_scores = [neighbouring_usages[x][0] for x in range(num_blocks-1)]
    backward_match_scores = [neighbouring_usages[x][1] for x in range(num_blocks-1)]
    
    combined_scores = []
    

    for i in range(len(forward_match_scores)):
        commons = {}
        for x in forward_match_scores[i].keys():
            commons[x] = (forward_match_scores[i][x]+backward_match_scores[i][x])/2
            
        combined_scores.append(commons)
        
    return [(block_haps_names,forward_match_scores),(block_haps_names,backward_match_scores),(block_haps_names,combined_scores)]
        
def get_combined_hap_score(hap_overlap_scores,hap_sample_scores,
                           overlap_importance=1):
    """
    Combines the results from match_haplotypes_by_overlap_probabalistic
    and match_haplotypes_by_samples_probabalistic (just the combined output
    of this function and not the forward/backward ones) to get one single
    likelihood score for each edge which is normalised to a maximum value
    of 100. 
    
    overlap_importance is a measure of how much we weight the overlap score vs
    the sample scores (individually). A value of 1 here means we weight the
    combined sample score twice as much as the overlap score (because sample
    score is combined version of both forward and backward sample scores).
    
    """
    ovr = hap_overlap_scores[1]
    samps = hap_sample_scores[1]
    
    total_weight = overlap_importance+2
    
    combined_dict = {}
    
    for i in range(len(ovr)):
        for d in ovr[i].keys():
            comb = (overlap_importance*ovr[i][d]+2*samps[i][d])/total_weight      
            combined_dict[d] = comb
    
    return combined_dict

def get_block_hap_similarities(haps_data):
    """
    Takes as input a list of haplotypes such as generated from generate_haplotypes_all and 
    calculates a similarity matrix between them with values 
    from 0 to 1 with higher values denoting more similar haps
    """
    
    scores = []
    
    keep_flags = np.array(haps_data[1],dtype=bool)
    
    hap_vals = haps_data[3]
    
    for i in hap_vals.keys():
        scores.append([])
        for j in hap_vals.keys():
            if j < i:
                scores[-1].append(0)
            else:
                
                first_hap = hap_vals[i][keep_flags]
                second_hap = hap_vals[j][keep_flags]
                hap_len = len(first_hap)
                
                scoring = 2.0*calc_distance(first_hap,second_hap,calc_type="haploid")/hap_len
                
                similarity = 1.0-min(1.0,scoring)
                scores[-1].append(similarity)
                
    scores = np.array(scores)
    scores = scores+scores.T-np.diag(scores.diagonal())
    
    scr_diag = np.sqrt(scores.diagonal())
    
    scores = scores/scr_diag
    scores = scores/scr_diag.reshape(1,-1).T
    
    return scores       

def calc_best_scoring(padded_nodes_list,node_scores,edge_scores):
    """
    Uses Breadth First Search going backwards to calculate the best scoring possible for each starting 
    node assuming we get to the end. This uses "I" as a dummy initial starting node
    and "S" as the final sink node
    """
    
    num_layers = len(padded_nodes_list)

    
    scorings = [{"S":0}]
    
    for layer in range(num_layers-2,-1,-1):
        this_nodes = padded_nodes_list[layer]
        next_nodes = padded_nodes_list[layer+1]
        
        layer_scores = {}
        
        for node in this_nodes:
            best_score = -np.inf
            for other in next_nodes:
                new_score = node_scores[node]+edge_scores[(node,other)]+scorings[-1][other]
                
                if new_score > best_score:
                    best_score = new_score
                
            layer_scores[node] = best_score
            
        scorings.append(layer_scores)
    
    return scorings[::-1]
    
def scorings_to_optimal_path(scorings,padded_nodes_list,node_scores,edge_scores):
    """
    Takes a list of dictionaries of optimal scorings from each node to the end and
    calculates the optimal path starting at I and ending at S
    """
    cur_path = ["I"]
    cur_node = "I"
    
    for i in range(len(padded_nodes_list)-1):

        cur_score = scorings[i][cur_node]
        
        next_nodes = padded_nodes_list[i+1]
        for new_node in next_nodes:
            score_removal = node_scores[cur_node]+edge_scores[(cur_node,new_node)]
            remaining_score = cur_score-score_removal
            
            if abs(remaining_score-scorings[i+1][new_node]) < 10**-10:
                cur_path.append(new_node)
                cur_node = new_node
                break
    
    return cur_path     

def generate_chained_block_haplotypes(haplotype_data,nodes_list,combined_scores,num_haplotypes,
                             node_usage_penalty=10,edge_usage_penalty=10):
    """
    Generates num_haplotypes many chromosome length haplotypes given
    a layered list of the nodes and a dictionary containing the combined likelihood
    scores for each possible edge between layers.
    
    Returns the haplotypes as a list of nodes, one from each layer from the start to 
    the end.
    
    This function works through a reverse Breadth First Search algorithm trying to maximize
    the score between the start and the end. The first haplotype is just the maximal path.
    
    For future haplotypes we apply a penalty to each node/edge already on a discovered 
    haplotype with a penalty also applied to other similar nodes to used nodes in the same
    layer and run the Breadth First search again. We repeat this process until we generate
    num_haplotypes many haplotypes.
    """
    
    num_layers = len(nodes_list)
    
    processing_pool = Pool(8)
    
    similarity_matrices = processing_pool.starmap(get_block_hap_similarities,
                                       zip(haplotype_data))#Similarity matrices for calculating the associated penalty when we use a node
    
    current_edge_scores = combined_scores.copy()
    current_node_scores = {"I":0,"S":0}
    for i in range(len(nodes_list)):
        for node in nodes_list[i]:
            current_node_scores[node] = 0
    
    nodes_copy = nodes_list.copy()
    nodes_copy.insert(0,["I"])
    nodes_copy.append(["S"])
    
    #Add edges from the dummy nodes to first and last layers
    for xm in range(len(nodes_list[0])):
        current_edge_scores[("I",(0,xm))] = 0
    for xm in range(len(nodes_list[-1])):
        current_edge_scores[((num_layers-1,xm),"S")] = 0
    
    found_haps = []
    
    for ite in range(num_haplotypes):
        best_scores = calc_best_scoring(nodes_copy,current_node_scores,current_edge_scores)
        
        found_hap = scorings_to_optimal_path(best_scores,nodes_copy,current_node_scores,current_edge_scores)
        
        #Now that we have our hap apply node penalties
        for i in range(1,len(found_hap)-1):
            layer = found_hap[i][0]
            used_hap = found_hap[i][1]
            reductions = (node_usage_penalty)*similarity_matrices[layer][used_hap,:]
            for nm in range(len(reductions)):
                current_node_scores[(layer,nm)] -= reductions[nm]
        
        #And apply edge penalties
        for i in range(1,len(found_hap)-2):
            current_edge_scores[(found_hap[i],found_hap[i+1])] -= edge_usage_penalty
        
        found_haps.append(found_hap[1:-1])

    return found_haps

def probability_to_information(probs_list):
    """
    Takes as input a list of two elements of the form 
    [1-p,p] with p in [0,1] and gives the signed information 
    of the data
    
    This is defined so that if p = 0 we return -1, if p = 1
    we return 1 and if p = 0.5 we return 0: more concretely if 
    p < 0.5 then sgn = -1, else sgn = +1 and we reurn sgn*(1-entropy)
    where the entropy is defined in the standard way
    """
    
    p = probs_list[1]
    
    if p == 1:
        return 1
    elif p == 0:
        return -1
    
    if p < 0.5:
        sgn = -1
    else:
        sgn = 1
    
    entropy = -(1-p)*math.log2(1-p)-p*math.log2(p)
    
    return sgn*(1-entropy)

def add_informations(first_information,second_information):
    """
    Adds together two signed information values using the
    relativistic velocity addition formula
    
    Both first_information and second_information must be 
    within [-1,1]
    """
    if abs(first_information) == 1 and abs(second_information) == 1:
        if first_information != second_information:
            return 0
        else:
            return first_information
        
    return (first_information+second_information)/(1+first_information*second_information)

def combine_probabilities(first_prob,second_prob,prior_prob,
                          required_accuracy=10**-13):
    """
    Combine two lists of the form [1-p,p] denoting the probability
    of having a ref/alt at a site for a hap and a given prior probability 
    for that site into a single new probability
    
    required_accuracy is the amount of accuracy we want in our final answer,
    the runtime of the algorithm is linear in -log(required_accuracy)
    
    required_accuracy can not be lower than about 10**-15 for floating point
    precision reasons. For safety keep it higher than 10**-14
    """ 
    
    first_information = probability_to_information(first_prob)
    second_information = probability_to_information(second_prob)
    
    prior_information = probability_to_information(prior_prob)
    
    first_relative = add_informations(first_information,-prior_information)
    second_relative = add_informations(second_information,-prior_information)
    
    combined_relative_information = add_informations(first_relative,second_relative)
    full_combined_information = add_informations(combined_relative_information,prior_information)
    
    #Invert from information space to get our probability back
    #We do this using binary search
    search_space = [0,1]
        
    while search_space[1]-search_space[0] > required_accuracy:
        midpoint = (search_space[0]+search_space[1])/2
        test_val = probability_to_information([1-midpoint,midpoint])
        
        # print("VALUE",test_val,full_combined_information,search_space)
        # print(search_space[1]-search_space[0])
        # print()
        if test_val < full_combined_information:
            search_space = [midpoint,search_space[1]]
        else:
            search_space = [search_space[0],midpoint]
    
    final_prob = (search_space[0]+search_space[1])/2
    
    return [1-final_prob,final_prob]

def combine_chained_blocks_to_single_hap(all_haps,
                                         hap_blocks,
                                         read_error_prob = 0.02,
                                         min_total_reads=5):
    """
    Takes in as input the block level haplotypes (such as generated by
    by generate_haplotypes_all) as well as a list giving the blocks which
    make up a haplotype and then converts this into a single long 
    chromosome length haplotype
    
    read_error_prob and min_total_reads are used to calculate the haplotype
    level priors for each site from the read counts in all_haps
    
    This function assumes that everything except for the very end of the 
    starting/finishing block ends up in exactly two blocks, i.e. the shift size
    is exactly half of the block size!!!
    
    """
    
    next_starting = []
    
    #Find overlap starting points
    for i in range(1,len(all_haps)):
        start_position = all_haps[i][0][0]
        insertion_point = np.searchsorted(all_haps[i-1][0],start_position)
        next_starting.append(insertion_point)
        
    
    final_haplotype = []
    final_locations = []
    
    for i in range(len(all_haps)-1):
        hap_here = hap_blocks[i][1]
        hap_next = hap_blocks[i+1][1]
        start_point = next_starting[i]
        overlap_length = len(all_haps[i][0])-start_point  

        
        if i == 0:
            start_data = all_haps[i][3][hap_here][:start_point]
            final_haplotype.extend(start_data)
            final_locations.extend(all_haps[i][0][:start_point])
        
        
        cur_overlap_data = all_haps[i][3][hap_here][start_point:]
        next_overlap_data = all_haps[i+1][3][hap_next][:overlap_length]
        
        reads_sum = np.sum(all_haps[i][2][:,start_point:,:],axis=0)

        num_samples = len(all_haps[i][2])
    
        hap_priors = []
        for j in range(len(reads_sum)):
        
            if sum(reads_sum[j]) >= max(min_total_reads,read_error_prob*num_samples):
                rat_val = (1+reads_sum[j][1])/(2+reads_sum[j][0]+reads_sum[j][1])
            else:
                rat_val = read_error_prob
            hap_priors.append([rat_val,1-rat_val])
        hap_priors = np.array(hap_priors)
        
        new_probs = []
        for j in range(len(hap_priors)):
            new_val = combine_probabilities(cur_overlap_data[j],next_overlap_data[j],hap_priors[j])

            new_probs.append(new_val)
        new_probs = np.array(new_probs)
        
        final_haplotype.extend(new_probs)
        final_locations.extend(all_haps[i+1][0][:overlap_length])
        
        if i == len(all_haps)-2:
            end_data = all_haps[i+1][3][hap_next][overlap_length:]
            final_haplotype.extend(end_data)
            final_locations.extend(all_haps[i+1][0][overlap_length:])
    
    return [np.array(final_locations),np.array(final_haplotype)]

def combine_all_blocks_to_long_haps(all_haps,
                                    hap_blocks_list,
                                    read_error_prob = 0.02,
                                    min_total_reads=5):
    """
    Multithreaded version of combine_chained_blocks_to_single_hap
    which processes all of our haps at once
    """
        
    processing_pool = Pool(8)
    
    processing_results = processing_pool.starmap(lambda x: combine_chained_blocks_to_single_hap(
                                        all_haps,x,read_error_prob=read_error_prob,
                                        min_total_reads=min_total_reads),
                                        zip(hap_blocks_list))
    
    sites_loc = processing_results[0][0]
    long_haps = [processing_results[x][1] for x in range(len(processing_results))]
    
    return [sites_loc,long_haps]

def combine_long_haplotypes(long_haps):
    """
    Create each possible combination of long diploids from 
    a set of long haplotypes and return the result as a high
    dimensional array
    
    This uses multiprocessing to speed up computation
    """
    def extract_and_combine(full_haps,first_index,second_index):
        first_hap = full_haps[first_index]
        second_hap = full_haps[second_index]
        
        return combine_haploids(first_hap,second_hap)
    
    num_haps = len(long_haps)
    all_combs = [(i,j) for i in range(num_haps) for j in range(num_haps)]
    
    processing_pool = Pool(8)
    
    all_combined = processing_pool.starmap(lambda x,y: extract_and_combine(long_haps,x,y),
                                           all_combs)
    
    #Reshape the flat array of size num_haps**2 into a num_haps*num_haps array
    total_combined = []
    for i in range(num_haps):
        subset = all_combined[num_haps*i:num_haps*(i+1)]
        total_combined.append(subset)
    
    total_combined = np.array(total_combined)
    
    print(total_combined.shape)
    
    return total_combined

def recombination_fudge(start_probs,distance,recomb_rate=10**-8):
    """
    Function which takes in the genotype copying probabilities and 
    updates them for a location "distance" number of sites downstream 
    based on up to a single recombination event happening within this 
    particular stretch of data
    """
    
    num_rows = start_probs.shape[0]
    num_cols = start_probs.shape[1]
    
    num_possible_switches = num_rows+num_cols-1
    
    non_switch_prob = (1-recomb_rate)**distance
    
    each_switch_prob = (1-non_switch_prob)/num_possible_switches
    
    #Account for the fact that we can recombine back to the haplotype pre recombination
    total_non_switch_prob = non_switch_prob+each_switch_prob
    
    final_mats = []
    
    for i in range(num_rows):
        for j in range(num_cols):
            base = np.zeros((num_rows,num_cols))
            base[i,:] = each_switch_prob*start_probs[i,j]*np.ones(num_cols)
            base[:,j] = each_switch_prob*start_probs[i,j]*np.ones(num_rows)
            base[i,j] = total_non_switch_prob*start_probs[i,j]
            final_mats.append(base)
    
    combined_probability = np.sum(final_mats,axis=0)
    
    return combined_probability
#%%

def nodes_list_to_pos(nodes_list,layer_dist = 4,vert_dist=2):
    """
    Takes as input a list of nodes which are a 2-tuple with the
    first element of the tuple signifying the layer and the second
    the identifier within the layer. The second identifer must begin 
    with 0 and count upwards from there for each layer
    
    Returns a list of positions for plotting the layers in networkx
    """
    
    offset = layer_dist/2
    
    pos_dict = {}
    
    num_in_layer = {}
    
    for node_layer in nodes_list:
        for node in node_layer:
            if node[0] not in num_in_layer.keys():
                num_in_layer[node[0]] = 0
            num_in_layer[node[0]] += 1
    
    for node_layer in nodes_list:
        for node in node_layer:
            x_val = node[0]*layer_dist+offset
        
            total_vert_dist = (num_in_layer[node[0]]-1)*vert_dist
        
        
            top_starting = total_vert_dist/2
        
            y_val = top_starting-node[1]*vert_dist
        
            pos_dict[node] = np.array([x_val,y_val])
    
    return pos_dict

def planarify_graph(nodes,edges):
    """
    Takes a list of layers of nodes and a list of edges
    between consecutive layers. Relabels the nodes and edges
    so that the final graph produced when plotted tries to 
    minimise the number of crossing edges
    """
    relabeling_dict = {}
    
    sorted_zero = sorted(edges[0],key=lambda x: x[0][1])
    
    cur_index = 0
    seen_old = set([])
    for edge in sorted_zero:
        if edge[0] not in seen_old:
            seen_old.add(edge[0])
            relabeling_dict[edge[0]] = (0,cur_index)
            cur_index += 1
    for node in nodes[0]:
        if node not in relabeling_dict.keys():
            relabeling_dict[node] = (0,cur_index)
            cur_index += 1
    
    
    for i in range(len(edges)):
        basic_edges = []
        for edge in edges[i]:
            basic_edges.append((relabeling_dict[edge[0]],edge[1]))
        sorted_basics = sorted(basic_edges,key=lambda x: x[0][1])
        
        cur_index = 0
        seen_old = set([])
        
        for edge in sorted_basics:
            if edge[1] not in seen_old:
                seen_old.add(edge[1])
                relabeling_dict[edge[1]] = (i+1,cur_index)
                cur_index += 1
        for node in nodes[i+1]:
            if node not in relabeling_dict.keys():
                relabeling_dict[node] = (i+1,cur_index)
                cur_index += 1
    
    final_edges = []
    
    for i in range(len(edges)):
        adding = []
        for edge in edges[i]:
            new_edge = (relabeling_dict[edge[0]],relabeling_dict[edge[1]])
            adding.append(new_edge)
        final_edges.append(adding)
    
    return (nodes,final_edges,relabeling_dict)

def change_labeling(haplotype_data,relabeling_dict):
    """
    Takes as input a list haplotype_data (from the output of 
    generate_haplotypes_all) and changes the labelling of 
    the haplotypes as given by relabeling_dict
    """
    
    new_labeling = []
    
    for i in range(len(haplotype_data)):
        hap_dict = haplotype_data[i][3]
        new_hap_dict = {}
        
        for k in hap_dict.keys():
            new_hap_dict[relabeling_dict[(i,k)][1]] = hap_dict[k]
        
        new_hap_dict = dict(sorted(new_hap_dict.items()))
        new_labeling.append([haplotype_data[0],haplotype_data[1],haplotype_data[2],new_hap_dict])
    
    return new_labeling
    
    
def generate_graph_from_matches(matches_list,
                                layer_dist = 4,
                                vert_dist = 2,
                                planarify=False,
                                size_usage_based=False,
                                hap_usages=None
                                ):
    """
    Takes as input a list of two elements: a list of nodes
    and a list of edges between nodes.
    
    Creates a layered networkx graph of the nodes with the
    edges between them
    
    If planarify = True then function tries to create
    a graph that is as planar as possible
    
    If size_usage_based = True then the size of each node
    is proportional to how many haps it gets used in.
    
    In such a case the additional parameter hap_usages must
    be provided
    """
        
    if size_usage_based == True:
        if hap_usages == None:
            assert False,"Block level haplotype usages not provided"
    
    nodes = matches_list[0]
    edges = matches_list[1]
    
    if planarify:
        pdr = planarify_graph(nodes,edges)
        nodes = pdr[0]
        edges = pdr[1]
        rev_map = {v:k for k,v in pdr[2].items()}
    
    num_layers = len(nodes)
    max_haps_in_layer = 0
    for layer in nodes:
        max_haps_in_layer = max(max_haps_in_layer,len(layer))
    
    nodes_pos = nodes_list_to_pos(nodes,layer_dist=layer_dist,vert_dist=vert_dist)
    
    if size_usage_based:
        node_sizes = []
        for block in range(len(nodes)):
            print("BK",block)
            block_sizes = []
            block_dict = {}
            print(nodes[block])
            for full_node in nodes[block]:
                print(full_node)
                node = full_node[1]
                if not planarify:
                    try:
                        block_dict[node] = 1+hap_usages[block][1][node]
                    except:
                        block_dict[node] = 1
                else:
                    new_label = pdr[2][(block,node)][1]
                    try:
                        block_dict[new_label] = 1+hap_usages[block][1][node]
                    except:
                        block_dict[new_label] = 1
            
            for i in range(len(block_dict)):
                block_sizes.append(block_dict[i])
            node_sizes.append(block_sizes)
        flattened_sizes = [4*x for xs in node_sizes for x in xs]
        use_size = flattened_sizes
    else:
        use_size = 600
    
    
    
    flattened_edges = [x for xs in edges for x in xs] #Flatten the edges list
    flattened_nodes = [x for xs in nodes for x in xs]
    
    G = nx.Graph()
    G.add_nodes_from(flattened_nodes)
    G.add_edges_from(flattened_edges)
    
    fig,ax =plt.subplots()
    nx.draw(G,pos=nodes_pos,node_size=use_size)
    ax.set_xlim(left=0,right=layer_dist*num_layers)
    ax.set_ylim(bottom=-0.5*vert_dist*max_haps_in_layer,top=1+0.5*vert_dist*max_haps_in_layer)
    
    # for i in range(num_layers):
    #     ax.text(x=layer_dist*(0.5+i),y=0.5+0.5*vert_dist*max_haps_in_layer,s=f"{i}",horizontalalignment="center")
    #     if i != 0:
    #         ax.axvline(x=layer_dist*i,color="k",linestyle="--")
    
    fig.set_facecolor("white")
    fig.set_size_inches((num_layers,8))
    plt.show()

#%%
bcf = read_bcf_file("./fish_vcf/AsAc.AulStuGenome.biallelic.bcf.gz")
contigs = bcf.header.contigs
names = bcf.header.samples
#%%
block_size = 100000
shift_size = 50000
chr1 = list(break_contig(bcf,"chr1",block_size=block_size,shift=shift_size))

#%%
starting = 50
ending = 100
#%%
full_data = get_vcf_subset(bcf,"chr1",starting*shift_size,(ending+1)*shift_size)
(full_positions,full_keep_flags,full_reads_array) = cleanup_block_reads(full_data,min_frequency=0)
(full_site_priors,full_probs_array) = reads_to_probabilities(full_reads_array)
#%%
combi = [chr1[i] for i in range(starting,ending)]
my_haps = generate_haplotypes_all(combi)
#%%
my_matches = get_best_matches_all_blocks(my_haps)
#%%
hat = match_haplotypes_by_overlap(my_haps)
ham = match_haplotypes_by_samples(my_haps)
#%%
hax = match_haplotypes_by_overlap_probabalistic(my_haps)
has = match_haplotypes_by_samples_probabalistic(my_haps)
#%%
generate_graph_from_matches(ham[2],planarify=True)
#%%
scor = get_combined_hap_score(hax,has[2])
#%%
finals = generate_chained_block_haplotypes(my_haps,hax[0],scor,6,edge_usage_penalty=10,node_usage_penalty=10)
#%%
sa = combine_all_blocks_to_long_haps(my_haps,finals)
#%%
mark = combine_long_haplotypes(sa[1])
#%%

def get_match_probabilities(full_combined_genotypes,sample_probs,site_locations,
                            keep_flags=None,recomb_rate=10**-8,value_error_rate=10**-3):
    """
    Function which takes in the full (square) array of combined 
    genotypes from haplotypes and runs a HMM-esque process to match
    the sample genotype to the best combination of haplotypes which 
    make it up.
    
    This is so of like Li-Stephens but works on probabalistic genotypes
    rather than fixed ones
    
    Only updates on those sites where keep_flag=1
    """
    
    data_shape = full_combined_genotypes.shape
    
    if keep_flags is None:
        keep_flags = np.array([1 for _ in range(data_shape[2])])
    
    if keep_flags.dtype != int:
        keep_flags = np.array(keep_flags,dtype=int)
        
    #Matrix which records the probabilities of seeing one thing when the true value is another
    #This is used for calculating posteriors later down the line,
    #Moving across rows we see true baseline genotype and across columns we have observed sample genotype
    eps = value_error_rate 
    value_error_matrix = [[(1-eps)**2,2*eps*(1-eps),eps**2],
                          [eps*(1-eps),eps**2+(1-eps)**2,eps*(1-eps)],
                          [eps**2,2*eps*(1-eps),(1-eps)**2]]
    
        

    num_haps = data_shape[0]
    num_geno = data_shape[0]*data_shape[1]
    
    num_sites = data_shape[2]
    
    current_probabilities = (1/num_geno)*np.ones((num_haps,num_haps))

    posterior_probabilities = []
    
    last_site_loc = None
    
    #Iterate backwards initially
    for loc in range(num_sites-1,-1,-1):
        if keep_flags[loc] != 1:
            posterior_probabilities.append(copy.deepcopy(current_probabilities))
        else:
            if last_site_loc == None:
                distance_since_last_site = 0
            else:
                distance_since_last_site = last_site_loc-site_locations[loc]
            last_site_loc = site_locations[loc]
                
            #Fudge prior due to possible recombinations
            updated_prior = recombination_fudge(current_probabilities,
                                                distance_since_last_site,
                                                recomb_rate=recomb_rate)
            
            site_sample_val = sample_probs[loc]
            
            genotype_vals = full_combined_genotypes[:,:,loc,:]

            #Calculate for each genotype combination the probability it equals x and the true data equals y
            all_combs = np.einsum("i,jkl->jkil",site_sample_val,genotype_vals)
            
            #Calculate for each genotype combination the probability of seeing the data given the true underlying value
            prob_data_given_comb = np.einsum("ijkl,kl->ij",all_combs,value_error_matrix)
            
            #Use Bayes's rule to calculate the probability of the combination given data using our prior
            nonorm_prob_comb_given_data = np.einsum("ij,ij->ij",prob_data_given_comb,updated_prior)
            
            #Normalize
            prob_comb_given_data = nonorm_prob_comb_given_data/np.sum(nonorm_prob_comb_given_data)
            
            posterior_probabilities.append(copy.deepcopy(prob_comb_given_data))
            current_probabilities = prob_comb_given_data
    
    #Repeat process going forward
    posterior_probabilities = []
    
    last_site_loc = None
    
    for loc in range(num_sites):
        if keep_flags[loc] != 1:
            posterior_probabilities.append(copy.deepcopy(current_probabilities))
        else:
            if last_site_loc == None:
                distance_since_last_site = 0
            else:
                distance_since_last_site = site_locations[loc]-last_site_loc
            
            last_site_loc = site_locations[loc]
                
            #Fudge prior due to possible recombinations
            updated_prior = recombination_fudge(current_probabilities,
                                                distance_since_last_site,
                                                recomb_rate=recomb_rate)
            site_sample_val = sample_probs[loc]
            
            genotype_vals = full_combined_genotypes[:,:,loc,:]

            #Calculate for each genotype combination the probability it equals x and the true data equals y
            all_combs = np.einsum("i,jkl->jkil",site_sample_val,genotype_vals)
            
            #Calculate for each genotype combination the probability of seeing the data given the true underlying value
            prob_data_given_comb = np.einsum("ijkl,kl->ij",all_combs,value_error_matrix)
            
            #Use Bayes's rule to calculate the probability of the combination given data using our prior
            nonorm_prob_comb_given_data = np.einsum("ij,ij->ij",prob_data_given_comb,updated_prior)
            
            #Normalize
            prob_comb_given_data = nonorm_prob_comb_given_data/np.sum(nonorm_prob_comb_given_data)
            
            posterior_probabilities.append(prob_comb_given_data)
            current_probabilities = prob_comb_given_data
            
    #Convert to upper triangular matrix for easier visualization
    upp_tri_post = []
    for item in posterior_probabilities:
        new = np.triu(item+np.transpose(item)-np.diag(np.diag(item)))
        upp_tri_post.append(new)
        
    max_loc = []
    
    for i in range(len(upp_tri_post)):
        max_loc.append(np.unravel_index(upp_tri_post[i].argmax(), upp_tri_post[i].shape))
    
    return (max_loc,upp_tri_post)
        
def get_full_match_probs(full_combined_genotypes,all_sample_probs,site_locations,
                            keep_flags=None,recomb_rate=10**-8,value_error_rate=10**-3):         
    """
    Multithreaded version to match all samples to their haplotype combinations
    """
    processing_pool = Pool(8)
    
    results = processing_pool.starmap(lambda x: get_match_probabilities(full_combined_genotypes,x,site_locations,
                                        keep_flags=keep_flags,recomb_rate=recomb_rate,value_error_rate=value_error_rate),
                                      zip(all_sample_probs))
    
    return results

def make_heatmap(probs_list):
    """
    Takes as input a list of matrices of probabilities for the hidden
    states of a HMM and makes a heatmap for them
    """
    
    def flattenutrm(matrix):
        """
        flatten an upper triangular matrix
        """
        num_rows = matrix.shape[0]
        
        fltr = []
        
        for i in range(num_rows):
            fltr.extend(matrix[i,i:])
        
        return fltr
    
    flattened_list = []
    

    for i in range(len(probs_list)):
        flattened_list.append(flattenutrm(probs_list[i]))
    flattened_array = np.array(flattened_list).transpose()
    
    hap_names = []
    
    num_haps = len(probs_list[0])
    
    
    for i in range(num_haps):
        for j in range(i,num_haps):
            hap_names.append(f"({i},{j})")

    fig,ax = plt.subplots()
    fig.set_size_inches(11,6)
    sns.heatmap(flattened_array,yticklabels=hap_names)
    plt.title(f"Recombination rate: {recomb_rate}")
    plt.show()
    
def compute_likeliest_path(full_combined_genotypes,sample_probs,site_locations,
                            keep_flags=None,recomb_rate=10**-8,value_error_rate=10**-3):
    """
    Compute the likeliest path explaining the sample using a combination of
    genotypes from full_combined_genotypes via the Viterbi algorithm
    
    We only update on sites which have keep_flags[i] = 1
    """
    
    data_shape = full_combined_genotypes.shape
    
    if keep_flags is None:
        keep_flags = np.array([1 for _ in range(data_shape[2])])
    
    if keep_flags.dtype != int:
        keep_flags = np.array(keep_flags,dtype=int)
        
    #Matrix which records the probabilities of seeing one thing when the true value is another
    #This is used for calculating posteriors later down the line,
    #Moving across rows we see true baseline genotype and across columns we have observed sample genotype
    eps = value_error_rate 
    value_error_matrix = [[(1-eps)**2,2*eps*(1-eps),eps**2],
                          [eps*(1-eps),eps**2+(1-eps)**2,eps*(1-eps)],
                          [eps**2,2*eps*(1-eps),(1-eps)**2]]
    
        

    num_haps = data_shape[0]
    num_rows = data_shape[0]
    num_cols = data_shape[1]
    num_geno = data_shape[0]*data_shape[1]
    
    num_sites = data_shape[2]
    
    starting_probabilities = make_upper_triangular((1/num_geno)*np.ones((num_haps,num_haps)))
    log_current_probabilities = np.log(starting_probabilities)
    
    prev_best = np.empty((data_shape[0],data_shape[1]),dtype=object)
    prev_best[:] = [[(i,j) for j in range(data_shape[1])] for i in range(data_shape[0])]

    log_probs_history = [copy.deepcopy(log_current_probabilities)]
    prev_best_history = [copy.deepcopy(prev_best)]
    
    last_site_loc = None
    
    for loc in range(num_sites):
        if keep_flags[loc] != 1:
            log_probs_history.append(copy.deepcopy(log_current_probabilities))
            prev_best_history.append(copy.deepcopy(prev_best))
        else:
            
            if last_site_loc == None:
                distance_since_last_site = 0
            else:
                distance_since_last_site = site_locations[loc]-last_site_loc
            
        
            #Update last_site_loc
            last_site_loc = site_locations[loc]
            
            #Create transition probability matrix            
            num_possible_switches = num_rows+num_cols-1

            non_switch_prob = (1-recomb_rate)**distance_since_last_site
            
            each_switch_prob= (1-non_switch_prob)/num_possible_switches

            #Account for the fact that we can recombine back to the haplotype pre recombination
            total_non_switch_prob = non_switch_prob+each_switch_prob
            
            transition_probs = np.zeros((num_rows,num_cols,num_rows,num_cols))
            
            for i in range(num_rows):
                for j in range(i,num_cols):
                    transition_probs[i,j,:,j] = each_switch_prob
                    transition_probs[i,j,i,:] = each_switch_prob
                    transition_probs[i,j,i,j] = total_non_switch_prob
                    
                    transition_probs[i,j,:,:] = make_upper_triangular(transition_probs[i,j,:,:])
                    
            
            
            site_sample_val = sample_probs[loc]
            
            genotype_vals = full_combined_genotypes[:,:,loc,:]

            #Calculate for each genotype combination the probability it equals x and the true data equals y
            all_combs = np.einsum("i,jkl->jkil",site_sample_val,genotype_vals)
            
            #Calculate for each genotype combination the probability of seeing the data given the true underlying genotype 
            prob_data_given_comb = np.einsum("ijkl,kl->ij",all_combs,value_error_matrix)
            
            #Calculate the probability of switching to state (k,l) and seeing our observed data given that we started in state (i,j)
            prob_switch_seen = np.einsum("ijkl,kl->ijkl",transition_probs,prob_data_given_comb)
            
            log_prob_switch_seen = np.log(prob_switch_seen)
            
            extended_log_cur_probs = copy.deepcopy(log_current_probabilities)
            extended_log_cur_probs = np.expand_dims(extended_log_cur_probs,2)
            extended_log_cur_probs = np.expand_dims(extended_log_cur_probs,3)
            extended_log_cur_probs = np.repeat(extended_log_cur_probs,num_rows,axis=2)
            extended_log_cur_probs = np.repeat(extended_log_cur_probs,num_cols,axis=3)
            
            log_total_combined_probability = extended_log_cur_probs+log_prob_switch_seen
            
            best_matches = np.empty((data_shape[0],data_shape[1]),dtype=object)
            best_log_probs = np.empty((data_shape[0],data_shape[1]),dtype=float)
            
            for k in range(num_rows):
                for l in range(num_cols):
                    comb_data = log_total_combined_probability[:,:,k,l]
                    max_combination = np.unravel_index(np.argmax(comb_data), comb_data.shape)
                    max_val = comb_data[max_combination]
                    
                    best_matches[k,l] = max_combination
                    best_log_probs[k,l] = max_val
            
            log_current_probabilities = best_log_probs
            prev_best = best_matches
            
            log_probs_history.append(copy.deepcopy(log_current_probabilities))
            prev_best_history.append(copy.deepcopy(prev_best))
    
    #Now we can work backwards to figure out the most likely path for the sample
    reversed_path = []
    
    cur_place = np.unravel_index(np.argmax(log_probs_history[-1]), log_probs_history[-1].shape)
    
    reversed_path.append(cur_place)
    
    for i in range(len(prev_best_history)-1,-1,-1):
        cur_place = prev_best_history[i][cur_place[0],cur_place[1]]
        reversed_path.append(cur_place)
    
    return (reversed_path,log_probs_history,prev_best_history)

def make_heatmap_path(best_path,num_haps):
    """
    Makes a heatmap based on max likelihood path
    """
    
    def flattenutrm(matrix):
        """
        flatten an upper triangular matrix
        """
        num_rows = matrix.shape[0]
        
        fltr = []
        
        for i in range(num_rows):
            fltr.extend(matrix[i,i:])
        
        return fltr
    
    flattened_list = []
    
    for i in range(len(best_path)):
        make_matrix = np.zeros((num_haps,num_haps))
        make_matrix[best_path[i][0],best_path[i][1]] = 1
        flattened_list.append(flattenutrm(make_matrix))
    
    flattened_array = np.array(flattened_list).transpose()
    
    hap_names = []

    for i in range(num_haps):
        for j in range(i,num_haps):
            hap_names.append(f"({i},{j})")

    fig,ax = plt.subplots()
    fig.set_size_inches(11,6)
    sns.heatmap(flattened_array,yticklabels=hap_names)
    plt.title(f"Recombination rate: {recomb_rate}")
    plt.show()


#%%
recomb_rate = 10**-8
resu = compute_likeliest_path(mark,full_probs_array[0],
            full_positions,keep_flags=full_keep_flags,
            recomb_rate=recomb_rate,value_error_rate=10**-3)    

make_heatmap_path(resu[0],6)

#%%
resu[0][8850:9000]
#%%
full_positions[8850:9000]