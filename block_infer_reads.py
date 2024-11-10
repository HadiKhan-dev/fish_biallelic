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
import networkx as nx
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
    
def cleanup_block_reads(block_list,min_frequency=0.1):
    """
    Turn a list of variant records site data into
    a list of site positions and a 3d matrix of the 
    number of reads for ref/alt for that sample at
    that site
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
            allele_depth = row.samples.get(sample).get("AD")
            allele_depth = allele_depth[:2]
            row_vals.append(list(allele_depth))
            
        cleaned_list.append(row_vals)
    
    reads_array = np.ascontiguousarray(np.array(cleaned_list).swapaxes(0,1))

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

def reads_to_probabilities(reads_array,read_error_prob = 0.02):
    """
    Convert a reads array to a probability of the underlying
    genotype being 0, 1 or 2
    """
    reads_sum = np.sum(reads_array,axis=0)
    site_ratios = []
    for i in range(len(reads_sum)):
        site_ratios.append((1+reads_sum[i][1])/(2+reads_sum[i][0]+reads_sum[i][1]))
    
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
            
            log_likelihood_00 = np.log(math.comb(total,ones)/1.0)+zeros*np.log(1-read_error_prob)+ones*np.log(read_error_prob)
            log_likelihood_11 = np.log(math.comb(total,zeros)/1.0)+zeros*np.log(read_error_prob)+ones*np.log(1-read_error_prob)
            log_likelihood_01 = np.log(math.comb(total,ones)/1.0)+total*np.log(1/2)
            
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
        
            log_likelihood_0 = np.log(float(math.comb(total,ones)))+zeros*math.log(1-read_error_prob)+ones*math.log(read_error_prob)
            log_likelihood_1 = np.log(float(math.comb(total,zeros)))+zeros*math.log(read_error_prob)+ones*math.log(1-read_error_prob)
            
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

def combine_haplotypes(initial_haps,
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


def combine_haplotypes_smart(initial_haps,
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
            return {}
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
    
    #Remove any haps that are too close to each other
    (representatives,
      label_mappings) = combine_haplotypes(
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
    

    final_haps = combine_haplotypes_smart(initial_haps,
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
    
    
    (positions,keep_flags,reads_array) = cleanup_block_reads(block_data,min_frequency=0.001)
    
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
        
    return (positions,keep_flags,reads_array,haps_history[-1])
    
    # truncated_haps = truncate_haps(haps_history[-1],matches_history[-1],reads_array,
    #                                max_cutoff_error_increase=max_cutoff_error_increase)
        
    # return (positions,truncated_haps)
    

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
        
        # (positions,keep_flags,reads_arr,found_haps) = generate_haplotypes_block(
        #     chromosome_data[i][0])
        
        # print(len(found_haps))
        # print()
        # total_positions.append(positions)
        # total_keep_flags.append(keep_flags)
        # reads_arrays.append(reads_arr)
        # haps.append(found_haps)
    
    return overall_haplotypes
    #return (total_positions,total_keep_flags,reads_arrays,haps)

def greatest_likelihood_hap(hap):
    """
    Convert a probabalistic hap to a deterministic one by
    choosing the highest probability allele at each site
    """
    return np.argmax(hap,axis=1)

def match_haplotypes_by_overlap(block_level_haps,
                     hap_cutoff_autoinclude=2,
                     hap_cutoff_noninclude=5):
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
    
    for i in range(1,len(block_level_haps)):
        start_position = block_level_haps[i][0][0]
        insertion_point = np.searchsorted(block_level_haps[i-1][0],start_position)
        next_starting.append(insertion_point)
        
    for i in range(len(block_level_haps)):
        block_haps_names.append([])
        block_counts[i] = len(block_level_haps[i][3])
        for name in block_level_haps[i][3].keys():
            block_haps_names[-1].append((i,name))


    for i in range(len(block_level_haps)-1):
        start_point = next_starting[i]
        overlap_length = len(block_level_haps[i][0])-start_point   
        
        cur_ends = {k:block_level_haps[i][3][k][start_point:] for k in block_level_haps[i][3].keys()}
        next_ends = {k:block_level_haps[i+1][3][k][:overlap_length] for k in block_level_haps[i+1][3].keys()}

        matches.append([])
        
        min_expected_connections = max(block_counts[i],block_counts[i+1])
        if block_counts[i] == 0 or block_counts[i+1] == 0:
            min_expected_connections = 0
            
        amount_added = 0
        all_edges_consideration = {}
        
        for first_name in cur_ends.keys(): 
            dist_values = {}

            for second_name in next_ends.keys():
                
                haps_dist = 100*calc_distance(cur_ends[first_name],
                                  next_ends[second_name],
                                  calc_type="haploid")/overlap_length
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
    
    for i in range(1,len(block_level_haps)):
        start_position = block_level_haps[i][0][0]
        insertion_point = np.searchsorted(block_level_haps[i-1][0],start_position)
        next_starting.append(insertion_point)
        
    for i in range(len(block_level_haps)):
        block_haps_names.append([])
        block_counts[i] = len(block_level_haps[i][3])
        for name in block_level_haps[i][3].keys():
            block_haps_names[-1].append((i,name))


    for i in range(len(block_level_haps)-1):
        start_point = next_starting[i]
        overlap_length = len(block_level_haps[i][0])-start_point   
        
        cur_ends = {k:block_level_haps[i][3][k][start_point:] for k in block_level_haps[i][3].keys()}
        next_ends = {k:block_level_haps[i+1][3][k][:overlap_length] for k in block_level_haps[i+1][3].keys()}
           
        similarities = {}
        for first_name in cur_ends.keys(): 

            for second_name in next_ends.keys():
                
                haps_dist = 100*calc_distance(cur_ends[first_name],
                                  next_ends[second_name],
                                  calc_type="haploid")/overlap_length
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
                
            transformed = 50*(val**2)
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
    
def match_haplotypes_by_samples(full_haps_data):
    """
    Alternate method of matching haplotypes in nearby blocks together
    by matching hap A with hap B if the samples which use hap A at its location
    disproportionately use hap B at its location
    
    full_haps_data is the return from a run of generate_haplotypes_all,
    containing info about the positions, read count array and haplotypes
    for each block.
    """
    
    def hap_matching_inner_loop(haps_data):
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

    def hap_matching_comparison(haps_data,matches_data,first_block_index,second_block_index):
        """
        For each hap at the first_block_index block this fn. compares
        where the samples which use that hap end up for the block at
        second_block_index and converts these numbers into percentage
        usages for each hap at index first_block_index
        
        It then choose those edges which have a high enough usage score
        and selectes them for having edges in the final graph
        """
        
        forward_matches = []
        backward_matches = []
        
        first_haps_data = haps_data[first_block_index][3]
        second_haps_data = haps_data[second_block_index][3]
        
        first_matches_data = matches_data[first_block_index]
        second_matches_data = matches_data[second_block_index]
        
        for hap in first_haps_data.keys():
            hap_usages = match_usage_find(hap,first_matches_data,second_matches_data)
            total_matches = sum(hap_usages.values())
            hap_percs = {x:100*hap_usages[x]/total_matches for x in hap_usages.keys()}
            
            if len(hap_percs) > 0:
                best_match = max(hap_percs,key=hap_percs.get)
                best_perc = hap_percs[best_match]
                make_links = []
                
                for k in hap_percs.keys():
                    if hap_percs[k]/best_perc > max_reduction_include or hap_percs[k] > auto_add_perc:
                        make_links.append(k)
                
                for node in make_links:
                    forward_matches.append(((first_block_index,hap),(second_block_index,node)))
        
        for hap in second_haps_data.keys():
            hap_usages = match_usage_find(hap,second_matches_data,first_matches_data)
            total_matches = sum(hap_usages.values())
            hap_percs = {x:100*hap_usages[x]/total_matches for x in hap_usages.keys()}
            
            if len(hap_percs) > 0:
                best_match = max(hap_percs,key=hap_percs.get)
                best_perc = hap_percs[best_match]
                
                make_links = []
                
                for k in hap_percs.keys():
                    if hap_percs[k]/best_perc > max_reduction_include or hap_percs[k] > auto_add_perc:
                        make_links.append(k)
                
                for node in make_links:
                    backward_matches.append(((first_block_index,node),(second_block_index,hap)))
        
        return (forward_matches,backward_matches)
        
    num_blocks = len(full_haps_data)
    
    auto_add_perc = 40
    max_reduction_include = 0.8
    
    all_matches = []
    
    block_haps_names = []
    for i in range(len(full_haps_data)):
        block_haps_names.append([])
        
        for nm in full_haps_data[i][3].keys():
            block_haps_names[-1].append((i,nm))
    
    match_best_results = []
    
    processing_pool = Pool(8)
    
    match_best_results = processing_pool.starmap(hap_matching_inner_loop,
                                                 zip(full_haps_data))
    
    neighbouring_usages = processing_pool.starmap(lambda x,y:
                        hap_matching_comparison(full_haps_data,match_best_results,x,y),
                        zip(list(range(num_blocks-1)),list(range(1,num_blocks))))
    
    forward_matches = [neighbouring_usages[x][0] for x in range(num_blocks-1)]
    backward_matches = [neighbouring_usages[x][1] for x in range(num_blocks-1)]
    

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
    
    def hap_matching_inner_loop(haps_data):
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
                    scaled_val = 50*(min(1,2*hap_percs[other_hap]/100))**2
                    
                    scaled_scores[((first_block_index,hap),
                                   (second_block_index,other_hap))] = scaled_val
                elif other_hap not in hap_percs.keys():
                    scaled_scores[((first_block_index,hap),
                                   (second_block_index,other_hap))] = 0
            forward_scores.update(scaled_scores)
            
        for hap in second_haps_data.keys():
            hap_usages = match_usage_find(hap,first_matches_data,second_matches_data)
            total_matches = sum(hap_usages.values())
            hap_percs = {x:100*hap_usages[x]/total_matches for x in hap_usages.keys()}
            
            scaled_scores = {}
            
            for other_hap in first_haps_data.keys():
                if other_hap in hap_percs.keys():
                    scaled_val = 50*(min(1,2*hap_percs[other_hap]/100))**2
                    
                    scaled_scores[((first_block_index,other_hap),
                                   (second_block_index,hap))] = scaled_val
                elif other_hap not in hap_percs.keys():
                    scaled_scores[((first_block_index,other_hap),
                                   (second_block_index,hap))] = 0
            backward_scores.update(scaled_scores)
        
        return (forward_scores,backward_scores)
        
    num_blocks = len(full_haps_data)
    
    auto_add_perc = 40
    max_reduction_include = 0.8
    
    
    block_haps_names = []
    for i in range(len(full_haps_data)):
        block_haps_names.append([])
        
        for nm in full_haps_data[i][3].keys():
            block_haps_names[-1].append((i,nm))
    
    match_best_results = []
    
    processing_pool = Pool(8)
    
    match_best_results = processing_pool.starmap(hap_matching_inner_loop,
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
                                planarify=False):
    """
    
    Takes as input a list of two elements: a list of nodes
    and a list of edges between nodes.
    
    Creates a layered networkx graph of the nodes with the
    edges between them
    
    If planarify = True then function tries to create
    a graph that is as planar as possible
    """
    
    
    
    nodes = matches_list[0]
    edges = matches_list[1]
    #print(nodes)
    
    if planarify:
        pdr = planarify_graph(nodes,edges)
        nodes = pdr[0]
        edges = pdr[1]
    
    num_layers = len(nodes)
    max_haps_in_layer = 0
    for layer in nodes:
        max_haps_in_layer = max(max_haps_in_layer,len(layer))
    
    nodes_pos = nodes_list_to_pos(nodes,layer_dist=layer_dist,vert_dist=vert_dist)
    
    flattened_edges = [x for xs in edges for x in xs] #Flatten the edges list
    flattened_nodes = [x for xs in nodes for x in xs]
    
    
    G = nx.Graph()
    G.add_nodes_from(flattened_nodes)
    G.add_edges_from(flattened_edges)
    
    fig,ax =plt.subplots()
    nx.draw(G,pos=nodes_pos,node_size=60)
    ax.set_xlim(left=0,right=layer_dist*num_layers)
    ax.set_ylim(bottom=-0.5*vert_dist*max_haps_in_layer,top=1+0.5*vert_dist*max_haps_in_layer)
    
    for i in range(num_layers):
        ax.text(x=layer_dist*(0.5+i),y=0.5+0.5*vert_dist*max_haps_in_layer,s=f"{i}",horizontalalignment="center")
        if i != 0:
            ax.axvline(x=layer_dist*i,color="k",linestyle="--")
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
combi = [chr1[i] for i in range(50,55)]
my_haps = generate_haplotypes_all(combi)
    
#%%
hat = match_haplotypes_by_overlap(my_haps)
ham = match_haplotypes_by_samples(my_haps)
#%%
hax = match_haplotypes_by_overlap_probabalistic(my_haps)
has = match_haplotypes_by_samples_probabalistic(my_haps)
#%%
generate_graph_from_matches(ham[2],planarify=True)
