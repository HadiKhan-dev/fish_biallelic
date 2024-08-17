import numpy as np
import pysam
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.cluster
from sklearn.decomposition import PCA
import hdbscan
from multiprocess import Pool
from sklearn.manifold import TSNE
import itertools
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
        print(starting_index)
        
            
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
    
    processing_pool = Pool(processes=8)
    
    
    dist_matrix = []
    
    dist_matrix = processing_pool.map(
        lambda x : calc_distance_row(x[0],genotype_array,x[1],missing_penalty),
        list(zip(genotype_array,range(num_samples))))
    
    #Convert to array and fill up lower diagonal
    dist_matrix = np.array(dist_matrix)
    dist_matrix = dist_matrix+dist_matrix.T-np.diag(dist_matrix.diagonal())

    del(processing_pool)
    
    return dist_matrix

#%%
def get_heterozygosity(genotype):
    num_sites = len(genotype)
    num_hetero = np.count_nonzero(genotype == 1)
    
    return 100*num_hetero/num_sites

def hdbscan_cluster(dist_matrix,min_cluster_size=2,cluster_selection_method="eom",alpha=1):
    def cluster_details(clustering,cut_distance,min_cluster_size):
        new_clustering = clustering.dbscan_clustering(cut_distance=cut_distance,min_cluster_size=min_cluster_size)

        new_labels = np.array(new_clustering)
        #new_probabilities = np.array(new_clustering.probabilities_)
        
        new_outlier_count = np.count_nonzero(new_labels == -1)
        
        new_clusters = set(new_labels)
        new_clusters.discard(-1)
        
        new_num_clusters = len(new_clusters)
        
        #print(f"EPS: {cut_distance :.2f}, NUM OUTLIERS: {new_outlier_count}, NUM CLUSTERS: {new_num_clusters}")
        
        return new_labels


    #Create clustering object from sklearn
    base_clustering = hdbscan.HDBSCAN(metric="precomputed",
                                      min_cluster_size=min_cluster_size,
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
    print(f"Outliers: {outliers}")
    num_outliers = np.count_nonzero(initial_labels == -1)
    
    all_clusters = set(initial_labels)
    all_clusters.discard(-1)
    
    initial_num_clusters = len(all_clusters)
    
    cluster_eps_dict = {}
    
    for eps in range(600,800):
        cluster_details(base_clustering,1*eps,min_cluster_size=2)
    
    
    return [initial_labels,initial_probabilities,base_clustering]

def get_initial_haps_one(genotype_array,
                     dist_matrix,
                     het_cutoff=10,
                     make_pca=False):
    """
    Get our initial haplotypes by finding high homozygosity samples
    """
    het_vals = np.array([get_heterozygosity(genotype_array[i]) for i in range(len(genotype_array))])
    
    homs_where = np.where(het_vals <= het_cutoff)[0]
    
    homs_array = genotype_array[homs_where]

    dist_submatrix = dist_matrix[np.ix_(homs_where,homs_where)]
    
    print(homs_array.shape)
    print(dist_submatrix.shape)
    
    initial_clusters = hdbscan_cluster(
                            dist_submatrix,
                            min_cluster_size=5,
                            cluster_selection_method="eom",
                            alpha=1.1)
    
    (representatives,filled_data) = fill_nan(
        homs_array,initial_clusters[0],
        cluster_probs=initial_clusters[1],
        prob_cutoff=1)
    
    if make_pca:
        
        pca = PCA(n_components=2)

        pca_proj = pca.fit_transform(filled_data)

        print("PCA SHAPE",pca_proj.shape)
        
        color_palette = sns.color_palette('Paired', 20)
        cluster_colors = [(color_palette[x]) if x >= 0
                          else (0.5, 0.5, 0.5)
                          for x in initial_clusters[0]]


        fig,ax = plt.subplots()
        plt.scatter(*pca_proj.T,c=cluster_colors)
        plt.title(f"PCA, Block size: {block_size}")
        fig.set_size_inches(8,6)
        plt.show()
    
    return representatives

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
        filled_clusters[cluster] = cluster_data
        
        #Find location of all NaN values in the cluster data
        missing = np.where(np.isnan(cluster_data))
        
        #Create a copy of the data for the cluster with NaN values filled in by the representative value for that site
        cluster_filled = np.copy(cluster_data)
        cluster_filled[missing] = np.take(site_rounded,missing[1])
        
        #Populate the filled data portion corresponding to this cluster
        for i in range(len(indices)):
            filled_data[indices[i]] = cluster_filled[i]
            

    #Fill in those rows of filled_data which weren't put into any cluster
    for i in range(len(filled_data)):
        if len(filled_data[i]) == 0:
            filled_data[i] = np.nan_to_num(data_vals[i],nan=0)

    #Return the results
    return (cluster_representatives,np.array(filled_data))

def fix_hap(haplotype):
    """
    Fix a haplotype by removing negative values and
    values bigger than 1 (setting them to 0 and 1
    respectively)
    
    haplotype must be a numpy array
    """
    haplotype[haplotype < 0] = 0
    haplotype[haplotype > 1] = 1
    
    return haplotype.astype(int)

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

def remove_duplicate_haps(haps_list,duplicate_divergence_bound=20):
    """
    Remove haplotypes from a list if they are very similar to
    previous haplotypes in the list
    """
    
    new_list = []
    
    for hap in haps_list:
        keep = True
        for selected in new_list:
            if hap_divergence(hap,selected) < duplicate_divergence_bound:
                keep = False
                break
        if keep:
            new_list.append(hap)
    return new_list

def perc_wrong(array):
    """
    Get the percentage of sites that are wrong (not between 0 and 1)
    for a haplotype
    """

    num_wrong = np.count_nonzero(np.logical_or(np.array(array) < 0,np.array(array) > 1))
    
    num_tot = len(array)
    
    return 100*num_wrong/num_tot

def find_new_haps(cur_haps,diploids,
                  wrong_limit=5,
                  duplicate_divergence_bound=20):
    """
    Takes as input a list of current haplotypes as well as
    a list of genotypes where some of the haplotypes are present
    in some of the genotypes. Tries to then subtract away known
    haplotypes from genotypes to recover new haplotypes and reports on them.
    """
    
    new_haps = []
    
    for i in range(len(diploids)):
        cur_dip = diploids[i]
        
        for j in range(len(cur_haps)):
            cur_hap = cur_haps[j]
            
            difference = cur_dip-cur_hap
            wrongness = perc_wrong(difference)
            
            fixed_hap = fix_hap(difference)
            
            print(i,j)
            print(wrongness)
            
            if wrongness < wrong_limit:
                print(f"Adding: {len(new_haps)}")
                new_haps.append(fixed_hap)
            print()
            
    
    consolidated_haps = list(itertools.chain.from_iterable([cur_haps,new_haps]))
    
    for ha in new_haps:
        print("--------")
        for hb in new_haps:
            print("Div:",hap_divergence(ha,hb))
            
    
    consolidated_haps = remove_duplicate_haps(consolidated_haps,
                                              duplicate_divergence_bound=duplicate_divergence_bound)
    
    print()
    
    print("New:",len(consolidated_haps))
    print()
    
    for i in range(len(consolidated_haps)):
        for j in range(len(consolidated_haps)):
            print(i,j,hap_divergence(consolidated_haps[i],consolidated_haps[j]))
        print()
    #print(basic_haps[1][:100])
    #print(new_haps[1][:100])
    
    
    return consolidated_haps

def match_best(haps_list,diploids):
    """
    Find the best matches of a pair of haploids for each diploid in the diploid list
    """
    
    dips_matches = []
    
    for i in range(len(diploids)):
        cur_best = (None,None)
        cur_div = np.inf
        
        for j in range(len(haps_list)):
            for k in range(len(haps_list)):
                difference = diploids[i]-haps_list[j]-haps_list[k]
                div = magnitude_percentage(difference)
                
                if div < cur_div:
                    cur_div = div
                    cur_best = (j,k)
        dips_matches.append((cur_best,cur_div))
    
    return dips_matches
def split_representatives(cluster_representatives,
                          heterozyg_limit=20,
                          wrong_limit=5,
                          duplicate_divergence_bound=20):
    """
    Try to split cluster representatives into haplotypes
    by looking for low heterozygosity ones 
    """
    
    def halve_genotype(genotype,trim_one=False):
        if trim_one:
            one_replacer = 0
        else:
            one_replacer = 1
        
        haplotype = []
        
        for element in genotype:
            if element == 0:
                haplotype.append(0)
            elif element == 1:
                haplotype.append(one_replacer)
            else:
                haplotype.append(1)
        
        return np.array(haplotype)    
    
    heterozygosities = {i:get_heterozygosity(cluster_representatives[i]) for i in cluster_representatives.keys()}
    representative_list = [cluster_representatives[i] for i in cluster_representatives.keys()]
    
    low_heterog = []
    basic_haps = []
    
    for i in heterozygosities.keys():
        if heterozygosities[i]< heterozyg_limit:
            low_heterog.append(i)
            basic_haps.append(halve_genotype(cluster_representatives[i],trim_one=True))

    print(f"Basic len: {len(basic_haps)}")
    
    #Exit early if we don't find any basic haps
    if len(basic_haps) == 0:
        return []
    
    cur_haps = basic_haps
    
    for i in range(4):
        cur_haps = find_new_haps(cur_haps,
                                 representative_list,
                                 wrong_limit=wrong_limit,
                                 duplicate_divergence_bound=duplicate_divergence_bound)
        
    print(f"Final Num haps: {len(cur_haps)}")
    best_matches = match_best(cur_haps,representative_list)
    
    
    return cur_haps

def demean_array(arr):
    """
    Remove the col mean from an array
    """
    mean = np.nanmean(arr,axis=0)
    
    diff = arr-mean
    
    return diff

def fit_haps_to_genotypes(haps,genotype_data):
    return match_best(haps,genotype_data)

#%%
def generate_haplotypes_block(block_data,make_pca=False):
    wrong_limit = 5
    
    (positions,genotype_array) = cleanup_block(block_data)
    
    genotype_avg = np.nanmean(genotype_array,axis=0)
    demeaned_genotypes = genotype_array-genotype_avg
    remaining = 100*np.nanmean(abs(demeaned_genotypes),axis=1)/2
    prefit_wrong = np.mean(remaining)
    
    dist_matrix = generate_distance_matrix(
        genotype_array,
        missing_penalty="None")
    
    done = False
    
    while not done:
        clustering_result = hdbscan_cluster(
                                dist_matrix,
                                min_cluster_size=5,
                                cluster_selection_method="eom",
                                alpha=1.1)

    
        (representatives,filled_data) = fill_nan(
            genotype_array,clustering_result[0],
            cluster_probs=clustering_result[1],
            prob_cutoff=1)

        splits = split_representatives(
            representatives,
            wrong_limit=wrong_limit,
            duplicate_divergence_bound=20)
        
        if len(splits) > 0:
            done = True
    
    genotype_matches = fit_haps_to_genotypes(splits,genotype_array)
    
    if make_pca:
        
        pca = PCA(n_components=2)

        pca_proj = pca.fit_transform(filled_data)

        color_palette = sns.color_palette('Paired', 20)
        cluster_colors = [(color_palette[x]) if x >= 0
                          else (0.5, 0.5, 0.5)
                          for x in clustering_result[0]]


        fig,ax = plt.subplots()
        plt.scatter(*pca_proj.T,c=cluster_colors)
        plt.title(f"PCA, Block size: {block_size}")
        fig.set_size_inches(8,6)
        plt.show()
        
    return splits

def generate_haplotypes_all(chromosome_data):
    haps = []
    
    for i in range(len(chromosome_data)):
        print(i,len(chromosome_data))
        
        haps.append(generate_haplotypes_block(chromosome_data[i]))
    
    return haps
#%%
bcf = read_bcf_file("./AsAc.AulStuGenome.biallelic.bcf.gz")
contigs = bcf.header.contigs
names = bcf.header.samples
#%%
block_size = 100000
chr1 = list(break_contig(bcf,"chr1",block_size=block_size,shift=block_size))
print("OneDone")
#%%
generate_haplotypes_all(chr1)
#%%
test = chr1[23]
res = generate_haplotypes_block(test,make_pca=True)
(positions,genotype_array) = cleanup_block(test)

genotype_avg = np.nanmean(genotype_array,axis=0)
demeaned_genotypes = genotype_array-genotype_avg
remaining = 100*np.nanmean(abs(demeaned_genotypes),axis=1)/2


fits = fit_haps_to_genotypes(res,genotype_array)
losses = [x[1] for x in fits]
plt.hist(remaining)
plt.hist(losses)
plt.show()
print(fits)
print()
print(np.mean(remaining))
print(np.mean(losses))

#%%
def get_initial_haps_two(genotype_array,
                     het_cutoff=10,
                     make_pca=False):
    """
    Get our initial haplotypes by finding high homozygosity samples
    """
    het_vals = np.array([get_heterozygosity(genotype_array[i]) for i in range(len(genotype_array))])
    
    homs_where = np.where(het_vals <= het_cutoff)[0]
    
    homs_array = genotype_array[homs_where]
    
    homs_array[homs_array == 1] = np.nan
    
    print(homs_array.shape)
    
    homs_array = homs_array/2
    
    dist_submatrix = generate_distance_matrix(
            homs_array,
            missing_penalty="None")
    
    initial_clusters = hdbscan_cluster(
                            dist_submatrix,
                            min_cluster_size=2,
                            cluster_selection_method="eom",
                            alpha=3.5)
    
    (representatives,filled_data) = fill_nan(
        homs_array,initial_clusters[0])
    
    print(initial_clusters[0])
    
    if make_pca:
        
        pca = PCA(n_components=2)

        pca_proj = pca.fit_transform(filled_data)

        second_clustering = hdbscan.HDBSCAN(metric="l1",
                                          min_cluster_size=5)
        
        second_clustering.fit(dist_submatrix)
        
        second_labels = np.array(second_clustering.labels_)
        print("Num clusters: ",max(second_labels)+1)
        
        (new_representatives,filled_data) = fill_nan(
            homs_array,second_labels)
        
        print("PCA SHAPE",pca_proj.shape)
        
        color_palette = sns.color_palette('Paired', 20)
        cluster_colors = [(color_palette[x]) if x >= 0
                          else (0.5, 0.5, 0.5)
                          for x in second_labels]
        

        fig,ax = plt.subplots()
        plt.scatter(*pca_proj.T,c=cluster_colors)
        plt.title(f"PCA, Block size: {block_size}")
        fig.set_size_inches(8,6)
        plt.show()
    
    return (representatives,new_representatives)
#%%
test = chr1[422]
(positions,genotype_array) = cleanup_block(test)
dist_matrix = generate_distance_matrix(
        genotype_array,
        missing_penalty="None")
    
hets = [get_heterozygosity(genotype_array[i]) for i in range(len(genotype_array))]
plt.plot(np.array(sorted(hets)))
plt.show()
print(np.array(sorted(hets)[:20]))

#firsts = get_initial_haps_one(genotype_array,dist_matrix,make_pca=True)
seconds = get_initial_haps_two(genotype_array,make_pca=True)