import numpy as np
import pysam
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

def get_vcf_subset(vcf_data,contig_name,start_index=0,end_index=None):
    """
    Simple function to extract records from a portion of a contig
    between two positions
    
    end_index=None means we iterate until the end of the data
    """
    
    data_list = []
    
    full_data = list(vcf_data.fetch(contig_name))
    
    for i in range(len(full_data)):
        record = full_data[i]
        if record.pos < start_index:
            continue
        if end_index != None:
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

def cleanup_block_reads_list(full_list_of_blocks):
    """
    Function which applies cleanup_block_reads to the first
    element of every single element of full_list_of_blocks
    """
    
    cleaned_pos_list = []
    cleaned_keep_flags = []
    cleaned_reads_arrays = []
    
    for i in range(len(full_list_of_blocks)):
        (a,b,c) = cleanup_block_reads(full_list_of_blocks[i][0])
        
        cleaned_pos_list.append(a)
        cleaned_keep_flags.append(b)
        cleaned_reads_arrays.append(c)
    
    return (cleaned_pos_list,cleaned_keep_flags,cleaned_reads_arrays)

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


