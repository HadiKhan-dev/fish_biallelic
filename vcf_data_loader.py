import numpy as np
import pysam
import warnings
from cyvcf2 import VCF
from multiprocess import Pool

warnings.filterwarnings("ignore")
np.seterr(divide='ignore',invalid="ignore")
pass
#%%

def generate_block_coordinates(vcf_file_path, contig,
                               block_size=100000, shift=50000):
    """
    Generator that calculates coordinates for blocks.
    Instead of reading the data, it yields (chrom, start, end) tuples.
    This is instant.
    """
    # We open VCF just to check the contig length (optional, but good practice)
    # If you don't care about the exact end, you can just loop until no records found.
    vcf = VCF(vcf_file_path)
    # cyvcf2 doesn't always expose contig lengths easily in the header depending on VCF version
    # So we will use a simple coordinate sliding window approach.
    
    # You might need to know the max length of the contig to stop correctly,
    # or handle the empty return in the worker. 
    # For now, we assume a simpler logic: we iterate until we hit an arbitrary large number
    # or you can look up the length from vcf.seqlens if available.
    
    try:
        contig_len = dict(zip(vcf.seqnames, vcf.seqlens))[contig]
    except:
        contig_len = 300_000_000 # Fallback for large chromosomes if header missing
    
    cur_start = 0
    while cur_start < contig_len:
        cur_end = cur_start + block_size
        yield (vcf_file_path, contig, cur_start, cur_end)
        cur_start += shift

def process_single_block_cyvcf2(vcf_path, chrom, start, end, 
                                min_frequency=0.0, 
                                read_error_prob=0.02, 
                                min_total_reads=5):
    """
    WORKER FUNCTION:
    Opens the VCF, queries the specific region, and returns Numpy arrays.
    """
    
    vcf = VCF(vcf_path)
    region_str = f"{chrom}:{start}-{end}"
    
    # Pre-allocate lists (faster than appending to numpy arrays dynamically)
    positions_list = []
    early_flags_list = []
    reads_list = []
    
    # Iterate over the region
    # cyvcf2 is very fast here
    for variant in vcf(region_str):
        
        # 1. Check coordinates (cyvcf2 region query is precise, but good to be safe)
        if variant.POS < start or variant.POS >= end:
            continue

        # 2. Basic Filtering (AF)
        af = variant.INFO.get('AF')
        # Handle cases where AF is a list or scalar
        if isinstance(af, (list, tuple)):
            af = af[0]
        elif af is None:
            af = 0.0
            
        if min_frequency <= af <= (1.0 - min_frequency):
            early_flags_list.append(1)
        else:
            early_flags_list.append(0)
            
        # 3. Extract Reads (AD)
        # format('AD') returns a numpy array of shape (n_samples, n_alleles)
        # This is the fastest way to get data out of cyvcf2
        ad = variant.format('AD')
        
        if ad is None:
            # Fallback for variants missing AD field
            # Create dummy zeros
            n_samples = len(vcf.samples)
            ad = np.zeros((n_samples, 2), dtype=np.int32)
        elif ad.shape[1] > 2:
            # If multi-allelic, just take Ref and First Alt
            ad = ad[:, :2]
            
        reads_list.append(ad)
        positions_list.append(variant.POS)

    # Clean up VCF handle
    vcf.close()

    # --- If block was empty ---
    if not positions_list:
        return (np.array([]), np.array([]), np.empty((0,0,0)))

    # --- Finalize Numpy Arrays ---
    positions = np.array(positions_list, dtype=np.int64)
    early_keep_flags = np.array(early_flags_list, dtype=np.int32)
    
    # Stack reads: (Sites, Samples, 2)
    reads_array_sites_first = np.stack(reads_list)
    
    # Check for negative values (cyvcf2 represents missing data as -1 sometimes)
    reads_array_sites_first[reads_array_sites_first < 0] = 0
    
    # Swap to (Samples, Sites, 2)
    reads_array = np.ascontiguousarray(reads_array_sites_first.swapaxes(0, 1))
    
    # --- Vectorized Flags Logic ---
    num_samples = reads_array.shape[0]
    total_read_pos = np.sum(reads_array, axis=(0, 2))
    
    late_keep_flags = (total_read_pos >= max(min_total_reads, read_error_prob * num_samples))
    keep_flags = np.bitwise_and(early_keep_flags, late_keep_flags).astype(int)
    
    return (positions, keep_flags, reads_array)


def cleanup_block_reads_list_cyvcf2(vcf_file_path, contig, 
                                    start_block_idx=0, end_block_idx=None,
                                    block_size=100000, shift_size=50000,
                                    num_processes=8):
    """
    Multiprocessing driver that calculates coordinates and processes specific blocks.
    """
    
    # 1. Generate ALL coordinates (Instant)
    # We convert to a list so we can slice it (e.g., [0:854])
    all_coords = list(generate_block_coordinates(vcf_file_path, contig, 
                                                 block_size=block_size, shift=shift_size))
    
    # 2. Slice the coordinates (Match your 'starting' and 'ending' logic)
    if end_block_idx is None:
        selected_coords = all_coords[start_block_idx:]
    else:
        selected_coords = all_coords[start_block_idx:end_block_idx]
        
    # 3. Run Multiprocessing
    # We pass the file path string, NOT the file object
    with Pool(processes=num_processes) as pool:
        results = pool.starmap(process_single_block_cyvcf2, selected_coords)
    
    # 4. Unpack Results
    # Filter out empty blocks if any
    cleaned_pos_list = [r[0] for r in results if r[0].size > 0]
    cleaned_keep_flags = [r[1] for r in results if r[0].size > 0]
    cleaned_reads_arrays = [r[2] for r in results if r[0].size > 0]
    
    return (cleaned_pos_list, cleaned_keep_flags, cleaned_reads_arrays)
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


