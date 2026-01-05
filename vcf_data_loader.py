import numpy as np
import warnings
from cyvcf2 import VCF
from multiprocess import Pool
from functools import partial

warnings.filterwarnings("ignore")
np.seterr(divide='ignore', invalid="ignore")

class GenomicData:
    """
    A container class for the processed VCF data blocks.
    
    Attributes:
        positions (list of np.ndarray): List where each element is the genomic positions for a block.
        keep_flags (list of np.ndarray): List where each element is the boolean mask for useful sites in a block.
        reads (list of np.ndarray): List where each element is the (Samples x Sites x 2) read count matrix.
    """
    def __init__(self, positions, keep_flags, reads):
        self.positions = positions
        self.keep_flags = keep_flags
        self.reads = reads

    def __len__(self):
        """Returns the number of data blocks loaded."""
        return len(self.positions)

    def __getitem__(self, idx):
        """
        Allows indexing or slicing. 
        Returns a tuple of (pos, reads, flags) for the requested index.
        """
        return self.positions[idx], self.reads[idx], self.keep_flags[idx]

    def __iter__(self):
        """Allows iterating over blocks as tuples: (pos, reads, flags)"""
        for i in range(len(self)):
            yield self[i]

def generate_block_coordinates(vcf_file_path, contig,
                               block_size=100000, shift=50000):
    """
    Generator that calculates coordinates for blocks.
    Instead of reading the data, it yields (chrom, start, end) tuples.
    """
    vcf = VCF(vcf_file_path)
    
    # Attempt to get contig length from header, fallback to arbitrary large number if missing
    try:
        contig_len = dict(zip(vcf.seqnames, vcf.seqlens))[contig]
    except:
        contig_len = 300_000_000 
    
    vcf.close() # Close handle immediately
    
    cur_start = 0
    while cur_start < contig_len:
        cur_end = cur_start + block_size
        yield (vcf_file_path, contig, cur_start, cur_end)
        cur_start += shift

def process_single_block(vcf_path, chrom, start, end, 
                                min_frequency=0.0, 
                                read_error_prob=0.02, 
                                min_total_reads=5):
    """
    WORKER FUNCTION:
    Opens the VCF, queries the specific region, and returns Numpy arrays.
    """
    
    vcf = VCF(vcf_path)
    region_str = f"{chrom}:{start}-{end}"
    
    # Pre-allocate lists
    positions_list = []
    early_flags_list = []
    reads_list = []
    
    try:
        # Iterate over the region
        for variant in vcf(region_str):
            
            # 1. Check coordinates (Your original design choice: Strict [start, end) interval)
            if variant.POS < start or variant.POS >= end:
                continue

            # 2. Basic Filtering (AF)
            af = variant.INFO.get('AF')
            if isinstance(af, (list, tuple)):
                af = af[0]
            elif af is None:
                af = 0.0
                
            if min_frequency <= af <= (1.0 - min_frequency):
                early_flags_list.append(1)
            else:
                early_flags_list.append(0)
                
            # 3. Extract Reads (AD)
            ad = variant.format('AD')
            
            if ad is None:
                n_samples = len(vcf.samples)
                ad = np.zeros((n_samples, 2), dtype=np.int32)
            elif ad.shape[1] > 2:
                # If multi-allelic, just take Ref and First Alt
                ad = ad[:, :2]
                
            reads_list.append(ad)
            positions_list.append(variant.POS)
    except Exception:
        # Gracefully handle cases where region queries might fail
        pass

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

def cleanup_block_reads_list(vcf_file_path, contig, 
                                    start_block_idx=0, end_block_idx=None,
                                    block_size=100000, shift_size=50000,
                                    num_processes=16,
                                    min_frequency=0.0, 
                                    read_error_prob=0.02, 
                                    min_total_reads=5):
    """
    Multiprocessing driver.
    Returns a GenomicData object containing lists of results.
    """
    
    # 1. Generate ALL coordinates
    all_coords = list(generate_block_coordinates(vcf_file_path, contig, 
                                                 block_size=block_size, shift=shift_size))
    
    # 2. Slice the coordinates
    if end_block_idx is None:
        selected_coords = all_coords[start_block_idx:]
    else:
        selected_coords = all_coords[start_block_idx:end_block_idx]
        
    # 3. Run Multiprocessing
    # Use partial to pass the keyword arguments to the worker
    worker = partial(process_single_block, 
                     min_frequency=min_frequency, 
                     read_error_prob=read_error_prob, 
                     min_total_reads=min_total_reads)

    with Pool(processes=num_processes) as pool:
        results = pool.starmap(worker, selected_coords)
    
    # 4. Unpack Results
    # FIX: We NO LONGER filter out empty blocks. 
    # This keeps the list index aligned with the spatial block index.
    cleaned_pos_list = [r[0] for r in results]
    cleaned_keep_flags = [r[1] for r in results]
    cleaned_reads_arrays = [r[2] for r in results]
    
    # 5. Return the new class container
    return GenomicData(cleaned_pos_list, cleaned_keep_flags, cleaned_reads_arrays)

def resample_reads_array(reads_array, resample_depth):
    """
    Resample the reads array to a reduced read depth.
    Useful for simulation or testing robustness.
    """
    array_shape = reads_array.shape
    if array_shape[0] == 0 or array_shape[1] == 0:
        return reads_array

    starting_depth = np.sum(reads_array, axis=None) / (array_shape[0] * array_shape[1])
    
    # Handle zero depth case
    if starting_depth == 0:
        return reads_array
        
    print("Initial Depth was:", starting_depth)
    
    cutoff = resample_depth / starting_depth
    
    if cutoff > 1:
        # FIX: Raise proper error instead of assert False
        raise ValueError(f"Cannot resample to a depth ({resample_depth}) higher than original data ({starting_depth}).")
        
    resampled_array = np.random.binomial(reads_array, cutoff)
    return resampled_array