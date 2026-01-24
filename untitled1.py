import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import pedigree_inference

def evaluate_phase_correction_accuracy(multi_contig_results, sample_names):
    """
    Evaluates the accuracy of the phase correction against Ground Truth.
    
    Metrics:
    1. Phasing Accuracy: % of sites where (Track0, Track1) matches (TruthA, TruthB).
       Automatically detects if TruthA maps to Track0 or Track1 globally (Label Swapping).
    2. Switch Errors: Number of times the mapping flips along the chromosome.
    """
    stats = []
    print("\n" + "="*60)
    print("EVALUATION: 8-State Trio HMM Correction")
    print("="*60)

    for r_name in multi_contig_results.keys():
        data = multi_contig_results[r_name]
        
        if 'truth_painting' not in data or 'corrected_painting' not in data:
            continue
            
        # 1. Discretize for comparison (150 SNPs/bin)
        truth_painting = data['truth_painting']
        corrected_painting = data['corrected_painting']
        
        # Grid: (Samples, Bins, 2)
        grid_truth, _, _ = pedigree_inference.discretize_paintings(truth_painting, snps_per_bin=150)
        grid_corr, _, _ = pedigree_inference.discretize_paintings(corrected_painting, snps_per_bin=150)
        
        for i, name in enumerate(sample_names):
            # Extract tracks (-1 = missing)
            T0, T1 = grid_truth[i, :, 0], grid_truth[i, :, 1]
            C0, C1 = grid_corr[i, :, 0], grid_corr[i, :, 1]
            
            # Valid bins only
            valid = (T0 != -1) & (T1 != -1) & (C0 != -1) & (C1 != -1)
            n_valid = np.sum(valid)
            
            if n_valid == 0: continue
            
            t0, t1 = T0[valid], T1[valid]
            c0, c1 = C0[valid], C1[valid]
            
            # Check Match Configurations
            # Config A: Truth 0 -> Corr 0 (Direct match)
            match_A = (t0 == c0) & (t1 == c1)
            
            # Config B: Truth 0 -> Corr 1 (Parent label swap / Phase Inverted)
            match_B = (t0 == c1) & (t1 == c0)
            
            # Determine Global Phase (Whichever matches better overall)
            score_A = np.sum(match_A)
            score_B = np.sum(match_B)
            
            accuracy = max(score_A, score_B) / n_valid * 100.0
            
            # Calculate Switch Errors
            # State 1 = Config A is correct, State -1 = Config B is correct
            state_vec = np.zeros(n_valid, dtype=int)
            state_vec[match_A & ~match_B] = 1
            state_vec[match_B & ~match_A] = -1
            
            # Remove ambiguous spots
            distinct = state_vec[state_vec != 0]
            switches = 0
            if len(distinct) > 1:
                switches = np.sum(distinct[:-1] != distinct[1:])
                
            stats.append({
                'Region': r_name,
                'Sample': name,
                'Accuracy': accuracy,
                'Switch_Errors': switches
            })
            
    df = pd.DataFrame(stats)
    
    # Summary
    print(f"Global Average Accuracy: {df['Accuracy'].mean():.2f}%")
    print(f"Global Average Switches: {df['Switch_Errors'].mean():.3f} per chrom")
    # Using 99.9 to allow for tiny edge discretization noises
    print(f"Perfect Samples: {len(df[df['Accuracy'] > 99.9])} / {len(df)}")
    
    # Show worst offenders
    print("\nWorst Samples:")
    print(df.sort_values("Accuracy").head(5))
    
    return df

def visualize_correction_result(multi_contig_results, sample_names, region, target_sample):
    """
    Visual comparison of Truth, Input (Control), and Output (Corrected).
    """
    print(f"\n=== VISUAL CHECK: {target_sample} on {region} ===")
    
    try:
        idx = sample_names.index(target_sample)
    except:
        print("Sample not found.")
        return

    data = multi_contig_results[region]
    
    # Handle missing keys gracefully
    truth = data.get('truth_painting')
    ctrl = data.get('control_painting')
    corr = data.get('corrected_painting')
    
    if not truth or not corr:
        print("Missing painting data.")
        return
        
    paintings = {'Truth': truth, 'Original': ctrl, 'Corrected': corr}
    
    # Setup Figure
    fig, axes = plt.subplots(3, 1, figsize=(15, 6), sharex=True)
    
    # Determine Color Palette (Global max ID)
    max_id = 0
    for p_list in paintings.values():
        if p_list:
            for chunk in p_list[idx].chunks:
                max_id = max(max_id, chunk.hap1, chunk.hap2)
    
    base_palette = sns.color_palette("tab20", 20)
    def get_color(fid):
        return base_palette[fid % 20]
    
    def plot_track(ax, obj, title):
        if obj is None: return
        
        sample_obj = obj[idx]
        end_pos = sample_obj.chunks[-1].end if sample_obj.chunks else 0
        ax.set_xlim(0, max(1, end_pos))
        ax.set_ylim(0, 2)
        ax.set_yticks([0.5, 1.5])
        ax.set_yticklabels(["Trk 0", "Trk 1"])
        ax.set_title(title)
        
        for c in sample_obj.chunks:
            width = c.end - c.start
            if width <= 0: continue
            
            # Track 0
            rect0 = mpatches.Rectangle((c.start, 0), width, 1, facecolor=get_color(c.hap1), edgecolor='none')
            ax.add_patch(rect0)
            if width > 100000: 
                ax.text(c.start + width/2, 0.5, str(c.hap1), ha='center', va='center', fontsize=8, color='white')
            
            # Track 1
            rect1 = mpatches.Rectangle((c.start, 1), width, 1, facecolor=get_color(c.hap2), edgecolor='none')
            ax.add_patch(rect1)
            if width > 100000:
                ax.text(c.start + width/2, 1.5, str(c.hap2), ha='center', va='center', fontsize=8, color='white')

    plot_track(axes[0], paintings['Truth'], f"Ground Truth: {target_sample}")
    plot_track(axes[1], paintings['Original'], f"Input (Control): Unphased / Switch Errors")
    plot_track(axes[2], paintings['Corrected'], f"Output (Corrected): 8-State Trio HMM")
    
    plt.tight_layout()
    plt.show()

# === RUN ===
df_results = evaluate_phase_correction_accuracy(multi_contig_results, sample_names)
visualize_correction_result(multi_contig_results, sample_names, "chr7", "F2_20")