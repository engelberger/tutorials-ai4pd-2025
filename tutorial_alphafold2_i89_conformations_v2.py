# %% [markdown]
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/engelberger/tutorials-ai4pd-2025/blob/main/tutorial_alphafold2_i89_conformations_v2.ipynb)
# 
# # Tutorial: Prediction of Protein Structures and Multiple Conformations using AlphaFold2
# 
# ## Clean Implementation Using AF2 Utils Package
# 
# **Duration:** 90 minutes  
# **Instructor:** Felipe Engelberger  
# **Date:** AI4PD Workshop 2025
# 
# ---
# 
# ## Learning Objectives
# 
# By the end of this tutorial, you will understand:
# 
# 1. **MSA's role in conformation selection**: How evolutionary information biases AlphaFold2 predictions
# 2. **Recycling mechanics**: How iterative refinement affects structure quality and conformation
# 3. **Conformational sampling strategies**: Practical techniques using dropout and MSA subsampling
# 4. **Structure analysis tools**: RMSD calculations, visualization, and ensemble analysis
# 5. **Real-world applications**: When and how to apply these techniques to proteins of interest
# 
# ## Tutorial Overview
# 
# We'll use the **i89 protein** as our model system. This 96-residue protein exhibits distinct conformational states that AlphaFold2 can capture through different prediction strategies:
# 
# - **State 1**: The conformation typically predicted with full MSA
# - **State 2**: An alternative conformation accessible without MSA
# 
# We have experimental structures for both states (`state1.pdb` and `state2.pdb`) for validation.
# 

# %% [markdown]
# ## Section 1: Environment Setup
# 
# First, let's set up our environment with the AF2 Utils package that provides a clean wrapper around ColabDesign.
# 

# %%
%%time
#@title Install Dependencies and Import AF2 Utils
#@markdown This cell handles all setup automatically

import os
import sys
import warnings
warnings.filterwarnings('ignore')

# Check if running in Colab
IN_COLAB = 'google.colab' in sys.modules

print("="*60)
print("ALPHAFOLD2 TUTORIAL SETUP")
print("="*60)

# Download af2_utils.py if not present
if not os.path.exists("af2_utils.py"):
    print("\nDownloading af2_utils.py...")
    os.system("wget -q https://raw.githubusercontent.com/engelberger/tutorials-ai4pd-2025/main/af2_utils.py")
    print("  - af2_utils.py downloaded")

# Download logmd_utils.py if not present
if not os.path.exists("logmd_utils.py"):
    print("\nDownloading logmd_utils.py...")
    os.system("wget -q https://raw.githubusercontent.com/engelberger/tutorials-ai4pd-2025/main/logmd_utils.py")
    print("  - logmd_utils.py downloaded")

# Import af2_utils
print("\nImporting AF2 Utils...")
import af2_utils as af2
print(f"  - AF2 Utils v{af2.__version__} loaded")

# Check installation status
print("\nChecking dependencies...")
status = af2.check_installation(verbose=False)
for component, installed in status.items():
    symbol = "+" if installed else "-"
    print(f"  {symbol} {component}: {'ready' if installed else 'missing'}")

# Check LogMD availability
print("\nChecking LogMD availability...")
logmd_available = af2.check_logmd()
if logmd_available:
    print("  + LogMD: available for interactive 3D visualization")
else:
    print("  - LogMD: not available (optional)")
    print("    Install with: pip install logmd")
    print("    Tutorial works without LogMD, but you'll miss interactive features!")

# Install missing dependencies if needed
missing = [k for k, v in status.items() if not v and k != 'environment_setup']
if missing:
    print(f"\nInstalling missing dependencies...")
    af2.install_dependencies(
        install_colabdesign='colabdesign' in missing,
        install_hhsuite='hhsuite' in missing,
        download_params='alphafold_params' in missing,
        verbose=True
    )

# Setup environment
print("\nConfiguring environment...")
af2.setup_environment(verbose=False)
print("  - JAX memory and environment configured")

print("\n" + "="*60)
print("SETUP COMPLETE - Ready for predictions!")
print("="*60)

# %%
#@title Import Additional Libraries
import numpy as np
import matplotlib.pyplot as plt
from Bio import PDB
from pathlib import Path
import json

print("Libraries imported successfully")


# %% [markdown]
# ## Section 2: The i89 Protein - Our Model System
# 
# The i89 protein is a 96-residue protein that can adopt multiple conformational states. We'll use it to demonstrate how AlphaFold2's predictions can be influenced by MSA depth, recycling, and sampling parameters.
# 

# %%
#@title Define i89 Sequence and Load Reference Structures

# i89 protein sequence (96 residues)
I89_SEQUENCE = "GSHMASMEDLQAEARAFLSEEMIAEFKAAFDMFDADGGGDISYKAVGTVFRMLGINPSKEVLDYLKEKIDVDGSGTIDFEEFLVLMVYIMKQDA"

print("i89 protein statistics:")
print(f"  Length: {len(I89_SEQUENCE)} residues")
print(f"  Sequence: {I89_SEQUENCE[:30]}...{I89_SEQUENCE[-20:]}")

# Check if reference structures exist, download if needed
if not os.path.exists("state1.pdb") or not os.path.exists("state2.pdb"):
    print("\nDownloading reference structures...")
    os.system("wget -q https://raw.githubusercontent.com/engelberger/tutorials-ai4pd-2025/main/state1.pdb")
    os.system("wget -q https://raw.githubusercontent.com/engelberger/tutorials-ai4pd-2025/main/state2.pdb")
    print("  - Reference structures downloaded")
else:
    print("\nReference structures found:")
    print("  - state1.pdb: Conformation typically predicted with MSA")
    print("  - state2.pdb: Alternative conformation accessible without MSA")

# Calculate RMSD between reference states
state1_coords = af2.load_pdb_coords("state1.pdb")
state2_coords = af2.load_pdb_coords("state2.pdb")
ref_rmsd = af2.calculate_rmsd(state1_coords, state2_coords)

print(f"\nRMSD between reference states: {ref_rmsd:.2f} Angstrom")
print("This indicates significant conformational difference!")


# %% [markdown]
# ## Section 2.5: Understanding the MSA - The Evolutionary Context
# 
# Before we predict structures, let's explore the Multiple Sequence Alignment (MSA) that provides evolutionary information to AlphaFold2. This MSA contains homologous sequences that help identify conserved regions and coevolving residues.
# 
# Understanding the MSA is crucial because:
# - It reveals evolutionary constraints on the protein
# - Coevolving residues indicate functional coupling
# - MSA depth directly affects AlphaFold2's conformational predictions
# - Strong evolutionary signals guide predictions toward functional states
# 

# %%
%%time
#@title Generate MSA for i89 Protein
#@markdown This generates the MSA without running structure prediction yet

print("="*60)
print("GENERATING MSA FOR i89")
print("="*60)
print("Searching for homologous sequences using MMseqs2...")
print("This may take 1-2 minutes...\n")

# Generate MSA independently of structure prediction
msa_full, del_matrix = af2.get_msa(
    sequences=[I89_SEQUENCE],
    jobname="i89_msa_analysis",
    mode="unpaired",
    cov=50,
    id=90,
    max_msa=512,
    verbose=True
)

# Save for later use in predictions
np.save("i89_msa.npy", msa_full)
np.save("i89_del_matrix.npy", del_matrix)

print("\n" + "="*60)
print("MSA GENERATION COMPLETE")
print("="*60)
print(f"  Number of sequences found: {len(msa_full)}")
print(f"  Sequence length: {msa_full.shape[1]} residues")
print(f"  MSA saved to: i89_msa.npy")
print(f"  Deletion matrix saved to: i89_del_matrix.npy")
print("\nThis MSA will be reused in all subsequent predictions!")
print("Now let's explore what evolutionary information it contains...")


# %%
#@title Quick Comparison: MSA vs Single Sequence Prediction
#@markdown This cell runs a single sequence prediction and compares it with the MSA prediction

print("="*60)
print("COMPARING MSA vs SINGLE SEQUENCE PREDICTIONS")
print("="*60)
print("Running single sequence prediction for comparison...\n")

# Setup model for single sequence
model_single = af2.setup_model(I89_SEQUENCE, verbose=False)
msa_single, del_single = af2.create_single_sequence_msa(I89_SEQUENCE)

# Run single sequence prediction
result_single = af2.predict_with_recycling(
    model_single,
    msa=msa_single,
    deletion_matrix=del_single,
    max_recycles=3,
    seed=0,
    verbose=False
)

# Calculate RMSDs for single sequence
pred_ca_single = result_single['structure'][:, 1, :]
rmsd_state1_single = af2.calculate_rmsd(pred_ca_single, state1_coords)
rmsd_state2_single = af2.calculate_rmsd(pred_ca_single, state2_coords)

# Create comprehensive comparison visualization
fig = plt.figure(figsize=(16, 12))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# === Top Row: 3D Structure Visualizations ===
# With MSA structure
ax1 = fig.add_subplot(gs[0, 0], projection='3d')
ca_coords_msa = result_with_msa['structure'][:, 1, :]
plddt_msa = result_with_msa['plddt'] * 100
scatter1 = ax1.scatter(ca_coords_msa[:, 0], ca_coords_msa[:, 1], ca_coords_msa[:, 2],
                       c=plddt_msa, cmap='RdYlBu_r', s=20, vmin=50, vmax=90)
ax1.plot(ca_coords_msa[:, 0], ca_coords_msa[:, 1], ca_coords_msa[:, 2], 'gray', alpha=0.3, linewidth=0.5)
ax1.set_title('With MSA (Full MMseqs2)', fontsize=11, fontweight='bold')
ax1.set_xlabel('X (Å)', fontsize=9)
ax1.set_ylabel('Y (Å)', fontsize=9)
ax1.set_zlabel('Z (Å)', fontsize=9)
plt.colorbar(scatter1, ax=ax1, label='pLDDT (%)', shrink=0.6)

# Single sequence structure
ax2 = fig.add_subplot(gs[0, 1], projection='3d')
ca_coords_single = result_single['structure'][:, 1, :]
plddt_single = result_single['plddt'] * 100
scatter2 = ax2.scatter(ca_coords_single[:, 0], ca_coords_single[:, 1], ca_coords_single[:, 2],
                       c=plddt_single, cmap='RdYlBu_r', s=20, vmin=50, vmax=90)
ax2.plot(ca_coords_single[:, 0], ca_coords_single[:, 1], ca_coords_single[:, 2], 'gray', alpha=0.3, linewidth=0.5)
ax2.set_title('Without MSA (Single Sequence)', fontsize=11, fontweight='bold')
ax2.set_xlabel('X (Å)', fontsize=9)
ax2.set_ylabel('Y (Å)', fontsize=9)
ax2.set_zlabel('Z (Å)', fontsize=9)
plt.colorbar(scatter2, ax=ax2, label='pLDDT (%)', shrink=0.6)

# RMSD comparison bar chart
ax3 = fig.add_subplot(gs[0, 2])
conditions = ['With MSA\n(MMseqs2)', 'Without MSA\n(Single Seq)']
rmsd1_values = [rmsd_state1, rmsd_state1_single]
rmsd2_values = [rmsd_state2, rmsd_state2_single]
x = np.arange(len(conditions))
width = 0.35
bars1 = ax3.bar(x - width/2, rmsd1_values, width, label='RMSD to State 1', color='steelblue', alpha=0.8)
bars2 = ax3.bar(x + width/2, rmsd2_values, width, label='RMSD to State 2', color='coral', alpha=0.8)
ax3.set_ylabel('RMSD (Å)', fontsize=10)
ax3.set_title('Conformational Preference', fontsize=11, fontweight='bold')
ax3.set_xticks(x)
ax3.set_xticklabels(conditions, fontsize=9)
ax3.legend(fontsize=9)
ax3.grid(True, alpha=0.3, axis='y')
# Add value labels on bars
for bar, val in zip(bars1, rmsd1_values):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
             f'{val:.1f}', ha='center', va='bottom', fontsize=8)
for bar, val in zip(bars2, rmsd2_values):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
             f'{val:.1f}', ha='center', va='bottom', fontsize=8)

# === Middle Row: Confidence Metrics ===
# pLDDT per residue comparison
ax4 = fig.add_subplot(gs[1, :])
positions = np.arange(1, len(I89_SEQUENCE) + 1)
ax4.plot(positions, plddt_msa, 'steelblue', linewidth=2, label='With MSA', alpha=0.8)
ax4.plot(positions, plddt_single, 'coral', linewidth=2, label='Without MSA', alpha=0.8)
ax4.fill_between(positions, plddt_msa, alpha=0.3, color='steelblue')
ax4.fill_between(positions, plddt_single, alpha=0.3, color='coral')
ax4.axhline(y=70, color='gray', linestyle='--', alpha=0.5, label='pLDDT=70 (confident)')
ax4.axhline(y=50, color='gray', linestyle=':', alpha=0.5, label='pLDDT=50 (low confidence)')
ax4.set_xlabel('Residue Position', fontsize=10)
ax4.set_ylabel('pLDDT (%)', fontsize=10)
ax4.set_title('Per-Residue Confidence Comparison', fontsize=11, fontweight='bold')
ax4.legend(loc='lower right', fontsize=8, ncol=2)
ax4.grid(True, alpha=0.3)
ax4.set_xlim(1, len(I89_SEQUENCE))
ax4.set_ylim(0, 100)

# === Bottom Row: PAE and Summary Statistics ===
# PAE for with MSA
ax5 = fig.add_subplot(gs[2, 0])
im1 = ax5.imshow(result_with_msa['pae'], cmap='Greens_r', vmin=0, vmax=30)
ax5.set_title('PAE - With MSA', fontsize=10, fontweight='bold')
ax5.set_xlabel('Residue', fontsize=9)
ax5.set_ylabel('Residue', fontsize=9)
plt.colorbar(im1, ax=ax5, label='Error (Å)', shrink=0.8)

# PAE for single sequence
ax6 = fig.add_subplot(gs[2, 1])
im2 = ax6.imshow(result_single['pae'], cmap='Greens_r', vmin=0, vmax=30)
ax6.set_title('PAE - Without MSA', fontsize=10, fontweight='bold')
ax6.set_xlabel('Residue', fontsize=9)
ax6.set_ylabel('Residue', fontsize=9)
plt.colorbar(im2, ax=ax6, label='Error (Å)', shrink=0.8)

# Summary statistics table
ax7 = fig.add_subplot(gs[2, 2])
ax7.axis('tight')
ax7.axis('off')

# Create comparison table
table_data = [
    ['Metric', 'With MSA', 'Without MSA', 'Difference'],
    ['Mean pLDDT (%)', f'{np.mean(plddt_msa):.1f}', f'{np.mean(plddt_single):.1f}', 
     f'{np.mean(plddt_msa) - np.mean(plddt_single):+.1f}'],
    ['RMSD to State 1 (Å)', f'{rmsd_state1:.2f}', f'{rmsd_state1_single:.2f}',
     f'{rmsd_state1 - rmsd_state1_single:+.2f}'],
    ['RMSD to State 2 (Å)', f'{rmsd_state2:.2f}', f'{rmsd_state2_single:.2f}',
     f'{rmsd_state2 - rmsd_state2_single:+.2f}'],
    ['Closer to', 'State 1' if rmsd_state1 < rmsd_state2 else 'State 2',
     'State 1' if rmsd_state1_single < rmsd_state2_single else 'State 2', '-'],
    ['Mean PAE (Å)', f'{np.mean(result_with_msa["pae"]):.1f}', 
     f'{np.mean(result_single["pae"]):.1f}',
     f'{np.mean(result_with_msa["pae"]) - np.mean(result_single["pae"]):+.1f}']
]

table = ax7.table(cellText=table_data, loc='center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(8)
table.scale(1.2, 1.8)
# Style header row
for i in range(4):
    table[(0, i)].set_facecolor('#40466e')
    table[(0, i)].set_text_props(weight='bold', color='white')
# Color code the "Closer to" row
table[(4, 1)].set_facecolor('lightblue' if rmsd_state1 < rmsd_state2 else 'lightcoral')
table[(4, 2)].set_facecolor('lightblue' if rmsd_state1_single < rmsd_state2_single else 'lightcoral')

ax7.set_title('Summary Statistics', fontsize=11, fontweight='bold', pad=20)

plt.suptitle('MSA vs Single Sequence: Comprehensive Comparison', fontsize=14, fontweight='bold', y=0.98)
plt.tight_layout()
plt.show()

# Print key insights
print("\n" + "="*60)
print("KEY INSIGHTS FROM COMPARISON")
print("="*60)
print(f"\n1. CONFORMATIONAL PREFERENCE:")
if rmsd_state1 < rmsd_state2 and rmsd_state1_single > rmsd_state2_single:
    print("   ✓ MSA drives conformational switching!")
    print(f"   - With MSA: Prefers State 1 (RMSD {rmsd_state1:.2f} Å)")
    print(f"   - Without MSA: Prefers State 2 (RMSD {rmsd_state2_single:.2f} Å)")
elif rmsd_state1 < rmsd_state2 and rmsd_state1_single < rmsd_state2_single:
    print("   - Both prefer State 1, but with different confidence")
    print(f"   - MSA strengthens State 1 preference")
else:
    print(f"   - Complex conformational landscape detected")

print(f"\n2. CONFIDENCE METRICS:")
plddt_diff = np.mean(plddt_msa) - np.mean(plddt_single)
if plddt_diff > 5:
    print(f"   ✓ MSA significantly improves confidence (+{plddt_diff:.1f}%)")
elif plddt_diff > 0:
    print(f"   - MSA modestly improves confidence (+{plddt_diff:.1f}%)")
else:
    print(f"   - Similar confidence levels (difference: {plddt_diff:.1f}%)")

print(f"\n3. STRUCTURAL DIVERGENCE:")
struct_rmsd = af2.calculate_rmsd(ca_coords_msa, ca_coords_single)
print(f"   - RMSD between predictions: {struct_rmsd:.2f} Å")
if struct_rmsd > 5:
    print("   ✓ Significant structural differences between conditions")
    print("   → MSA presence fundamentally changes the prediction")
elif struct_rmsd > 2:
    print("   - Moderate structural differences")
    print("   → Local conformational changes induced by MSA")
else:
    print("   - Minor structural differences")
    print("   → Overall fold is conserved")

print("\n" + "="*60)
print("CONCLUSION")
print("="*60)
print("This comparison demonstrates that MSA is not just about improving")
print("confidence - it fundamentally influences which conformation AlphaFold2")
print("predicts by providing evolutionary constraints that guide the model")
print("toward biologically relevant states.")


# %%
#@title Interactive 3D Visualization with LogMD
#@markdown Creates side-by-side interactive viewers for MSA vs Single Sequence

if af2.check_logmd():
    print("Creating interactive 3D visualization...")
    
    # Create LogMD trajectory for the MSA prediction
    traj_msa = af2.create_logmd_from_prediction(
        result_with_msa,
        sequence=I89_SEQUENCE,
        project="",  # Public upload
        description="With MSA"
    )
    
    # Create LogMD trajectory for the single sequence prediction  
    traj_single = af2.create_logmd_from_prediction(
        result_single,
        sequence=I89_SEQUENCE,
        project="",  # Public upload
        description="Without MSA"
    )
    
    if traj_msa and traj_single:
        print("\nThis allows you to rotate, zoom, and explore the structure!")
        print(f"[logmd] Load_time={traj_msa.load_time:.2f}s 🚀")
        print(f"[logmd] Url={traj_msa.url} 🚀")
        print("\nInteractive 3D Structure Viewer")
        print("Open in new window")
        
        # Create side-by-side display
        from IPython.display import display, HTML
        
        # Enhanced URLs with pLDDT coloring
        url_msa = f"{traj_msa.url}?preset=polymer-cartoon&plddt"
        url_single = f"{traj_single.url}?preset=polymer-cartoon&plddt"
        
        html_content = f"""
        <div style="display: flex; gap: 20px; margin-top: 20px;">
            <div style="flex: 1; text-align: center;">
                <h4 style="color: steelblue;">With MSA (MMseqs2)</h4>
                <iframe src="{url_msa}" width="400" height="400" frameborder="0"></iframe>
                <p style="font-size: 12px; color: gray;">
                    RMSD to State 1: {rmsd_state1:.2f} Å<br>
                    RMSD to State 2: {rmsd_state2:.2f} Å<br>
                    Mean pLDDT: {np.mean(plddt_msa):.1f}%
                </p>
            </div>
            <div style="flex: 1; text-align: center;">
                <h4 style="color: coral;">Without MSA (Single Seq)</h4>
                <iframe src="{url_single}" width="400" height="400" frameborder="0"></iframe>
                <p style="font-size: 12px; color: gray;">
                    RMSD to State 1: {rmsd_state1_single:.2f} Å<br>
                    RMSD to State 2: {rmsd_state2_single:.2f} Å<br>
                    Mean pLDDT: {np.mean(plddt_single):.1f}%
                </p>
            </div>
        </div>
        <div style="text-align: center; margin-top: 20px;">
            <p><strong>Interactive viewer features:</strong></p>
            <ul style="list-style: none; font-size: 12px; color: gray;">
                <li>• Left click + drag: Rotate structure</li>
                <li>• Scroll: Zoom in/out</li>
                <li>• Colors show pLDDT confidence (blue=high, red=low)</li>
                <li>• Compare conformations side-by-side</li>
            </ul>
        </div>
        """
        
        display(HTML(html_content))
        
        print("\n" + "="*60)
        print("INTERACTIVE COMPARISON")
        print("="*60)
        print("Key observations from the 3D structures:")
        print("  1. Notice the overall fold differences")
        print("  2. Compare the loop regions (especially residues 85-95)")
        print("  3. Observe confidence differences (color intensity)")
        print("  4. MSA provides evolutionary constraints → State 1")
        print("  5. Single sequence relies on learned patterns → State 2")
    else:
        print("Failed to create LogMD visualizations")
else:
    print("LogMD not available - install with: pip install logmd")
    print("After installation, restart kernel and re-run this cell")


# %%
#@title Interactive MSA Quality Analysis
#@markdown Explore MSA coverage and sequence diversity with interactive plots

# Parse MSA file for visualization
from colabdesign.af.contrib import predict
sequences, deletion_matrix = predict.parse_a3m("i89_msa_analysis/msa.a3m")

# Create MSA visualizer
vis = af2.MSACoevolutionVisualizer()

# Load MSA data with metadata
msa_data = af2.MSAData(
    array=msa_full,
    deletion_matrix=del_matrix,
    sequences=sequences,
    neff=msa_full.shape[0],
    length=msa_full.shape[1],
    condition_name="i89 Full MSA (MMseqs2)"
)

print("="*60)
print("MSA QUALITY DIAGNOSTICS")
print("="*60)
vis.print_diagnostics(msa_data)

# Create interactive MSA quality plots
print("\n" + "="*60)
print("INTERACTIVE MSA VISUALIZATIONS")
print("="*60)
print("Creating interactive plots...")
print("Hover over the plots to see detailed information!\n")

fig_msa = af2.plot_msa_interactive(msa_data, title="i89 MSA Quality Metrics")
fig_msa.show()

print("\n" + "="*60)
print("UNDERSTANDING THE PLOTS")
print("="*60)
print("Coverage plot (top):")
print("  - Shows percentage of sequences with residues at each position")
print("  - Gaps or low coverage indicate flexible/disordered regions")
print("  - High coverage (>90%) indicates well-conserved positions")
print("\nSequence Identity Distribution (bottom):")
print("  - Shows how similar homologous sequences are to i89")
print("  - Peak around 30-50% identity is typical for good MSAs")
print("  - Too high (>90%): Limited diversity, weak coevolution signal")
print("  - Too low (<20%): May include false positives")
print("\nKey insight:")
print(f"  With {msa_data.neff} sequences, we have strong evolutionary signal!")
print("  This will guide AlphaFold2 toward the biologically relevant conformation")


# %%
#@title Coevolution Analysis - Identifying Coupled Residues
#@markdown Coevolution reveals which residues evolved together to maintain protein function

print("="*60)
print("COEVOLUTION ANALYSIS")
print("="*60)
print("Computing coevolution matrix using Direct Coupling Analysis (DCA)...")
print("This identifies residues that co-evolve to maintain function\n")

# Compute coevolution with caching
coev_matrix = vis.compute_coevolution(msa_data)

# Create interactive coevolution heatmap
print("Creating interactive coevolution heatmap...")
print("Hover over any position to see residue identities and coevolution scores!\n")

fig_coev = vis.plot_heatmap(
    coev_matrix,
    title="i89 Coevolution Matrix - Interactive",
    msa_data=msa_data
)

fig_coev.show()

# Analyze coevolution statistics
upper_tri = np.triu_indices_from(coev_matrix, k=6)
coev_values = coev_matrix[upper_tri]

print("\n" + "="*60)
print("COEVOLUTION STATISTICS")
print("="*60)
print(f"  Max coevolution score: {np.max(coev_values):.3f}")
print(f"  Mean coevolution score: {np.mean(coev_values):.3f}")
print(f"  Standard deviation: {np.std(coev_values):.3f}")
print(f"  Strong pairs (score > 0.5): {np.sum(coev_values > 0.5)}")

# Find top coevolving pairs
top_n = 10
top_indices = np.argsort(coev_values)[-top_n:]

print(f"\nTop {top_n} coevolving residue pairs:")
for idx in reversed(top_indices):
    i, j = upper_tri[0][idx], upper_tri[1][idx]
    score = coev_values[idx]
    print(f"  Positions {i+1:3d}-{j+1:3d}: {score:.4f}")

print("\n" + "="*60)
print("BIOLOGICAL INTERPRETATION")
print("="*60)
print("Strong coevolution indicates:")
print("  1. Structural contacts - residues that are close in 3D space")
print("  2. Functional coupling - allosteric networks and communication")
print("  3. Compensatory mutations - maintaining stability under evolutionary pressure")
print("\nThis evolutionary information:")
print("  - Guides AlphaFold2 toward functional conformations")
print("  - Helps predict long-range interactions")
print("  - Distinguishes between alternative states")


# %%
#@title Focus: Calcium-Binding Loop Region
#@markdown The loop at residues 85-95 is critical for calcium binding and undergoes conformational changes

print("="*60)
print("CALCIUM-BINDING LOOP ANALYSIS")
print("="*60)
print("Analyzing residues 85-95 (the calcium-binding loop)...")
print("This region is responsible for the conformational differences between states!")
print(f"Coevolution matrix shape: {coev_matrix.shape}")
print(f"Expected sequence length: {len(I89_SEQUENCE)}\n")

# Extract coevolution for calcium-binding region
# Ensure we don't exceed matrix bounds
matrix_size = coev_matrix.shape[0]
ca_start = 84  # 0-indexed (residue 85)
ca_end = min(96, matrix_size)  # Don't exceed matrix size

if ca_end > ca_start:
    ca_coev_submatrix = coev_matrix[ca_start:ca_end, ca_start:ca_end]
    
    # Statistics for this region
    upper_tri_ca = np.triu_indices_from(ca_coev_submatrix, k=1)
    ca_coev_values = ca_coev_submatrix[upper_tri_ca]
    max_ca_coev = np.max(ca_coev_values) if len(ca_coev_values) > 0 else 0
    mean_ca_coev = np.mean(ca_coev_values) if len(ca_coev_values) > 0 else 0
    
    print(f"Calcium-binding loop coevolution (positions {ca_start+1}-{ca_end}):")
    print(f"  Max score in region: {max_ca_coev:.3f}")
    print(f"  Mean score in region: {mean_ca_coev:.3f}")
    print(f"  Overall mean (all positions): {np.mean(coev_values):.3f}")
    
    if mean_ca_coev > np.mean(coev_values) * 1.1:
        print(f"\nStrong evolutionary signal detected in Ca-binding loop!")
        print(f"  {mean_ca_coev/np.mean(coev_values):.1f}x higher than average")
        print(f"  This will bias AlphaFold2 toward State 1 (Ca-bound conformation)")
    else:
        print(f"\nModerate evolutionary signal in Ca-binding loop")
    
    # Find coevolving pairs involving the calcium-binding loop
    print(f"\nResidue pairs coevolving with Ca-binding loop (score > 0.4):")
    found_pairs = 0
    for i in range(ca_start, min(ca_end, matrix_size)):
        for j in range(min(ca_end, matrix_size), matrix_size):
            if coev_matrix[i, j] > 0.4:
                print(f"  Loop residue {i+1} <-> Residue {j+1}: {coev_matrix[i,j]:.3f}")
                found_pairs += 1
                if found_pairs >= 5:  # Limit output
                    break
        if found_pairs >= 5:
            break
    
    if found_pairs > 0:
        print(f"\nFound {found_pairs}+ strong coevolving pairs involving the loop")
        print("These connections maintain the calcium-binding geometry!")
    else:
        print("\nFew strong connections found (may indicate flexible region)")
else:
    print("WARNING: Matrix too small to analyze calcium-binding loop region")

print("\n" + "="*60)
print("KEY PREDICTION")
print("="*60)
print("Based on this evolutionary analysis:")
print("  - Strong coevolution in calcium-binding loop")
print("  - Well-connected to rest of structure")
print("  - Prediction: AlphaFold2 will favor State 1 (Ca-bound)")
print("\nNext, we'll see if this prediction holds when we run AlphaFold2!")


# %%
#@title Preview: What Happens Without MSA?
#@markdown Let's create a single-sequence MSA to see the difference in coevolution signal

print("="*60)
print("COMPARISON: WITH vs WITHOUT MSA")
print("="*60)
print("Creating single-sequence MSA for comparison...")
print("This simulates what happens when no homologous sequences are found\n")

## Create single-sequence MSA
#msa_single, del_matrix_single = af2.create_single_sequence_msa(I89_SEQUENCE)
#
## Parse for visualization
#import tempfile
#with tempfile.NamedTemporaryFile(mode='w', suffix='.a3m', delete=False) as tmp:
#    tmp.write(f">i89\n{I89_SEQUENCE}\n")
#    tmp_path = tmp.name
#
#sequences_single, deletion_matrix_single = predict.parse_a3m(tmp_path)
#os.unlink(tmp_path)
#
## Create MSAData for single sequence
#msa_data_single = af2.MSAData(
#    array=msa_single,
#    deletion_matrix=del_matrix_single,
#    sequences=sequences_single,
#    neff=1,
#    length=len(I89_SEQUENCE),
#    condition_name="i89 Single Sequence (No MSA)"
#)
#
#print("Computing coevolution for single-sequence MSA...")
#coev_single = vis.compute_coevolution(msa_data_single)
#
## Compare coevolution signals
#print("\n" + "="*60)
#print("COEVOLUTION SIGNAL COMPARISON")
#print("="*60)
#print(f"WITH MSA ({msa_data.neff} sequences):")
#print(f"  Mean coevolution: {np.mean(coev_matrix):.4f}")
#print(f"  Max coevolution: {np.max(coev_matrix):.4f}")
#print(f"  Ca-loop mean: {mean_ca_coev:.4f}")
#
#print(f"\nWITHOUT MSA (1 sequence):")
#print(f"  Mean coevolution: {np.mean(coev_single):.4f}")
#print(f"  Max coevolution: {np.max(coev_single):.4f}")
#print(f"  Ca-loop mean: {np.mean(coev_single[84:96, 84:96]):.4f}")
#
## Create side-by-side comparison
#print("\nCreating visual comparison...")
#
#conditions = {
#    "With MSA (MMseqs2)": msa_data,
#    "Without MSA (Single Sequence)": msa_data_single
#}
#
#fig_main, fig_diff = af2.compare_coevolution_conditions(
#    conditions,
#    show_difference=True,
#    reference_condition="Without MSA (Single Sequence)"
#)
#
#print("\nShowing side-by-side comparison...")
#fig_main.show()
#
#if fig_diff is not None:
#    print("\nShowing difference plot (With MSA - Without MSA)...")
#    print("Positive values (red) = stronger coevolution WITH MSA")
#    fig_diff.show()
#
#print("\n" + "="*60)
#print("KEY INSIGHTS FOR STRUCTURE PREDICTION")
#print("="*60)
#print("1. WITH MSA:")
#print("   - Clear coevolution patterns provide evolutionary guidance")
#print("   - Strong signal in functional regions (like Ca-binding loop)")
#print("   - AlphaFold2 will be biased toward evolutionarily stable State 1")
#
#print("\n2. WITHOUT MSA:")
#print("   - No coevolution signal (single sequence)")
#print("   - AlphaFold2 relies only on learned structural patterns")
#print("   - May sample alternative conformations (like State 2)")
#
#print("\n3. PREDICTION:")
#print("   - Next section will test this hypothesis")
#print("   - We expect: Full MSA → State 1, No MSA → State 2")
#print("   - The coevolution data explains WHY different conformations emerge!")
#
## Save both coevolution matrices for later comparison
#np.save("i89_coev_with_msa.npy", coev_matrix)
#np.save("i89_coev_single_seq.npy", coev_single)
#print("\nCoevolution matrices saved for later analysis")


# %% [markdown]
# ## Section 3: Basic Prediction with Full MSA
# 
# Let's start by predicting the i89 structure with a full MSA. This typically results in a conformation closer to State 1.
# 

# %%
%%time
#@title Prediction with Full MSA (With PDB Saving)
#@markdown Using enhanced prediction to save all recycle PDBs

print("="*60)
print("PREDICTION WITH FULL MSA - SAVING ALL PDBs")
print("="*60)

# Create job folder with sequence hash
job_folder_msa = af2.create_job_folder(I89_SEQUENCE, "i89_with_msa")
print(f"Job folder: {job_folder_msa}")
print(f"Sequence hash: {af2.get_hash(I89_SEQUENCE)}\n")

# Setup model
model = af2.setup_model(I89_SEQUENCE, verbose=False)

# Load or generate MSA
import os
if os.path.exists("i89_msa.npy") and os.path.exists("i89_del_matrix.npy"):
    print("Loading pre-generated MSA from Section 2.5...")
    msa = np.load("i89_msa.npy")
    deletion_matrix = np.load("i89_del_matrix.npy")
    print(f"  Loaded MSA with {len(msa)} sequences")
else:
    print("Generating MSA using MMseqs2...")
    print("⚠️ This may take 2-5 minutes depending on server load...")
    print("If it takes longer than 5 minutes, interrupt and use single_sequence mode instead.\n")
    msa, deletion_matrix = af2.get_msa([I89_SEQUENCE], "temp_msa", verbose=True)
    # Save for future use
    np.save("i89_msa.npy", msa)
    np.save("i89_del_matrix.npy", deletion_matrix)
    print(f"  Generated MSA with {len(msa)} sequences")

print(f"  This MSA contains coevolution information")
print(f"  Prediction: Based on coevolution, expecting State 1 (Ca-bound)\n")

# Run prediction with recycling and PDB saving
print("Starting prediction with recycling...")
print("This will run 4 iterations (recycle 0-3) and save PDB for each\n")

try:
    result_with_msa = af2.predict_with_recycling(
        model,
        msa=msa,
        deletion_matrix=deletion_matrix,
        max_recycles=3,
        seed=0,
        save_pdbs=True,  # Save PDB for each recycle
        job_folder=job_folder_msa,
        sequence=I89_SEQUENCE,
        model_name="with_msa",
        verbose=True
    )
    print("\nPrediction completed successfully!")
except Exception as e:
    print(f"\nError during prediction: {e}")
    print("Falling back to quick prediction without PDB saving...")
    result_with_msa = af2.quick_predict(
        sequence=I89_SEQUENCE,
        msa_mode="mmseqs2",
        num_recycles=3,
        verbose=True
    )

# Calculate RMSD to reference states
pred_ca = result_with_msa['structure'][:, 1, :]  # CA atoms
rmsd_state1 = af2.calculate_rmsd(pred_ca, state1_coords)
rmsd_state2 = af2.calculate_rmsd(pred_ca, state2_coords)

print("\n" + "="*60)
print("RESULTS")
print("="*60)
print(f"RMSD to State 1: {rmsd_state1:.2f} Angstrom")
print(f"RMSD to State 2: {rmsd_state2:.2f} Angstrom")
print(f"Mean pLDDT: {result_with_msa['metrics']['plddt']*100:.1f}%")

if rmsd_state1 < rmsd_state2:
    print(f"\nPrediction is closer to State 1 (as expected with MSA)")
    print(f"Delta: {rmsd_state2 - rmsd_state1:.2f} Angstrom difference")
else:
    print(f"\nPrediction is closer to State 2")
    print(f"Delta: {rmsd_state1 - rmsd_state2:.2f} Angstrom difference")


# %%
#@title Alternative: Fast Prediction with Single Sequence (No MSA)
#@markdown Use this if MMseqs2 is taking too long or not working

print("="*60)
print("FAST PREDICTION (SINGLE SEQUENCE) - WITH PDB SAVING")
print("="*60)
print("Using single sequence mode for faster prediction...\n")

# Create job folder
job_folder_fast = af2.create_job_folder(I89_SEQUENCE, "i89_fast")
print(f"Job folder: {job_folder_fast}")
print(f"Sequence hash: {af2.get_hash(I89_SEQUENCE)}\n")

# Setup model
model_fast = af2.setup_model(I89_SEQUENCE, verbose=False)

# Create single sequence MSA (instant, no database search)
msa_single, del_single = af2.create_single_sequence_msa(I89_SEQUENCE)
print(f"Created single sequence MSA: shape {msa_single.shape}")

# Run prediction
print("\nStarting prediction...")
result_fast = af2.predict_with_recycling(
    model_fast,
    msa=msa_single,
    deletion_matrix=del_single,
    max_recycles=3,
    seed=0,
    save_pdbs=True,
    job_folder=job_folder_fast,
    sequence=I89_SEQUENCE,
    model_name="fast",
    verbose=True
)

print("\n" + "="*60)
print("RESULTS")
print("="*60)
print(f"Mean pLDDT: {result_fast['metrics']['plddt']*100:.1f}%")
print(f"PDBs saved in: {job_folder_fast}/pdb/recycles/")
print("\nNote: Single sequence predictions may be less accurate")
print("but are useful for testing the pipeline quickly.")


# %%
#@title View Saved PDB Files from Full MSA Prediction
#@markdown List and optionally create trajectory from saved PDBs

print("="*60)
print("SAVED PDB FILES FROM FULL MSA PREDICTION")
print("="*60)

if 'job_folder_msa' in locals():
    print(f"Job folder: {job_folder_msa}/")
    
    # List PDB files
    import glob
    recycle_pdbs = sorted(glob.glob(f"{job_folder_msa}/pdb/recycles/*.pdb"))
    
    if recycle_pdbs:
        print(f"\nFound {len(recycle_pdbs)} PDB files:")
        for pdb in recycle_pdbs:
            print(f"  - {os.path.basename(pdb)}")
        
        # Optionally create LogMD trajectory
        if af2.check_logmd():
            print("\n" + "="*60)
            print("CREATING LOGMD TRAJECTORY FROM SAVED PDBs")
            print("="*60)
            
            traj_msa = af2.create_recycle_trajectory(
                job_folder=job_folder_msa,
                sequence=I89_SEQUENCE,
                model_name="with_msa",
                seed=0,
                align_to_first=True,
                project="i89_msa_trajectory",
                verbose=True
            )
            
            if traj_msa:
                print(f"\nView trajectory: {traj_msa.url}")
    else:
        print("No PDB files found. Run the prediction cell first.")
else:
    print("No prediction results found. Run the previous cell first!")


# %%
#@title Visualize Structure and Confidence

# Plot 3D structure with pLDDT coloring
fig = af2.plot_3d_structure(
    atom_positions=result_with_msa['structure'],
    plddt=result_with_msa['plddt'],
    save_path="i89_with_msa_structure.png",
    show=True
)

# Plot confidence metrics
fig = af2.plot_confidence(
    plddt=result_with_msa['plddt'] * 100,
    pae=result_with_msa['pae'],
    save_path="i89_with_msa_confidence.png",
    show=True
)


# %%
#@title Interactive 3D Visualization (Optional - Requires LogMD)

if af2.check_logmd():
    print("Creating interactive 3D visualization...")
    print("This allows you to rotate, zoom, and explore the structure!\n")
    
    # Create simple trajectory with just the final structure
    # Note: project="" uses public upload (more reliable than named projects)
    traj = af2.create_trajectory_from_ensemble(
        predictions=[result_with_msa],
        sequence=I89_SEQUENCE,
        project="",  # Empty = public upload (recommended!)
        align_structures=False,
        verbose=False
    )
    
    if traj:
        import logmd_utils
        logmd_utils.display_trajectory_in_notebook(traj)
        print("\nInteractive viewer features:")
        print("  - Left click + drag: Rotate structure")
        print("  - Right click + drag: Zoom")
        print("  - Colors show pLDDT confidence (blue=high, red=low)")
        print(f"\nShare this URL: {traj.url}")
    else:
        print("Failed to create visualization")
else:
    print("LogMD not available - skipping interactive 3D visualization")
    print("The static plots above show the structure and confidence metrics")
    print("\nTo enable interactive 3D:")
    print("  1. Run: !pip install logmd")
    print("  2. Restart kernel")
    print("  3. Re-run from the beginning")


# %% [markdown]
# ## Section 4: MSA Manipulation - Exploring Conformational Control
# 
# Now let's see how removing MSA information affects the predicted conformation. Without MSA, AlphaFold2 relies more on learned structural patterns.
# 

# %%
%%time
#@title Prediction without MSA (Single Sequence) - With PDB Saving

print("="*60)
print("PREDICTION WITHOUT MSA - SAVING ALL PDBs")
print("="*60)

# Create job folder with sequence hash
job_folder_no_msa = af2.create_job_folder(I89_SEQUENCE, "i89_no_msa")
print(f"Job folder: {job_folder_no_msa}")
print(f"Sequence hash: {af2.get_hash(I89_SEQUENCE)}\n")

# Setup model and single sequence MSA
model_no_msa = af2.setup_model(I89_SEQUENCE, verbose=False)
msa_single, deletion_matrix_single = af2.create_single_sequence_msa(I89_SEQUENCE)

# Run prediction with recycling and PDB saving
result_no_msa = af2.predict_with_recycling(
    model_no_msa,
    msa=msa_single,
    deletion_matrix=deletion_matrix_single,
    max_recycles=3,
    seed=0,
    save_pdbs=True,  # Save PDB for each recycle
    job_folder=job_folder_no_msa,
    sequence=I89_SEQUENCE,
    model_name="no_msa",
    verbose=True
)

# Calculate RMSD to reference states
pred_ca_no_msa = result_no_msa['structure'][:, 1, :]
rmsd_state1_no_msa = af2.calculate_rmsd(pred_ca_no_msa, state1_coords)
rmsd_state2_no_msa = af2.calculate_rmsd(pred_ca_no_msa, state2_coords)

print("\n" + "="*60)
print("RESULTS WITHOUT MSA")
print("="*60)
print(f"RMSD to State 1: {rmsd_state1_no_msa:.2f} Angstrom")
print(f"RMSD to State 2: {rmsd_state2_no_msa:.2f} Angstrom")
print(f"Mean pLDDT: {result_no_msa['metrics']['plddt']*100:.1f}%")

if rmsd_state2_no_msa < rmsd_state1_no_msa:
    print(f"\nPrediction is closer to State 2 (as expected without MSA)")
    print(f"Delta: {rmsd_state1_no_msa - rmsd_state2_no_msa:.2f} Angstrom difference")
else:
    print(f"\nPrediction is closer to State 1")
    print(f"Delta: {rmsd_state2_no_msa - rmsd_state1_no_msa:.2f} Angstrom difference")


# %%
#@title Compare Both Predictions

# Prepare comparison data
comparison_results = [
    {'rmsd_state1': rmsd_state1, 'rmsd_state2': rmsd_state2},
    {'rmsd_state1': rmsd_state1_no_msa, 'rmsd_state2': rmsd_state2_no_msa}
]

# Create comparison plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# RMSD to State 1
labels = ['With MSA', 'Without MSA']
rmsd1_values = [r['rmsd_state1'] for r in comparison_results]
ax1.bar(labels, rmsd1_values, color='steelblue')
ax1.set_ylabel('RMSD (Å)')
ax1.set_title('RMSD to State 1')
ax1.set_ylim(0, max(rmsd1_values + [r['rmsd_state2'] for r in comparison_results]) * 1.2)

# RMSD to State 2
rmsd2_values = [r['rmsd_state2'] for r in comparison_results]
ax2.bar(labels, rmsd2_values, color='coral')
ax2.set_ylabel('RMSD (Å)')
ax2.set_title('RMSD to State 2')
ax2.set_ylim(0, max(rmsd1_values + rmsd2_values) * 1.2)

plt.suptitle('MSA Effect on Conformational Preference', fontsize=14)
plt.tight_layout()
plt.show()

print("\nKey Finding:")
print("MSA presence/absence can switch the predicted conformation!")
print(f"Conformational shift: {abs(rmsd_state1 - rmsd_state1_no_msa):.1f} Angstrom")


# %%
#@title Interactive 3D Comparison: With vs Without MSA
#@markdown Compare both conformations side-by-side in 3D

if af2.check_logmd():
    print("Creating side-by-side 3D visualizations...")
    print("You'll get two URLs to open in separate browser tabs!\n")
    
    # Trajectory 1: With MSA (should be closer to State 1)
    traj_with = af2.create_trajectory_from_ensemble(
        predictions=[result_with_msa],
        sequence=I89_SEQUENCE,
        project="i89_with_msa_final",
        sort_by_rmsd=True,
        reference_coords=state1_coords,
        verbose=False
    )
    
    # Trajectory 2: Without MSA (should be closer to State 2)
    traj_without = af2.create_trajectory_from_ensemble(
        predictions=[result_no_msa],
        sequence=I89_SEQUENCE,
        project="i89_without_msa_final",
        sort_by_rmsd=True,
        reference_coords=state2_coords,
        verbose=False
    )
    
    print("="*60)
    print("SIDE-BY-SIDE COMPARISON URLS")
    print("="*60)
    if traj_with:
        print(f"\nWith MSA (State 1-like):")
        print(f"  {traj_with.url}")
    
    if traj_without:
        print(f"\nWithout MSA (State 2-like):")
        print(f"  {traj_without.url}")
    
    print("\n" + "="*60)
    print("HOW TO COMPARE")
    print("="*60)
    print("1. Open both URLs in separate browser tabs")
    print("2. Arrange windows side-by-side")
    print("3. Rotate both structures to same orientation")
    print("4. Notice the conformational differences!")
    print("\nKey differences to look for:")
    print("  - Overall fold compactness")
    print("  - Loop positions and orientations")
    print("  - Domain arrangements")
else:
    print("LogMD not available - use static plots above for comparison")
    print("\nThe bar charts show RMSD differences quantitatively")
    print("Install LogMD for interactive 3D comparison!")


# %% [markdown]
# ## Section 5: Recycling for Conformational Refinement
# 
# Recycling is AlphaFold2's iterative refinement process. Let's explore how the number of recycles affects structure quality and conformational preference.
# 

# %% [markdown]
# ## Section 5.5: Real-time Structure Visualization with LogMD
# 
# LogMD provides interactive 3D visualization of structures as they evolve during prediction. This makes it easy to see how recycling refines the structure and how different conditions affect the final conformation.
# 

# %%
#@title Check LogMD Availability

# Check if LogMD is available
if af2.check_logmd():
    print("LogMD is available!")
    print("  - Interactive 3D visualization enabled")
    print("  - Real-time trajectory creation supported")
else:
    print("LogMD not available. Installing...")
    print("  Run: !pip install logmd")
    print("\nLogMD provides:")
    print("  - Interactive 3D structure viewer")
    print("  - Real-time visualization during prediction")
    print("  - Trajectory creation from ensembles")
    print("\nAfter installation, restart the kernel to use LogMD features.")


# %%
%%time
#@title Visualize Recycling Evolution with LogMD (Now with PDB Saving!)
#@markdown Watch how the structure refines through iterative recycling and save all PDB files

if af2.check_logmd():
    print("="*60)
    print("RECYCLING EVOLUTION WITH REAL-TIME VISUALIZATION & PDB SAVING")
    print("="*60)
    print("Watch the structure refine through recycling iterations...")
    print("All PDB files will be saved for offline analysis!\n")
    
    # Run prediction with LogMD to capture every recycle AND save PDBs
    result_logmd = af2.predict_with_logmd(
        sequence=I89_SEQUENCE,
        msa_mode="single_sequence",  # Use single sequence for faster demo
        num_recycles=6,
        project="i89_recycling_evolution",
        save_pdbs=True,  # NEW: Save PDB files for each recycle
        jobname="i89_logmd",  # NEW: Job name for folder
        show_viewer=True,
        verbose=True
    )
    
    print("\n" + "="*60)
    print("RECYCLING INSIGHTS")
    print("="*60)
    print(f"Final pLDDT: {result_logmd['metrics']['plddt']*100:.1f}%")
    print(f"Total recycles: {len(result_logmd['all_structures'])}")
    print("\nWhat you're seeing:")
    print("  - Recycles 0-2: Large conformational changes")
    print("  - Recycles 3-4: Fine-tuning and refinement")
    print("  - Recycles 5-6: Convergence (minimal changes)")
    print("\nInteractive viewer controls:")
    print("  - Mouse drag: Rotate structure")
    print("  - Scroll: Zoom in/out")
    print("  - Play button: Animate through recycles")
    print("  - Slider: Jump to specific recycle")
    print("\nNotice how:")
    print("  - Early recycles show major structural rearrangements")
    print("  - Later recycles show convergence")
    print("  - pLDDT confidence improves (colors get bluer)")
else:
    print("LogMD not available - falling back to standard recycling analysis")
    print("\nFor real-time visualization:")
    print("  1. Install LogMD: !pip install logmd")
    print("  2. Restart kernel")
    print("  3. Re-run from Section 1")
    print("\nContinuing with quantitative analysis...")


# %%
#@title Browse and Load Saved PDB Files
#@markdown Explore the saved PDB files and create a LogMD trajectory from them

import os
import glob

if 'result_logmd' in locals() and 'job_folder' in result_logmd:
    job_folder = result_logmd['job_folder']
    
    print("="*60)
    print("SAVED PDB FILES STRUCTURE")
    print("="*60)
    print(f"Job folder: {job_folder}/")
    print(f"  (Hash: {af2.get_hash(I89_SEQUENCE)})")
    print()
    
    # List all PDB files
    recycle_pdbs = sorted(glob.glob(f"{job_folder}/pdb/recycles/*.pdb"))
    
    if recycle_pdbs:
        print(f"Found {len(recycle_pdbs)} recycle PDB files:")
        for pdb in recycle_pdbs:
            # Get file size
            size = os.path.getsize(pdb) / 1024  # KB
            print(f"  - {os.path.basename(pdb)}: {size:.1f} KB")
        
        print("\n" + "="*60)
        print("CREATING TRAJECTORY FROM SAVED PDBs")
        print("="*60)
        
        # Create LogMD trajectory from saved PDBs
        recycle_traj = af2.create_recycle_trajectory(
            job_folder=job_folder,
            sequence=I89_SEQUENCE,
            model_name="model",
            seed=0,
            align_to_first=True,  # Align all to first recycle (AlphaMask style)
            project="i89_recycle_from_pdbs",
            verbose=True
        )
        
        if recycle_traj:
            print("\nTrajectory features:")
            print("  - All structures aligned to first recycle")
            print("  - pLDDT values extracted from B-factor column")
            print("  - Ready for structural analysis")
            print(f"\nView trajectory: {recycle_traj.url}")
    else:
        print("No recycle PDB files found.")
        
    print("\n" + "="*60)
    print("USAGE EXAMPLES")
    print("="*60)
    print("You can now:")
    print("  1. Download PDBs for PyMOL/ChimeraX visualization")
    print("  2. Analyze structural changes between recycles")
    print("  3. Calculate RMSD between iterations")
    print("  4. Share results with collaborators")
    print("\nExample PyMOL commands:")
    print(f"  load {job_folder}/pdb/recycles/model_r0_seed0.pdb")
    print(f"  load {job_folder}/pdb/recycles/model_r6_seed0.pdb")
    print("  align model_r6_seed0, model_r0_seed0")
    print("  spectrum b, blue_white_red, minimum=50, maximum=90")
else:
    print("No LogMD results found. Run the previous cell first!")


# %%
%%time
#@title Test Recycling with Early Stopping (With PDB Saving)

print("="*60)
print("TESTING RECYCLING WITH EARLY STOPPING & PDB SAVING")
print("="*60)

# Create job folder for this test
job_folder_recycling = af2.create_job_folder(I89_SEQUENCE, "i89_early_stop")
print(f"Job folder: {job_folder_recycling}\n")

# Setup model
model = af2.setup_model(I89_SEQUENCE, verbose=False)

# Generate MSA
msa, deletion_matrix = af2.create_single_sequence_msa(I89_SEQUENCE)

# Run prediction with recycling, early stopping, and PDB saving
result_recycling = af2.predict_with_recycling(
    model,
    msa=msa,
    deletion_matrix=deletion_matrix,
    max_recycles=6,
    early_stop_tolerance=0.5,  # Stop if RMSD change < 0.5 Angstrom
    seed=0,
    save_pdbs=True,  # NEW: Save PDB for each recycle
    job_folder=job_folder_recycling,  # NEW: Specify folder
    sequence=I89_SEQUENCE,  # NEW: Required for PDB saving
    model_name="early_stop",  # NEW: Custom model name
    verbose=True
)

# Plot convergence
recycle_trajectory = result_recycling['trajectory']
recycles = [r['recycle'] for r in recycle_trajectory]
plddt_values = [r['metrics']['plddt'] * 100 for r in recycle_trajectory]
rmsd_changes = [r['rmsd_change'] if r['rmsd_change'] is not None else 0 for r in recycle_trajectory]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# pLDDT convergence
ax1.plot(recycles, plddt_values, 'o-', color='green', linewidth=2)
ax1.set_xlabel('Recycle')
ax1.set_ylabel('Mean pLDDT (%)')
ax1.set_title('pLDDT Convergence')
ax1.grid(True, alpha=0.3)

# RMSD changes
ax2.plot(recycles[1:], rmsd_changes[1:], 's-', color='purple', linewidth=2)
ax2.axhline(y=0.5, color='red', linestyle='--', label='Early stop threshold')
ax2.set_xlabel('Recycle')
ax2.set_ylabel('RMSD Change (Å)')
ax2.set_title('Structure Convergence')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print(f"\nConverged at recycle {len(recycle_trajectory)-1}")
print(f"Final pLDDT: {result_recycling['metrics']['plddt']*100:.1f}%")


# %%
#@title Enhanced Recycle Analysis with GPU-Accelerated RMSD
#@markdown Analyze conformational trajectory during recycling using batch RMSD calculations

if 'recycle_trajectory' in locals() and len(recycle_trajectory) > 1:
    print("="*60)
    print("ENHANCED RECYCLE TRAJECTORY ANALYSIS (GPU-ACCELERATED)")
    print("="*60)
    
    # Extract all structures from recycle trajectory
    recycle_structures = [r['structure'] for r in recycle_trajectory]
    
    print(f"Analyzing {len(recycle_structures)} recycle iterations...")
    
    # Calculate RMSD to both reference states for all recycles at once
    print("Computing batch RMSD to reference states using GPU...")
    recycle_rmsds = af2.calculate_batch_rmsd_to_references(
        recycle_structures,
        ref1_path="state1.pdb",
        ref2_path="state2.pdb",
        use_gpu=True  # GPU acceleration
    )
    
    # Extract data for plotting
    recycles = [r['recycle'] for r in recycle_trajectory]
    rmsd1_vals = [r['rmsd_state1'] for r in recycle_rmsds]
    rmsd2_vals = [r['rmsd_state2'] for r in recycle_rmsds]
    plddt_vals = [r['metrics']['plddt'] * 100 for r in recycle_trajectory]
    
    # Create comprehensive visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: RMSD to both states
    ax = axes[0, 0]
    ax.plot(recycles, rmsd1_vals, 'o-', label='RMSD to State 1', color='steelblue', linewidth=2)
    ax.plot(recycles, rmsd2_vals, 's-', label='RMSD to State 2', color='coral', linewidth=2)
    ax.set_xlabel('Recycle', fontsize=11)
    ax.set_ylabel('RMSD (Å)', fontsize=11)
    ax.set_title('Conformational Trajectory During Recycling', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: RMSD difference (conformational preference)
    ax = axes[0, 1]
    rmsd_diff = [r2 - r1 for r1, r2 in zip(rmsd1_vals, rmsd2_vals)]
    ax.plot(recycles, rmsd_diff, 'd-', color='purple', linewidth=2)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Recycle', fontsize=11)
    ax.set_ylabel('RMSD₂ - RMSD₁ (Å)', fontsize=11)
    ax.set_title('Conformational Preference', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.fill_between(recycles, 0, rmsd_diff, where=[d > 0 for d in rmsd_diff], 
                     alpha=0.3, color='steelblue', label='Closer to State 1')
    ax.fill_between(recycles, 0, rmsd_diff, where=[d < 0 for d in rmsd_diff], 
                     alpha=0.3, color='coral', label='Closer to State 2')
    ax.legend(fontsize=9)
    
    # Plot 3: pLDDT evolution
    ax = axes[1, 0]
    ax.plot(recycles, plddt_vals, 'o-', color='green', linewidth=2)
    ax.set_xlabel('Recycle', fontsize=11)
    ax.set_ylabel('Mean pLDDT (%)', fontsize=11)
    ax.set_title('Confidence Evolution', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Combined metrics (normalized)
    ax = axes[1, 1]
    # Normalize values for comparison
    rmsd1_norm = [(r - min(rmsd1_vals)) / (max(rmsd1_vals) - min(rmsd1_vals)) if max(rmsd1_vals) > min(rmsd1_vals) else 0.5 for r in rmsd1_vals]
    rmsd2_norm = [(r - min(rmsd2_vals)) / (max(rmsd2_vals) - min(rmsd2_vals)) if max(rmsd2_vals) > min(rmsd2_vals) else 0.5 for r in rmsd2_vals]
    plddt_norm = [(p - min(plddt_vals)) / (max(plddt_vals) - min(plddt_vals)) if max(plddt_vals) > min(plddt_vals) else 0.5 for p in plddt_vals]
    
    ax.plot(recycles, rmsd1_norm, 'o-', label='RMSD₁ (norm)', alpha=0.7)
    ax.plot(recycles, rmsd2_norm, 's-', label='RMSD₂ (norm)', alpha=0.7)
    ax.plot(recycles, plddt_norm, '^-', label='pLDDT (norm)', alpha=0.7)
    ax.set_xlabel('Recycle', fontsize=11)
    ax.set_ylabel('Normalized Value', fontsize=11)
    ax.set_title('All Metrics (Normalized)', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.suptitle('Comprehensive Recycle Trajectory Analysis', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.show()
    
    # Print summary statistics
    print("\n" + "="*60)
    print("RECYCLE TRAJECTORY SUMMARY")
    print("="*60)
    
    print(f"\nInitial state (Recycle 0):")
    print(f"  RMSD to State 1: {rmsd1_vals[0]:.2f} Å")
    print(f"  RMSD to State 2: {rmsd2_vals[0]:.2f} Å")
    print(f"  pLDDT: {plddt_vals[0]:.1f}%")
    
    print(f"\nFinal state (Recycle {recycles[-1]}):")
    print(f"  RMSD to State 1: {rmsd1_vals[-1]:.2f} Å")
    print(f"  RMSD to State 2: {rmsd2_vals[-1]:.2f} Å")
    print(f"  pLDDT: {plddt_vals[-1]:.1f}%")
    
    print(f"\nTrajectory changes:")
    print(f"  ΔRMSD to State 1: {rmsd1_vals[-1] - rmsd1_vals[0]:+.2f} Å")
    print(f"  ΔRMSD to State 2: {rmsd2_vals[-1] - rmsd2_vals[0]:+.2f} Å")
    print(f"  ΔpLDDT: {plddt_vals[-1] - plddt_vals[0]:+.1f}%")
    
    # Determine conformational preference
    final_preference = "State 1" if rmsd1_vals[-1] < rmsd2_vals[-1] else "State 2"
    print(f"\nFinal conformation closer to: {final_preference}")
    
    # Check for conformational switching
    switches = 0
    for i in range(1, len(rmsd_diff)):
        if rmsd_diff[i] * rmsd_diff[i-1] < 0:  # Sign change
            switches += 1
    
    if switches > 0:
        print(f"\nConformational switches detected: {switches}")
        print("  → Structure explores multiple conformations during recycling!")
    else:
        print("\nNo conformational switches detected")
        print("  → Structure converges consistently toward one state")
        
else:
    print("No recycle trajectory found. Run Section 5 first to generate recycling data.")


# %% [markdown]
# ## GPU Acceleration for Large-Scale RMSD Calculations
# 
# ### Why GPU Acceleration Matters
# 
# When analyzing conformational ensembles from AlphaFold2, RMSD calculations can become a computational bottleneck:
# 
# - **Sequential CPU Processing**: Traditional approaches calculate RMSDs one at a time, leading to O(N²) time complexity for all-vs-all comparisons
# - **GPU Batch Processing**: Leverages parallel computation using JAX's vectorization capabilities
# - **Performance Gains**: Typically 5-50x speedup depending on ensemble size and GPU hardware
# 
# ### How GPU Acceleration Works
# 
# The GPU-accelerated RMSD functions in `af2_utils` use several optimization strategies:
# 
# 1. **JAX Compilation**: Functions are JIT-compiled for optimal performance
#    - `@jax.jit` decorator compiles functions to XLA (Accelerated Linear Algebra)
#    - First call compiles, subsequent calls are near-instant
# 
# 2. **Vectorization with vmap**: Automatic parallelization across batches
#    ```python
#    _rmsd_parallel_jax = jax.jit(jax.vmap(_rmsd_jax, (None, 0)))
#    ```
#    - Transforms single-structure function to batch operation
#    - Executes all calculations simultaneously on GPU
# 
# 3. **Optimized Kabsch Algorithm**: Fast optimal superposition
#    - Matrix operations leverage GPU's parallel architecture
#    - SVD decomposition accelerated by specialized GPU kernels
# 
# ### When to Use GPU Acceleration
# 
# GPU acceleration is particularly beneficial for:
# 
# - **Large Conformational Ensembles**: >10 structures
# - **All-vs-All RMSD Matrices**: Quadratic scaling benefits most from parallelization
# - **Real-time Analysis**: Interactive exploration during prediction
# - **High-throughput Screening**: Processing multiple proteins or conditions
# 
# ### Performance Comparison
# 
# | Task | CPU Time | GPU Time | Speedup |
# |------|----------|----------|---------|
# | 10 structures to 2 references | ~0.5s | ~0.02s | 25x |
# | 50×50 RMSD matrix | ~12s | ~0.3s | 40x |
# | 100 structures batch | ~20s | ~0.5s | 40x |
# 
# *Note: Actual performance depends on hardware and data size*
# 
# ### Best Practices
# 
# 1. **Batch Operations**: Process multiple structures at once rather than iterating
# 2. **Memory Management**: GPU memory is limited - process in chunks for very large datasets
# 3. **Fallback Strategy**: Always include CPU fallback for environments without GPU
# 4. **Data Transfer**: Minimize CPU-GPU data transfers by batching operations
# 
# ### Implementation in This Tutorial
# 
# Throughout this tutorial, we use GPU acceleration for:
# - Batch RMSD calculations to reference states
# - All-vs-all ensemble RMSD matrices
# - Recycle trajectory analysis
# - Performance benchmarking
# 
# The functions automatically detect GPU availability and fall back to CPU when needed, ensuring compatibility across all environments.
# 

# %% [markdown]
# ## Section 6: Sampling Multiple Conformations
# 
# Now let's explore techniques for sampling multiple conformations using dropout and different random seeds.
# 

# %%
%%time
#@title Generate Conformational Ensemble (With PDB Saving)

print("="*60)
print("GENERATING CONFORMATIONAL ENSEMBLE WITH PDB SAVING")
print("="*60)

# Use high-level API to generate ensemble with different MSA conditions
# Now with PDB saving for all predictions!
all_predictions = af2.predict_conformational_ensemble(
    sequence=I89_SEQUENCE,
    msa_modes=["mmseqs2", "single_sequence"],
    num_seeds=3,
    num_recycles=3,
    use_dropout=True,
    jobname="i89_ensemble",
    save_all_pdbs=True,  # NEW: Save all prediction PDBs
    verbose=True
)

print(f"\nGenerated {len(all_predictions)} structures total")

# GPU-ACCELERATED RMSD CALCULATION
# Extract all structures at once for batch processing
pred_coords_list = [pred['structure'] for pred in all_predictions]

print("\nCalculating RMSDs using GPU acceleration...")
# Use batch GPU calculation for both reference states
rmsds = af2.calculate_batch_rmsd_to_references(
    pred_coords_list,
    ref1_path="state1.pdb",
    ref2_path="state2.pdb",
    use_gpu=True  # Enable GPU acceleration
)

# Combine with metadata
ensemble_rmsds = []
for pred, rmsd_dict in zip(all_predictions, rmsds):
    ensemble_rmsds.append({
        'msa_mode': pred['msa_mode'],
        'seed': pred['seed'],
        'rmsd_state1': rmsd_dict['rmsd_state1'],
        'rmsd_state2': rmsd_dict['rmsd_state2'],
        'plddt': pred['metrics']['plddt'] * 100
    })

# Analyze by MSA mode
with_msa = [r for r in ensemble_rmsds if r['msa_mode'] == 'mmseqs2']
without_msa = [r for r in ensemble_rmsds if r['msa_mode'] == 'single_sequence']

print("\n" + "="*60)
print("ENSEMBLE STATISTICS")
print("="*60)
print(f"\nWith MSA ({len(with_msa)} structures):")
print(f"  Mean RMSD to State 1: {np.mean([r['rmsd_state1'] for r in with_msa]):.2f} ± {np.std([r['rmsd_state1'] for r in with_msa]):.2f} Å")
print(f"  Mean RMSD to State 2: {np.mean([r['rmsd_state2'] for r in with_msa]):.2f} ± {np.std([r['rmsd_state2'] for r in with_msa]):.2f} Å")
print(f"  Mean pLDDT: {np.mean([r['plddt'] for r in with_msa]):.1f}%")

print(f"\nWithout MSA ({len(without_msa)} structures):")
print(f"  Mean RMSD to State 1: {np.mean([r['rmsd_state1'] for r in without_msa]):.2f} ± {np.std([r['rmsd_state1'] for r in without_msa]):.2f} Å")
print(f"  Mean RMSD to State 2: {np.mean([r['rmsd_state2'] for r in without_msa]):.2f} ± {np.std([r['rmsd_state2'] for r in without_msa]):.2f} Å")
print(f"  Mean pLDDT: {np.mean([r['plddt'] for r in without_msa]):.1f}%")


# %%
#@title Browse Ensemble PDB Files
#@markdown Explore the saved ensemble PDB files

if all_predictions and 'job_folder' in all_predictions[0]:
    ensemble_folder = all_predictions[0]['job_folder']
    
    print("="*60)
    print("ENSEMBLE PDB FILES")
    print("="*60)
    print(f"Ensemble folder: {ensemble_folder}/")
    print(f"  (Sequence hash: {af2.get_hash(I89_SEQUENCE)})")
    print()
    
    # List all ensemble PDB files
    ensemble_pdbs = sorted(glob.glob(f"{ensemble_folder}/pdb/*.pdb"))
    
    if ensemble_pdbs:
        print(f"Found {len(ensemble_pdbs)} PDB files:")
        
        # Group by type
        msa_pdbs = [p for p in ensemble_pdbs if 'mmseqs2' in p]
        single_pdbs = [p for p in ensemble_pdbs if 'single_sequence' in p]
        best_pdb = [p for p in ensemble_pdbs if 'best.pdb' in p]
        
        if msa_pdbs:
            print("\nWith MSA (mmseqs2):")
            for pdb in msa_pdbs:
                print(f"  - {os.path.basename(pdb)}")
        
        if single_pdbs:
            print("\nWithout MSA (single_sequence):")
            for pdb in single_pdbs:
                print(f"  - {os.path.basename(pdb)}")
        
        if best_pdb:
            print("\nBest prediction:")
            print(f"  - {os.path.basename(best_pdb[0])}")
        
        print("\n" + "="*60)
        print("DOWNLOAD INSTRUCTIONS")
        print("="*60)
        print("To download all PDB files:")
        print(f"  1. Navigate to: {ensemble_folder}/pdb/")
        print("  2. Select all files")
        print("  3. Right-click and download")
        print("\nOr use command line:")
        print(f"  !zip -r ensemble_pdbs.zip {ensemble_folder}/pdb/")
        print("  # Then download ensemble_pdbs.zip")
        
        print("\n" + "="*60)
        print("ANALYSIS IDEAS")
        print("="*60)
        print("With these PDB files you can:")
        print("  - Align all structures in PyMOL/ChimeraX")
        print("  - Calculate pairwise RMSD matrix")
        print("  - Perform clustering analysis")
        print("  - Create morphing animations")
        print("  - Compare MSA vs no-MSA conformations")
else:
    print("No ensemble predictions found. Run the previous cell first!")


# %%
#@title Visualize Ensemble Distribution

# Create RMSD scatter plot
fig, ax = plt.subplots(figsize=(10, 8))

# Plot points colored by MSA mode
for r in ensemble_rmsds:
    color = 'steelblue' if r['msa_mode'] == 'mmseqs2' else 'coral'
    marker = 'o' if r['msa_mode'] == 'mmseqs2' else 's'
    label = 'With MSA' if r['msa_mode'] == 'mmseqs2' else 'Without MSA'
    ax.scatter(r['rmsd_state1'], r['rmsd_state2'], 
              c=color, marker=marker, s=100, alpha=0.7,
              label=label if r['seed'] == 0 else "")

# Add reference point (State 1 vs State 2)
ax.scatter([0], [ref_rmsd], marker='*', s=500, c='red', 
          label=f'State 1 vs State 2 ({ref_rmsd:.1f}Å)')

# Add diagonal line
max_rmsd = max(max([r['rmsd_state1'] for r in ensemble_rmsds]),
               max([r['rmsd_state2'] for r in ensemble_rmsds]))
ax.plot([0, max_rmsd], [0, max_rmsd], 'k--', alpha=0.3)

ax.set_xlabel('RMSD to State 1 (Å)', fontsize=12)
ax.set_ylabel('RMSD to State 2 (Å)', fontsize=12)
ax.set_title('Ensemble Distribution in RMSD Space', fontsize=14)
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Analyze ensemble diversity
structures = [pred['structure'] for pred in all_predictions]
ensemble_stats = af2.analyze_ensemble(structures, verbose=True)


# %%
%%time
#@title GPU-Accelerated All-vs-All RMSD Matrix
#@markdown Compute and visualize pairwise RMSD between all structures using GPU acceleration

import matplotlib.pyplot as plt
import seaborn as sns

print("="*60)
print("ALL-VS-ALL RMSD MATRIX (GPU-ACCELERATED)")
print("="*60)

# Extract CA coordinates for all structures
ca_coords = [pred['structure'][:, 1, :] for pred in all_predictions]

print(f"Computing {len(ca_coords)}×{len(ca_coords)} RMSD matrix using GPU...")

# Calculate all-vs-all RMSD matrix using GPU
rmsd_matrix, mean_pairwise_rmsd = af2.calculate_all_vs_all_rmsd(
    ca_coords,
    use_gpu=True  # Enable GPU acceleration
)

# Create labels for the heatmap
labels = [f"{p['msa_mode'][:3]}_s{p['seed']}" for p in all_predictions]

# Visualize RMSD matrix
plt.figure(figsize=(12, 10))
sns.heatmap(rmsd_matrix, 
            cmap='viridis', 
            square=True,
            cbar_kws={'label': 'RMSD (Å)'},
            xticklabels=labels,
            yticklabels=labels,
            annot=True if len(ca_coords) <= 10 else False,
            fmt='.1f')

plt.title(f'All-vs-All RMSD Matrix\nMean Pairwise RMSD: {mean_pairwise_rmsd:.2f} Å', fontsize=14)
plt.xlabel('Structure', fontsize=12)
plt.ylabel('Structure', fontsize=12)
plt.tight_layout()
plt.show()

# Calculate statistics
upper_tri_indices = np.triu_indices_from(rmsd_matrix, k=1)
rmsd_values = rmsd_matrix[upper_tri_indices]

print("\n" + "="*60)
print("ENSEMBLE DIVERSITY METRICS")
print("="*60)
print(f"Mean pairwise RMSD: {mean_pairwise_rmsd:.2f} Å")
print(f"Std deviation: {rmsd_values.std():.2f} Å")
print(f"Min RMSD: {rmsd_values.min():.2f} Å")
print(f"Max RMSD: {rmsd_values.max():.2f} Å")

# Analyze clustering by MSA mode
with_msa_indices = [i for i, p in enumerate(all_predictions) if p['msa_mode'] == 'mmseqs2']
without_msa_indices = [i for i, p in enumerate(all_predictions) if p['msa_mode'] == 'single_sequence']

if with_msa_indices and without_msa_indices:
    # Within-group RMSDs
    within_msa_rmsds = []
    for i in range(len(with_msa_indices)):
        for j in range(i+1, len(with_msa_indices)):
            within_msa_rmsds.append(rmsd_matrix[with_msa_indices[i], with_msa_indices[j]])
    
    within_nomsa_rmsds = []
    for i in range(len(without_msa_indices)):
        for j in range(i+1, len(without_msa_indices)):
            within_nomsa_rmsds.append(rmsd_matrix[without_msa_indices[i], without_msa_indices[j]])
    
    # Between-group RMSDs
    between_rmsds = []
    for i in with_msa_indices:
        for j in without_msa_indices:
            between_rmsds.append(rmsd_matrix[i, j])
    
    print("\nClustering Analysis:")
    if within_msa_rmsds:
        print(f"  Within MSA group: {np.mean(within_msa_rmsds):.2f} ± {np.std(within_msa_rmsds):.2f} Å")
    if within_nomsa_rmsds:
        print(f"  Within no-MSA group: {np.mean(within_nomsa_rmsds):.2f} ± {np.std(within_nomsa_rmsds):.2f} Å")
    if between_rmsds:
        print(f"  Between groups: {np.mean(between_rmsds):.2f} ± {np.std(between_rmsds):.2f} Å")
    
    print("\nInterpretation:")
    if np.mean(between_rmsds) > max(np.mean(within_msa_rmsds) if within_msa_rmsds else 0, 
                                     np.mean(within_nomsa_rmsds) if within_nomsa_rmsds else 0):
        print("  ✓ Clear separation between MSA and no-MSA conformations")
        print("  → MSA presence drives conformational selection")
    else:
        print("  ✗ Overlapping conformational distributions")
        print("  → Other factors (seeds, dropout) also influence conformation")


# %%
%%time
#@title GPU vs CPU Performance Comparison
#@markdown Compare the speed improvement from GPU acceleration for RMSD calculations

import time

if len(all_predictions) >= 5:  # Only run if we have enough structures
    print("="*60)
    print("GPU vs CPU PERFORMANCE COMPARISON")
    print("="*60)
    
    # Extract coordinates for testing (limit to 10 for fair comparison)
    test_coords = [pred['structure'] for pred in all_predictions[:min(10, len(all_predictions))]]
    n_structures = len(test_coords)
    
    print(f"Testing with {n_structures} structures...")
    print(f"Total RMSD calculations: {n_structures * 2} (to both reference states)")
    print()
    
    # CPU timing
    print("Running CPU benchmark...")
    start_cpu = time.time()
    rmsds_cpu = af2.calculate_batch_rmsd_to_references(
        test_coords, 
        ref1_path="state1.pdb",
        ref2_path="state2.pdb",
        use_gpu=False  # Force CPU
    )
    cpu_time = time.time() - start_cpu
    print(f"  CPU Time: {cpu_time:.3f} seconds")
    
    # GPU timing (if available)
    try:
        print("\nRunning GPU benchmark...")
        start_gpu = time.time()
        rmsds_gpu = af2.calculate_batch_rmsd_to_references(
            test_coords,
            ref1_path="state1.pdb", 
            ref2_path="state2.pdb",
            use_gpu=True  # Force GPU
        )
        gpu_time = time.time() - start_gpu
        print(f"  GPU Time: {gpu_time:.3f} seconds")
        
        # Verify results are consistent
        cpu_vals = [r['rmsd_state1'] for r in rmsds_cpu]
        gpu_vals = [r['rmsd_state1'] for r in rmsds_gpu]
        max_diff = max(abs(c - g) for c, g in zip(cpu_vals, gpu_vals))
        
        print("\n" + "="*60)
        print("PERFORMANCE RESULTS")
        print("="*60)
        print(f"Speedup: {cpu_time/gpu_time:.1f}x faster with GPU")
        print(f"Time saved: {cpu_time - gpu_time:.3f} seconds")
        print(f"Result consistency: Max difference = {max_diff:.6f} Å")
        
        # Visualize performance comparison
        import matplotlib.pyplot as plt
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Bar chart comparison
        methods = ['CPU', 'GPU']
        times = [cpu_time, gpu_time]
        colors = ['coral', 'steelblue']
        
        bars = ax1.bar(methods, times, color=colors, width=0.6)
        ax1.set_ylabel('Time (seconds)', fontsize=12)
        ax1.set_title('Execution Time Comparison', fontsize=14)
        ax1.set_ylim(0, max(times) * 1.2)
        
        # Add value labels on bars
        for bar, time_val in zip(bars, times):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{time_val:.3f}s',
                    ha='center', va='bottom', fontsize=11)
        
        # Speedup visualization
        speedup = cpu_time / gpu_time
        ax2.barh(['Speedup'], [speedup], color='green', height=0.3)
        ax2.set_xlabel('Speedup Factor', fontsize=12)
        ax2.set_title('GPU Acceleration Factor', fontsize=14)
        ax2.set_xlim(0, max(speedup * 1.2, 2))
        ax2.text(speedup, 0, f'  {speedup:.1f}x', 
                va='center', ha='left', fontsize=14, fontweight='bold')
        
        plt.suptitle(f'RMSD Calculation Performance ({n_structures} structures)', fontsize=16)
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"\nGPU acceleration not available: {e}")
        print("Falling back to CPU-only execution")
        print("\nTo enable GPU acceleration:")
        print("  1. Ensure CUDA is installed")
        print("  2. Install JAX with GPU support: pip install --upgrade jax[cuda11_pip]")
        print("  3. Restart the kernel")
        
else:
    print("Not enough structures for meaningful performance comparison")
    print("Generate an ensemble first (Section 6)")


# %%
#@title Create LogMD Trajectory from Ensemble
#@markdown Visualize the entire ensemble as an interactive 3D trajectory

if af2.check_logmd():
    print("Creating LogMD trajectory from ensemble predictions...")
    
    # Create trajectory sorted by RMSD to State 2
    trajectory = af2.create_trajectory_from_ensemble(
        predictions=all_predictions,
        sequence=I89_SEQUENCE,
        project="i89_ensemble",
        align_structures=True,
        sort_by_rmsd=True,
        reference_coords=state2_coords,
        max_structures=20,  # Limit for faster loading
        verbose=True
    )
    
    if trajectory:
        print("\n" + "="*60)
        print("ENSEMBLE TRAJECTORY CREATED")
        print("="*60)
        print("Features:")
        print("  - All structures aligned for comparison")
        print("  - Sorted by RMSD to State 2")
        print("  - Colored by pLDDT confidence")
        print("  - Animated transition between conformations")
        print("\nUse the viewer controls to:")
        print("  - Play/pause the animation")
        print("  - Step through individual frames")
        print("  - Rotate and zoom the view")
        
        # Display in notebook
        try:
            import logmd_utils
            logmd_utils.display_trajectory_in_notebook(trajectory)
        except:
            print(f"\nView trajectory at: {trajectory.url}")
    else:
        print("Failed to create trajectory")
else:
    print("LogMD not available - skipping ensemble trajectory")
    print("\nTo use this feature:")
    print("  1. Install LogMD: pip install logmd")
    print("  2. Restart kernel")
    print("  3. Re-run this cell")


# %% [markdown]
# 

# %%
#@title Compare MSA Conditions with LogMD
#@markdown Create separate trajectories for with/without MSA predictions

if af2.check_logmd():
    print("Creating separate trajectories for MSA comparison...")
    
    # Separate predictions by MSA mode
    with_msa_preds = [p for p in all_predictions if p.get('msa_mode') == 'mmseqs2']
    without_msa_preds = [p for p in all_predictions if p.get('msa_mode') == 'single_sequence']
    
    print(f"\nWith MSA: {len(with_msa_preds)} predictions")
    print(f"Without MSA: {len(without_msa_preds)} predictions")
    
    # Create trajectory for predictions with MSA
    if with_msa_preds:
        traj_with_msa = af2.create_trajectory_from_ensemble(
            predictions=with_msa_preds,
            sequence=I89_SEQUENCE,
            project="i89_with_msa",
            align_structures=True,
            sort_by_rmsd=True,
            reference_coords=state1_coords,  # Sort by state 1
            verbose=False
        )
        if traj_with_msa:
            print(f"\nWith MSA trajectory: {traj_with_msa.url}")
    
    # Create trajectory for predictions without MSA
    if without_msa_preds:
        traj_without_msa = af2.create_trajectory_from_ensemble(
            predictions=without_msa_preds,
            sequence=I89_SEQUENCE,
            project="i89_without_msa",
            align_structures=True,
            sort_by_rmsd=True,
            reference_coords=state2_coords,  # Sort by state 2
            verbose=False
        )
        if traj_without_msa:
            print(f"Without MSA trajectory: {traj_without_msa.url}")
    
    print("\n" + "="*60)
    print("MSA COMPARISON TRAJECTORIES")
    print("="*60)
    print("Key observations:")
    print("  - With MSA predictions cluster near State 1")
    print("  - Without MSA predictions cluster near State 2")
    print("  - MSA provides evolutionary bias toward native state")
    print("\nOpen both URLs in separate tabs to compare side-by-side!")
else:
    print("LogMD not available - skipping MSA comparison")


# %%
#@title Create Progressive Ensemble Animation
#@markdown Watch structures being added one by one to the ensemble

if af2.check_logmd():
    print("Creating progressive ensemble animation...")
    print("This shows how the ensemble builds up structure by structure!\n")
    
    # Sort by generation order (MSA mode first, then by seed)
    sorted_ensemble = sorted(
        all_predictions, 
        key=lambda x: (x.get('msa_mode', ''), x.get('seed', 0))
    )
    
    traj = af2.create_trajectory_from_ensemble(
        predictions=sorted_ensemble,
        sequence=I89_SEQUENCE,
        project="i89_ensemble_progressive",
        align_structures=True,
        max_structures=20,  # Limit for performance
        verbose=False
    )
    
    if traj:
        print("="*60)
        print("PROGRESSIVE ENSEMBLE ANIMATION")
        print("="*60)
        print(f"View progressive build-up: {traj.url}")
        print("\nThis animation shows:")
        print("  - First 3 frames: Predictions WITH MSA (seed 0, 1, 2)")
        print("  - Next 3 frames: Predictions WITHOUT MSA (seed 0, 1, 2)")
        print("  - Notice the conformational transition!")
        print("\nTips for viewing:")
        print("  - Play the animation to see ensemble grow")
        print("  - Pause to inspect individual structures")
        print("  - Compare how different seeds explore conformational space")
    else:
        print("Failed to create progressive animation")
else:
    print("LogMD not available - skipping progressive animation")
    print("This feature shows how the ensemble builds up incrementally")


# %%



# %% [markdown]
# ## Section 7: Advanced Analysis - Coevolution and MSA Quality
# 
# Let's examine how MSA quality and coevolution patterns influence predictions.
# 

# %%
#@title Generate and Analyze MSA with Interactive Visualizations
#@markdown This cell generates the MSA and provides interactive plots to explore MSA quality and coevolution patterns

print("Generating MSA for i89 protein...")
print("This will use MMseqs2 to find homologous sequences\n")

# Generate MSA using MMseqs2
msa_full, del_matrix = af2.get_msa([I89_SEQUENCE], "i89_msa_analysis", verbose=False)

# Parse MSA into structured format for visualization
from colabdesign.af.contrib import predict
sequences, deletion_matrix = predict.parse_a3m("i89_msa_analysis/msa.a3m")

# Create MSA visualizer
vis = af2.MSACoevolutionVisualizer()

# Load MSA data with metadata
msa_data = af2.MSAData(
    array=msa_full,
    deletion_matrix=del_matrix,
    sequences=sequences,
    neff=msa_full.shape[0],
    length=msa_full.shape[1],
    condition_name="i89 MMseqs2 MSA"
)

print(f"MSA Generated Successfully!")
print(f"  Sequences found: {msa_data.neff}")
print(f"  Sequence length: {msa_data.length}")
print(f"\n" + "="*60)

# Print diagnostics
vis.print_diagnostics(msa_data)

print("\n" + "="*60)
print("Creating interactive visualizations...")
print("Hover over plots to see detailed information!")

# Create interactive MSA quality plots
fig_msa = af2.plot_msa_interactive(msa_data, title="i89 MSA Quality Analysis")
fig_msa.show()

print("\nKey observations about MSA quality:")
print("  - Coverage shows how well each position is represented")
print("  - Sequence identity distribution indicates MSA diversity")
print("  - Higher diversity generally provides better evolutionary signal")


# %%
#@title Interactive Coevolution Analysis
#@markdown Explore residue-residue coevolution with interactive heatmap. 
#@markdown Hover over the plot to see which amino acids are coevolving!

print("Computing coevolution matrix...")
print("This uses Direct Coupling Analysis (DCA) to identify coevolving residue pairs\n")

# Compute coevolution with caching
coev_matrix = vis.compute_coevolution(msa_data)

# Create interactive coevolution plot
fig_coev = vis.plot_heatmap(
    coev_matrix, 
    title="i89 Coevolution Matrix - Interactive",
    msa_data=msa_data
)

fig_coev.show()

print("\n" + "="*60)
print("Understanding the Coevolution Plot:")
print("="*60)
print("  - Brighter colors indicate stronger coevolution")
print("  - Hover over any position to see:")
print("    * Residue identities at positions i and j")
print("    * Coevolution score")
print("    * Overall MSA statistics")
print("  - Diagonal is zeroed out (self-coevolution)")
print("\nBiological Insight:")
print("  - Strong coevolution often indicates:")
print("    1. Structural contacts (residues close in 3D)")
print("    2. Functional coupling (allosteric networks)")
print("    3. Compensatory mutations maintaining protein stability")

# Identify calcium-binding loop region (residues ~85-95)
print("\n" + "="*60)
print("CALCIUM-BINDING LOOP ANALYSIS:")
print("="*60)
print("The calcium-binding loop (around residues 85-95) shows strong coevolution.")
print("This is the region that changes conformation between State 1 and State 2!")
print(f"\nCoevolution matrix shape: {coev_matrix.shape}")
print(f"Expected sequence length: {len(I89_SEQUENCE)}")
if coev_matrix.shape[0] != len(I89_SEQUENCE):
    print(f"Note: Matrix size differs from sequence length (possibly due to MSA processing)")
print("\nLook for strong coevolution signals in this region:")

# Extract coevolution for calcium-binding region
# Ensure we don't exceed matrix bounds
matrix_size = coev_matrix.shape[0]
ca_start = 84  # 0-indexed start (residue 85)
ca_end = min(96, matrix_size)  # 0-indexed end (residue 96), but don't exceed matrix size
ca_region = range(ca_start, ca_end)
ca_coev_submatrix = coev_matrix[np.ix_(ca_region, ca_region)]
max_ca_coev = np.max(ca_coev_submatrix) if ca_coev_submatrix.size > 0 else 0
mean_ca_coev = np.mean(ca_coev_submatrix[np.triu_indices_from(ca_coev_submatrix, k=1)]) if ca_coev_submatrix.size > 0 else 0

print(f"  Analyzing residues {ca_start+1} to {ca_end} (0-indexed: {ca_start}:{ca_end})")
print(f"  Max coevolution in Ca-binding region: {max_ca_coev:.3f}")
print(f"  Mean coevolution in Ca-binding region: {mean_ca_coev:.3f}")
# Calculate overall mean properly excluding diagonal
upper_tri_all = np.triu_indices_from(coev_matrix, k=1)
overall_mean = np.mean(coev_matrix[upper_tri_all]) if coev_matrix.size > 0 else 0
print(f"  Overall mean coevolution: {overall_mean:.3f}")

# Find top coevolving pairs involving the calcium-binding loop
upper_tri = np.triu_indices_from(coev_matrix, k=6)
coev_values = coev_matrix[upper_tri]
top_indices = np.argsort(coev_values)[-10:]  # Top 10 pairs

print("\nTop 10 coevolving residue pairs:")
for idx in reversed(top_indices):
    i, j = upper_tri[0][idx], upper_tri[1][idx]
    score = coev_values[idx]
    # Check if either residue is in Ca-binding region
    in_ca_region = (ca_start <= i < ca_end) or (ca_start <= j < ca_end)
    marker = " [Ca-binding loop]" if in_ca_region else ""
    print(f"  Positions {i+1:3d}-{j+1:3d}: {score:.4f}{marker}")


# %%
#@title Compare Coevolution: With vs Without MSA
#@markdown This demonstrates how MSA depth affects coevolution signal and ultimately structure prediction

print("Creating single-sequence MSA (no homologs) for comparison...")
print("This simulates what happens when we predict without MSA context\n")

# Create single-sequence MSA (only accepts one argument)
msa_single, deletion_matrix_single = af2.create_single_sequence_msa(I89_SEQUENCE)

# Create sequences array for MSAData (single sequence)
sequences_single = [np.array(list(I89_SEQUENCE))]

# Create MSAData for single sequence
msa_data_single = af2.MSAData(
    array=msa_single,
    deletion_matrix=deletion_matrix_single,
    sequences=sequences_single,
    neff=1,
    length=len(I89_SEQUENCE),
    condition_name="i89 Single Sequence (No MSA)"
)

print("Computing coevolution for single-sequence MSA...")
coev_single = vis.compute_coevolution(msa_data_single)

print("\n" + "="*60)
print("COMPARING COEVOLUTION SIGNALS:")
print("="*60)

# Compare using the comparison function
conditions = {
    "With MSA (MMseqs2)": msa_data,
    "Without MSA (Single Sequence)": msa_data_single
}

fig_main, fig_diff = af2.compare_coevolution_conditions(
    conditions,
    show_difference=True,
    reference_condition="Without MSA (Single Sequence)"
)

print("\nShowing side-by-side comparison...")
fig_main.show()

if fig_diff is not None:
    print("\nShowing difference plot...")
    print("(Positive values = stronger coevolution WITH MSA)")
    fig_diff.show()

print("\n" + "="*60)
print("KEY INSIGHTS:")
print("="*60)
print("1. WITH MSA (left):")
print("   - Clear coevolution patterns visible")
print("   - Strong signal in calcium-binding loop")
print("   - Evolutionary information guides AF2 to State 1")
print("\n2. WITHOUT MSA (right):")
print("   - Minimal coevolution (single sequence)")
print("   - No evolutionary guidance")
print("   - AF2 defaults to alternative conformation (State 2)")
print("\n3. DIFFERENCE PLOT:")
print("   - Shows regions where MSA provides most information")
print("   - Red/blue regions indicate differential coevolution")
print("   - Calcium-binding loop shows strong difference!")
print("\nThis explains WHY we see different conformations:")
print("  - With MSA: Evolutionary pressure maintains State 1 (Ca-bound)")
print("  - Without MSA: AF2 explores alternative State 2 (Ca-free)")
print("  - MSA depth directly influences conformational outcome!")


# %%
#@title Focused Analysis: Calcium-Binding Loop Coevolution
#@markdown Zoom into the calcium-binding loop to see the evolutionary signal that drives State 1 prediction

print("Creating focused visualization of calcium-binding loop region...")
print("This region (residues 85-95) is critical for calcium binding\n")

# Extract subregion around calcium-binding loop
ca_start, ca_end = 75, 105  # Broader region for context (0-indexed)
ca_region_indices = range(ca_start, min(ca_end, len(I89_SEQUENCE)))

# Create submatrices for the calcium-binding region
coev_full_subregion = coev_matrix[ca_start:ca_end, ca_start:ca_end]
coev_single_subregion = coev_single[ca_start:ca_end, ca_start:ca_end]

# Create comparison plot for calcium-binding region
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=[
            "With MSA (Ca-binding region)",
            "Without MSA (Ca-binding region)", 
            "Difference (With - Without)"
        ],
        horizontal_spacing=0.12
    )
    
    # With MSA
    fig.add_trace(
        go.Heatmap(
            z=coev_full_subregion,
            colorscale="Viridis",
            showscale=False,
            x=list(range(ca_start+1, min(ca_end+1, len(I89_SEQUENCE)+1))),
            y=list(range(ca_start+1, min(ca_end+1, len(I89_SEQUENCE)+1))),
            hovertemplate="Position i: %{x}<br>Position j: %{y}<br>Coevolution: %{z:.3f}<extra></extra>"
        ),
        row=1, col=1
    )
    
    # Without MSA
    fig.add_trace(
        go.Heatmap(
            z=coev_single_subregion,
            colorscale="Viridis",
            showscale=False,
            x=list(range(ca_start+1, min(ca_end+1, len(I89_SEQUENCE)+1))),
            y=list(range(ca_start+1, min(ca_end+1, len(I89_SEQUENCE)+1))),
            hovertemplate="Position i: %{x}<br>Position j: %{y}<br>Coevolution: %{z:.3f}<extra></extra>"
        ),
        row=1, col=2
    )
    
    # Difference
    diff_ca = coev_full_subregion - coev_single_subregion
    max_abs = float(np.max(np.abs(diff_ca))) or 1.0
    
    fig.add_trace(
        go.Heatmap(
            z=diff_ca,
            colorscale="RdBu_r",
            zmid=0,
            zmin=-max_abs,
            zmax=max_abs,
            showscale=True,
            x=list(range(ca_start+1, min(ca_end+1, len(I89_SEQUENCE)+1))),
            y=list(range(ca_start+1, min(ca_end+1, len(I89_SEQUENCE)+1))),
            hovertemplate="Position i: %{x}<br>Position j: %{y}<br>Difference: %{z:.3f}<extra></extra>"
        ),
        row=1, col=3
    )
    
    fig.update_layout(
        title="Calcium-Binding Loop Coevolution Analysis (Residues 76-105)",
        height=500,
        width=1400
    )
    
    fig.update_xaxes(title_text="Residue Position", row=1, col=1)
    fig.update_xaxes(title_text="Residue Position", row=1, col=2)
    fig.update_xaxes(title_text="Residue Position", row=1, col=3)
    fig.update_yaxes(title_text="Residue Position", row=1, col=1)
    fig.update_yaxes(title_text="Residue Position", row=1, col=2)
    fig.update_yaxes(title_text="Residue Position", row=1, col=3)
    
    fig.show()
    
    print("\n" + "="*60)
    print("CALCIUM-BINDING LOOP EVOLUTIONARY SIGNAL:")
    print("="*60)
    print(f"Region analyzed: Residues {ca_start+1}-{min(ca_end, len(I89_SEQUENCE))}")
    print(f"\nCoevolution statistics for Ca-binding loop core (85-95):")
    print(f"  With MSA    : Mean = {mean_ca_coev:.4f}, Max = {max_ca_coev:.4f}")
    print(f"  Without MSA : Mean = {np.mean(coev_single[84:96, 84:96]):.4f}, Max = {np.max(coev_single[84:96, 84:96]):.4f}")
    
    print("\n" + "="*60)
    print("BIOLOGICAL INTERPRETATION:")
    print("="*60)
    print("The strong coevolution signal in this region reveals:")
    print("  1. Evolutionary conservation of calcium-binding function")
    print("  2. Coordinated mutations maintaining Ca2+ binding geometry")
    print("  3. This guides AlphaFold2 toward the functional State 1 conformation")
    print("\nWhen MSA is absent:")
    print("  - No coevolution signal to guide prediction")
    print("  - AlphaFold2 explores alternative low-energy state (State 2)")
    print("  - Both states are valid local minima in the energy landscape")
    print("\nConclusion:")
    print("  MSA depth is not just about sequence identity - it's about")
    print("  capturing functional constraints through evolutionary coupling!")

except ImportError:
    print("Plotly not available - using matplotlib fallback")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    im1 = axes[0].imshow(coev_full_subregion, cmap='viridis')
    axes[0].set_title("With MSA")
    axes[0].set_xlabel("Position")
    axes[0].set_ylabel("Position")
    plt.colorbar(im1, ax=axes[0])
    
    im2 = axes[1].imshow(coev_single_subregion, cmap='viridis')
    axes[1].set_title("Without MSA")
    axes[1].set_xlabel("Position")
    plt.colorbar(im2, ax=axes[1])
    
    diff_ca = coev_full_subregion - coev_single_subregion
    im3 = axes[2].imshow(diff_ca, cmap='RdBu_r', vmin=-np.max(np.abs(diff_ca)), vmax=np.max(np.abs(diff_ca)))
    axes[2].set_title("Difference")
    axes[2].set_xlabel("Position")
    plt.colorbar(im3, ax=axes[2])
    
    plt.tight_layout()
    plt.show()


# %% [markdown]
# ## Section 8: Saving Results
# 
# Let's save our predictions for further analysis.
# 

# %%
#@title Save Best Predictions to PDB

# Save predictions with highest confidence
best_with_msa = max(with_msa, key=lambda x: x['plddt'])
best_without_msa = max(without_msa, key=lambda x: x['plddt'])

# Find corresponding structures
for pred in all_predictions:
    if pred['msa_mode'] == 'mmseqs2' and pred['seed'] == best_with_msa['seed']:
        af2.save_pdb(
            atom_positions=pred['structure'],
            sequence=I89_SEQUENCE,
            output_path="i89_best_with_msa.pdb",
            plddt=pred['plddt']
        )
        print(f"Saved: i89_best_with_msa.pdb (pLDDT: {best_with_msa['plddt']:.1f}%)")
        break

for pred in all_predictions:
    if pred['msa_mode'] == 'single_sequence' and pred['seed'] == best_without_msa['seed']:
        af2.save_pdb(
            atom_positions=pred['structure'],
            sequence=I89_SEQUENCE,
            output_path="i89_best_without_msa.pdb",
            plddt=pred['plddt']
        )
        print(f"Saved: i89_best_without_msa.pdb (pLDDT: {best_without_msa['plddt']:.1f}%)")
        break

# Save ensemble statistics
import json
with open("i89_ensemble_stats.json", "w") as f:
    json.dump({
        'n_structures': len(all_predictions),
        'msa_modes': list(set(r['msa_mode'] for r in ensemble_rmsds)),
        'rmsd_stats': {
            'with_msa': {
                'mean_rmsd_state1': float(np.mean([r['rmsd_state1'] for r in with_msa])),
                'mean_rmsd_state2': float(np.mean([r['rmsd_state2'] for r in with_msa])),
                'mean_plddt': float(np.mean([r['plddt'] for r in with_msa]))
            },
            'without_msa': {
                'mean_rmsd_state1': float(np.mean([r['rmsd_state1'] for r in without_msa])),
                'mean_rmsd_state2': float(np.mean([r['rmsd_state2'] for r in without_msa])),
                'mean_plddt': float(np.mean([r['plddt'] for r in without_msa]))
            }
        },
        'ensemble_diversity': {
            'mean_pairwise_rmsd': float(ensemble_stats['mean_pairwise_rmsd']),
            'max_pairwise_rmsd': float(ensemble_stats['max_pairwise_rmsd'])
        }
    }, f, indent=2)

print("\nSaved ensemble statistics to i89_ensemble_stats.json")


# %% [markdown]
# ## Summary and Key Takeaways
# 
# ### What We've Learned
# 
# 1. **MSA Controls Conformation**: 
#    - With MSA → State 1 preference
#    - Without MSA → State 2 preference
#    - MSA depth can be tuned for intermediate states
# 
# 2. **Recycling Refines Structure**:
#    - Most improvement in first 3-6 recycles
#    - Early stopping saves computation
#    - Convergence can be monitored via RMSD changes
# 
# 3. **Sampling Strategies**:
#    - Dropout introduces stochasticity
#    - Multiple seeds explore conformational space
#    - MSA subsampling provides control
# 
# 4. **Interactive Visualization with LogMD**:
#    - Real-time structure evolution during prediction
#    - Trajectory creation from ensemble predictions
#    - Side-by-side comparison of different conditions
#    - Immediate visual feedback on conformational changes
# 
# 5. **Analysis Methods**:
#    - RMSD for known references
#    - Coevolution reveals functional coupling
#    - Ensemble statistics quantify diversity
# 
# ### Practical Guidelines
# 
# - **For single structure**: Use full MSA, 3-6 recycles
# - **For conformational sampling**: Vary MSA depth, use dropout
# - **For efficiency**: Implement early stopping
# - **For validation**: Compare to known structures when available
# - **For visualization**: Use LogMD to inspect structure evolution and compare ensembles
# 
# ### Next Steps
# 
# Try these techniques on your proteins of interest:
# 1. Proteins with known conformational changes
# 2. Intrinsically disordered regions
# 3. Domain movements
# 4. Oligomeric assemblies
# 
# ### Resources
# 
# - **AF2 Utils Documentation**: See README_tutorial.md
# - **LogMD Utils Documentation**: Interactive 3D visualization tools
# - **ColabDesign**: https://github.com/sokrypton/ColabDesign
# - **AlphaFold**: https://alphafold.ebi.ac.uk/
# - **LogMD**: https://logmd.dev for molecular visualization
# 
# ---
# 
# **Thank you for participating in this tutorial!**
# 

# %% [markdown]
# 


