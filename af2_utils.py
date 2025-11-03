"""
AF2 Utils - AlphaFold2 Utilities Package

A comprehensive wrapper around ColabDesign for AlphaFold2 structure prediction,
including MSA generation, template processing, structure prediction, analysis,
and visualization functions.

This package reimplements functionality from predict.py as a clean, well-organized
module suitable for use in tutorials and research workflows.

Author: Felipe Engelberger
Date: 2025
License: MIT
"""

import os
import sys
import time
import gc
import re
import tempfile
import warnings
from typing import Optional, List, Dict, Any, Tuple, Union
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp

# Version info
__version__ = "1.0.0"
__all__ = [
    # Setup functions
    'setup_environment', 'install_dependencies', 'check_installation',
    # Core utilities
    'clear_memory', 'get_pdb', 'run_mmseqs2_wrapper',
    # MSA functions
    'run_hhalign', 'run_hhfilter', 'get_msa', 'parse_a3m', 'create_single_sequence_msa',
    # Template functions
    'get_template_feats', 'process_templates',
    # Prediction functions
    'setup_model', 'predict_structure', 'predict_with_recycling', 'predict_ensemble',
    # Analysis functions
    'get_coevolution', 'calculate_rmsd', 'analyze_ensemble', 'get_chain_metrics',
    # Visualization functions
    'plot_3d_structure', 'plot_confidence', 'plot_msa', 'plot_coevolution', 'plot_ensemble_analysis',
]

# Global state for checking if environment is setup
_ENVIRONMENT_SETUP = False
_COLABDESIGN_AVAILABLE = False


# =============================================================================
# SECTION 1: SETUP AND INSTALLATION
# =============================================================================

def setup_environment(unified_memory: bool = True, verbose: bool = True) -> None:
    """
    Configure environment for AlphaFold2 predictions.
    
    Sets up JAX memory configuration and adds HHsuite to PATH.
    
    Args:
        unified_memory: Enable unified memory for JAX (recommended for large models)
        verbose: Print setup information
    """
    global _ENVIRONMENT_SETUP
    
    if verbose:
        print("Setting up AF2 environment...")
    
    # Configure JAX memory
    if unified_memory:
        os.environ["TF_FORCE_UNIFIED_MEMORY"] = "1"
        os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "4.0"
        if verbose:
            print("  - Unified memory enabled")
    
    # Add HHsuite to PATH if available
    if os.path.exists("hhsuite/bin") and "hhsuite" not in os.environ.get('PATH', ''):
        os.environ['PATH'] += ":hhsuite/bin:hhsuite/scripts"
        if verbose:
            print("  - HHsuite added to PATH")
    
    _ENVIRONMENT_SETUP = True
    if verbose:
        print("  - Environment setup complete")


def install_dependencies(install_colabdesign: bool = True,
                        install_hhsuite: bool = True,
                        download_params: bool = True,
                        verbose: bool = True) -> None:
    """
    Install required dependencies for AlphaFold2 predictions.
    
    Args:
        install_colabdesign: Install ColabDesign package
        install_hhsuite: Install HH-suite for alignments
        download_params: Download AlphaFold2 parameters
        verbose: Print installation progress
    """
    if verbose:
        print("Installing AF2 dependencies...")
    
    # Install ColabDesign
    if install_colabdesign and not check_colabdesign():
        if verbose:
            print("  - Installing ColabDesign (gamma branch)...")
        os.system("pip -q install git+https://github.com/sokrypton/ColabDesign.git@gamma")
        
        # Try to create symlink for colabdesign
        try:
            os.system("ln -s /usr/local/lib/python3.*/dist-packages/colabdesign colabdesign")
        except:
            pass
        
        # Download colabfold_utils
        if not os.path.exists("colabfold_utils.py"):
            if verbose:
                print("  - Downloading colabfold utilities...")
            os.system("wget -q https://raw.githubusercontent.com/sokrypton/ColabFold/main/colabfold/colabfold.py -O colabfold_utils.py")
    
    # Install HHsuite
    if install_hhsuite and not os.path.exists("hhsuite"):
        if verbose:
            print("  - Installing HH-suite...")
        os.makedirs("hhsuite", exist_ok=True)
        os.system("curl -fsSL https://github.com/soedinglab/hh-suite/releases/download/v3.3.0/hhsuite-3.3.0-SSE2-Linux.tar.gz | tar xz -C hhsuite/")
    
    # Download AlphaFold parameters
    if download_params and not os.path.exists("params/done.txt"):
        if verbose:
            print("  - Downloading AlphaFold2 parameters (this may take a while)...")
        os.makedirs("params", exist_ok=True)
        
        # Use aria2c for faster parallel download
        os.system("apt-get install -qq aria2")
        os.system("aria2c -q -x 16 https://storage.googleapis.com/alphafold/alphafold_params_2022-12-06.tar")
        os.system("tar -xf alphafold_params_2022-12-06.tar -C params")
        os.system("rm alphafold_params_2022-12-06.tar")
        
        # Create marker file
        Path("params/done.txt").touch()
        
        if verbose:
            print("  - AlphaFold2 parameters downloaded")
    
    # Setup environment after installation
    setup_environment(verbose=False)
    
    if verbose:
        print("  - Installation complete!")


def check_colabdesign() -> bool:
    """Check if ColabDesign is available."""
    global _COLABDESIGN_AVAILABLE
    try:
        import colabdesign
        _COLABDESIGN_AVAILABLE = True
        return True
    except ImportError:
        _COLABDESIGN_AVAILABLE = False
        return False


def check_installation(verbose: bool = True) -> Dict[str, bool]:
    """
    Check which components are installed and ready.
    
    Returns:
        Dictionary with installation status of each component
    """
    status = {
        'colabdesign': check_colabdesign(),
        'hhsuite': os.path.exists("hhsuite/bin"),
        'alphafold_params': os.path.exists("params/done.txt"),
        'environment_setup': _ENVIRONMENT_SETUP,
    }
    
    if verbose:
        print("Installation Status:")
        for component, installed in status.items():
            symbol = "✓" if installed else "✗"
            print(f"  {symbol} {component}: {'installed' if installed else 'not installed'}")
    
    return status



# =============================================================================
# SECTION 2: CORE UTILITIES
# =============================================================================

def clear_memory() -> None:
    """Clear JAX memory buffers to free GPU/TPU memory."""
    try:
        backend = jax.lib.xla_bridge.get_backend()
        for buf in backend.live_buffers():
            buf.delete()
    except Exception as e:
        warnings.warn(f"Failed to clear memory: {e}")
    
    # Also run garbage collection
    gc.collect()


def get_pdb(pdb_code: str = "", output_dir: str = "tmp") -> str:
    """
    Download or locate a PDB file.
    
    Args:
        pdb_code: PDB ID (4-letter code), AlphaFold ID, or file path
        output_dir: Directory to save downloaded files
        
    Returns:
        Path to the PDB/CIF file
    """
    # Check if running in Colab
    in_colab = 'google.colab' in sys.modules
    
    if pdb_code is None or pdb_code == "":
        if in_colab:
            # Upload file in Colab
            from google.colab import files
            upload_dict = files.upload()
            pdb_string = upload_dict[list(upload_dict.keys())[0]]
            with open("tmp.pdb", "wb") as out:
                out.write(pdb_string)
            return "tmp.pdb"
        else:
            raise ValueError("pdb_code must be provided when not running in Colab")
    
    elif os.path.isfile(pdb_code):
        return pdb_code
    
    elif len(pdb_code) == 4:
        # Download from RCSB PDB
        os.makedirs(output_dir, exist_ok=True)
        output_path = f"{output_dir}/{pdb_code}.cif"
        if not os.path.exists(output_path):
            os.system(f"wget -qnc https://files.rcsb.org/download/{pdb_code}.cif -P {output_dir}/")
        return output_path
    
    else:
        # Assume it's an AlphaFold ID
        os.makedirs(output_dir, exist_ok=True)
        output_path = f"{output_dir}/AF-{pdb_code}-F1-model_v4.pdb"
        if not os.path.exists(output_path):
            os.system(f"wget -qnc https://alphafold.ebi.ac.uk/files/AF-{pdb_code}-F1-model_v4.pdb -P {output_dir}/")
        return output_path


def run_mmseqs2_wrapper(*args, **kwargs):
    """
    Wrapper for run_mmseqs2 that adds user agent.
    
    This is required by the MMseqs2 API.
    """
    try:
        from colabfold_utils import run_mmseqs2
        kwargs['user_agent'] = "colabdesign/gamma"
        return run_mmseqs2(*args, **kwargs)
    except ImportError:
        raise ImportError("colabfold_utils not found. Please run install_dependencies().")



# =============================================================================
# SECTION 3: MSA FUNCTIONS
# =============================================================================

def run_hhalign(query_sequence: str, 
                target_sequence: str,
                query_a3m: Optional[str] = None,
                target_a3m: Optional[str] = None) -> Tuple[List[str], List[int]]:
    """
    Run HHalign for sequence alignment.
    
    Args:
        query_sequence: Query protein sequence
        target_sequence: Target protein sequence
        query_a3m: Path to query A3M file (optional)
        target_a3m: Path to target A3M file (optional)
        
    Returns:
        Tuple of (aligned_sequences, start_indices)
    """
    from colabdesign.af.contrib import predict
    
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.a3m') as tmp_query, \
         tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.a3m') as tmp_target, \
         tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.hhr') as tmp_alignment:
        
        try:
            if query_a3m is None:
                tmp_query.write(f">Q\n{query_sequence}\n")
                tmp_query.flush()
                query_a3m = tmp_query.name
            
            if target_a3m is None:
                tmp_target.write(f">T\n{target_sequence}\n")
                tmp_target.flush()
                target_a3m = tmp_target.name
            
            # Run hhalign
            os.system(f"hhalign -hide_cons -i {query_a3m} -t {target_a3m} -o {tmp_alignment.name}")
            X, start_indices = predict.parse_hhalign_output(tmp_alignment.name)
            
            return X, start_indices
        finally:
            # Cleanup temp files
            for f in [tmp_query.name, tmp_target.name, tmp_alignment.name]:
                try:
                    os.unlink(f)
                except:
                    pass


def run_do_not_align(query_sequence: str, target_sequence: str, **kwargs) -> Tuple[List[str], List[int]]:
    """
    Skip alignment and return sequences as-is.
    
    Useful when you want to use template structure without alignment.
    """
    return [query_sequence, target_sequence], [0, 0]


def run_hhfilter(input_path: str, output_path: str, id: int = 90, qid: int = 10) -> None:
    """
    Filter MSA sequences using HHfilter.
    
    Args:
        input_path: Input MSA file path
        output_path: Output MSA file path
        id: Maximum pairwise sequence identity (%)
        qid: Minimum coverage with query (%)
    """
    os.system(f"hhfilter -id {id} -qid {qid} -i {input_path} -o {output_path}")


def create_single_sequence_msa(sequence: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create a minimal MSA with just the query sequence.
    
    This is useful for predictions without evolutionary information.
    
    Args:
        sequence: Amino acid sequence
        
    Returns:
        Tuple of (msa, deletion_matrix) as numpy arrays
    """
    from colabdesign.af.contrib import predict
    
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.a3m') as tmp:
        tmp.write(f">query\n{sequence}\n")
        tmp.flush()
        
        msa, deletion_matrix = predict.parse_a3m(tmp.name)
        
        # Cleanup
        try:
            os.unlink(tmp.name)
        except:
            pass
    
    return msa, deletion_matrix


def parse_a3m(a3m_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Parse A3M format MSA file.
    
    Args:
        a3m_path: Path to A3M file
        
    Returns:
        Tuple of (msa, deletion_matrix)
    """
    from colabdesign.af.contrib import predict
    return predict.parse_a3m(a3m_path)


def get_msa(sequences: Union[str, List[str]],
            jobname: str,
            mode: str = "unpaired",
            cov: int = 50,
            id: int = 90,
            qid: int = 0,
            max_msa: int = 4096,
            do_not_filter: bool = False,
            verbose: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate MSA using MMseqs2.
    
    Args:
        sequences: Protein sequence(s)
        jobname: Job name for output directory
        mode: MSA pairing mode ("unpaired", "paired", "unpaired_paired")
        cov: Minimum coverage (%)
        id: Maximum sequence identity (%)
        qid: Minimum sequence identity with query (%)
        max_msa: Maximum number of MSA sequences
        do_not_filter: Skip filtering step
        verbose: Print progress information
        
    Returns:
        Tuple of (msa, deletion_matrix)
    """
    from colabdesign.af.contrib import predict
    
    # Ensure sequences is a list
    if isinstance(sequences, str):
        sequences = [sequences]
    
    # Create output directory
    os.makedirs(jobname, exist_ok=True)
    
    if verbose:
        print(f"Generating MSA for {len(sequences)} sequence(s)...")
    
    try:
        msa, deletion_matrix = predict.get_msa(
            sequences, 
            jobname,
            mode=mode,
            cov=cov, 
            id=id, 
            qid=qid, 
            max_msa=max_msa,
            do_not_filter=do_not_filter,
            mmseqs2_fn=run_mmseqs2_wrapper,
            hhfilter_fn=run_hhfilter
        )
        
        if verbose:
            print(f"  - Generated MSA with {len(msa)} sequences")
        
        return msa, deletion_matrix
        
    except Exception as e:
        if verbose:
            print(f"  - MSA generation failed: {e}")
            print("  - Falling back to single sequence mode")
        return create_single_sequence_msa("".join(sequences))



# =============================================================================
# SECTION 4: TEMPLATE FUNCTIONS
# =============================================================================

def get_template_feats(pdb: Union[str, List[str]],
                      chain: Union[str, List[str]],
                      query_seq: str,
                      query_a3m: Optional[str] = None,
                      copies: int = 1,
                      propagate_to_copies: bool = True,
                      use_seq: bool = True,
                      do_not_align: bool = False) -> Dict[str, Any]:
    """
    Get template features from PDB structure(s).
    
    Args:
        pdb: PDB ID(s) or file path(s)
        chain: Chain ID(s)
        query_seq: Query sequence
        query_a3m: Path to query A3M file
        copies: Number of copies for oligomers
        propagate_to_copies: Propagate template to all copies
        use_seq: Use template sequence information
        do_not_align: Skip alignment step
        
    Returns:
        Dictionary of template features
    """
    from colabdesign.af.contrib import predict
    
    align_fn = run_do_not_align if do_not_align else run_hhalign
    
    batch = predict.get_template_feats(
        pdb, chain,
        query_seq=query_seq,
        query_a3m=query_a3m,
        copies=copies,
        propagate_to_copies=propagate_to_copies,
        use_seq=use_seq,
        get_pdb_fn=get_pdb,
        align_fn=align_fn
    )
    
    return batch



# =============================================================================
# SECTION 5: PREDICTION FUNCTIONS
# =============================================================================

def setup_model(sequence: Union[str, List[str]],
                copies: int = 1,
                model_type: str = "auto",
                num_msa: int = 512,
                num_extra_msa: int = 1024,
                use_cluster_profile: bool = True,
                use_templates: bool = False,
                num_templates: int = 0,
                debug: bool = False,
                verbose: bool = True) -> Any:
    """
    Initialize AlphaFold2 model for prediction.
    
    Args:
        sequence: Amino acid sequence(s)
        copies: Number of copies for oligomers
        model_type: Model type ("alphafold2_ptm", "alphafold2_multimer_v3", "auto")
        num_msa: Number of MSA sequences to use
        num_extra_msa: Number of extra MSA sequences
        use_cluster_profile: Use MSA cluster profiles
        use_templates: Use template structures
        num_templates: Number of templates
        debug: Enable debug mode
        verbose: Print setup information
        
    Returns:
        Initialized ColabDesign AF model
    """
    from colabdesign import mk_af_model
    
    # Ensure sequence is a list
    if isinstance(sequence, str):
        sequences = [sequence]
    else:
        sequences = sequence
    
    # Calculate lengths
    lengths = [len(s) for s in sequences]
    
    # Auto-select model type
    if model_type == "auto":
        if len(lengths) > 1 or copies > 1:
            model_type = "alphafold2_multimer_v3"
        else:
            model_type = "alphafold2_ptm"
    
    if verbose:
        print(f"Setting up {model_type} model...")
        print(f"  - Sequence length(s): {lengths}")
        print(f"  - Copies: {copies}")
    
    # Model options
    model_opts = {
        "num_msa": num_msa,
        "num_extra_msa": num_extra_msa,
        "num_templates": num_templates,
        "use_cluster_profile": use_cluster_profile,
        "model_type": model_type,
        "use_templates": use_templates,
        "protocol": "hallucination",
        "best_metric": "plddt",
        "optimize_seq": False,
        "debug": debug,
    }
    
    # Initialize model
    model = mk_af_model(**model_opts)
    
    # Prepare inputs
    model.prep_inputs(lengths, copies=copies, seed=0)
    
    if verbose:
        print("  - Model initialized successfully")
    
    return model


def predict_structure(model: Any,
                     msa: Optional[np.ndarray] = None,
                     deletion_matrix: Optional[np.ndarray] = None,
                     num_recycles: int = 3,
                     use_dropout: bool = False,
                     seed: int = 0,
                     verbose: bool = True) -> Dict[str, Any]:
    """
    Run structure prediction with AlphaFold2.
    
    Args:
        model: Initialized AF2 model
        msa: MSA array (if None, uses model's existing MSA)
        deletion_matrix: Deletion matrix for MSA
        num_recycles: Number of recycling iterations
        use_dropout: Enable dropout for stochastic predictions
        seed: Random seed
        verbose: Print prediction progress
        
    Returns:
        Dictionary with prediction results
    """
    from colabdesign.shared.protein import _np_rmsd
    
    # Set MSA if provided
    if msa is not None:
        model.set_msa(msa, deletion_matrix)
    
    # Set seed
    model.set_seed(seed)
    
    # Set recycling
    model.set_opt(num_recycles=num_recycles)
    
    if verbose:
        print(f"Running prediction (recycles={num_recycles}, dropout={use_dropout}, seed={seed})...")
    
    # Run prediction
    model.predict(dropout=use_dropout, verbose=False)
    
    # Extract results
    results = {
        'structure': model.aux['atom_positions'],
        'plddt': model.aux['plddt'],
        'pae': model.aux.get('pae', None),
        'ptm': model.aux.get('ptm', 0.0),
        'metrics': {
            'plddt': model.aux['plddt'].mean(),
            'ptm': model.aux.get('ptm', 0.0),
        }
    }
    
    if verbose:
        print(f"  - Mean pLDDT: {results['metrics']['plddt']:.3f}")
        if results['pae'] is not None:
            print(f"  - pTM: {results['metrics']['ptm']:.3f}")
    
    return results


def predict_with_recycling(model: Any,
                           msa: Optional[np.ndarray] = None,
                           deletion_matrix: Optional[np.ndarray] = None,
                           max_recycles: int = 6,
                           early_stop_tolerance: float = 0.5,
                           seed: int = 0,
                           verbose: bool = True) -> Dict[str, Any]:
    """
    Run prediction with early stopping based on RMSD convergence.
    
    Args:
        model: Initialized AF2 model
        msa: MSA array
        deletion_matrix: Deletion matrix
        max_recycles: Maximum number of recycles
        early_stop_tolerance: RMSD threshold for early stopping (Angstroms)
        seed: Random seed
        verbose: Print progress
        
    Returns:
        Dictionary with prediction results and convergence info
    """
    from colabdesign.shared.protein import _np_rmsd
    
    # Set MSA if provided
    if msa is not None:
        model.set_msa(msa, deletion_matrix)
    
    model.set_seed(seed)
    
    results_per_recycle = []
    prev_pos = None
    
    for recycle in range(max_recycles + 1):
        model.set_opt(num_recycles=recycle)
        model._inputs.pop("prev", None)  # Reset previous state
        
        # Run prediction
        model.predict(dropout=False, verbose=False)
        
        # Get current positions
        current_pos = model.aux['atom_positions'][:, 1]  # CA atoms
        
        # Calculate RMSD if not first recycle
        rmsd_change = None
        if prev_pos is not None:
            rmsd_change = _np_rmsd(prev_pos, current_pos, use_jax=False)
        
        # Store results
        result = {
            'recycle': recycle,
            'structure': model.aux['atom_positions'].copy(),
            'plddt': model.aux['plddt'].copy(),
            'pae': model.aux.get('pae', None),
            'rmsd_change': rmsd_change,
            'metrics': {
                'plddt': model.aux['plddt'].mean(),
                'ptm': model.aux.get('ptm', 0.0),
            }
        }
        results_per_recycle.append(result)
        
        if verbose:
            log_str = f"  Recycle {recycle}: pLDDT={result['metrics']['plddt']:.3f}"
            if rmsd_change is not None:
                log_str += f", RMSD change={rmsd_change:.3f}Å"
            print(log_str)
        
        # Early stopping check
        if rmsd_change is not None and rmsd_change < early_stop_tolerance:
            if verbose:
                print(f"  - Early stop at recycle {recycle} (RMSD < {early_stop_tolerance}Å)")
            break
        
        prev_pos = current_pos
    
    # Return the last (best) result with full trajectory
    final_result = results_per_recycle[-1]
    final_result['trajectory'] = results_per_recycle
    
    return final_result


def predict_ensemble(model: Any,
                    msa: Optional[np.ndarray] = None,
                    deletion_matrix: Optional[np.ndarray] = None,
                    num_seeds: int = 5,
                    num_recycles: int = 3,
                    use_dropout: bool = True,
                    seed_start: int = 0,
                    verbose: bool = True) -> List[Dict[str, Any]]:
    """
    Generate ensemble of predictions with different seeds/dropout.
    
    Args:
        model: Initialized AF2 model
        msa: MSA array
        deletion_matrix: Deletion matrix
        num_seeds: Number of different random seeds
        num_recycles: Number of recycles per prediction
        use_dropout: Enable dropout for diversity
        seed_start: Starting seed value
        verbose: Print progress
        
    Returns:
        List of prediction results
    """
    ensemble = []
    
    if verbose:
        print(f"Generating ensemble with {num_seeds} seeds...")
    
    for i in range(num_seeds):
        seed = seed_start + i
        
        result = predict_structure(
            model,
            msa=msa,
            deletion_matrix=deletion_matrix,
            num_recycles=num_recycles,
            use_dropout=use_dropout,
            seed=seed,
            verbose=False
        )
        
        result['seed'] = seed
        result['ensemble_idx'] = i
        ensemble.append(result)
        
        if verbose:
            print(f"  Seed {seed}: pLDDT={result['metrics']['plddt']:.3f}")
    
    if verbose:
        mean_plddt = np.mean([r['metrics']['plddt'] for r in ensemble])
        print(f"  - Ensemble mean pLDDT: {mean_plddt:.3f}")
    
    return ensemble



# =============================================================================
# SECTION 6: ANALYSIS FUNCTIONS
# =============================================================================

@jax.jit
def get_coevolution(msa: np.ndarray) -> np.ndarray:
    """
    Compute coevolution matrix from MSA using direct coupling analysis.
    
    Args:
        msa: MSA array with shape (N_sequences, L_residues)
        
    Returns:
        Coevolution matrix with shape (L, L)
    """
    # Convert to one-hot encoding
    Y = jax.nn.one_hot(msa, 22)
    N, L, A = Y.shape
    Y_flat = Y.reshape(N, -1)
    
    # Compute covariance
    c = jnp.cov(Y_flat.T)
    
    # Inverse covariance with regularization
    shrink = 4.5 / jnp.sqrt(N) * jnp.eye(c.shape[0])
    ic = jnp.linalg.inv(c + shrink)
    
    # Partial correlation coefficient
    ic_diag = jnp.diag(ic)
    pcc = ic / jnp.sqrt(ic_diag[:, None] * ic_diag[None, :])
    
    # Frobenius norm
    raw = jnp.sqrt(jnp.square(pcc.reshape(L, A, L, A)[:, :20, :, :20]).sum((1, 3)))
    
    # Zero out diagonal
    i = jnp.arange(L)
    raw = raw.at[i, i].set(0)
    
    # Average product correction (APC)
    ap = raw.sum(0, keepdims=True) * raw.sum(1, keepdims=True) / raw.sum()
    coev = (raw - ap).at[i, i].set(0)
    
    return np.array(coev)


def calculate_rmsd(coords1: np.ndarray, 
                  coords2: np.ndarray,
                  align: bool = True) -> float:
    """
    Calculate RMSD between two coordinate sets.
    
    Args:
        coords1: First coordinate set (N, 3)
        coords2: Second coordinate set (N, 3)
        align: Perform Kabsch alignment before RMSD calculation
        
    Returns:
        RMSD value in Angstroms
    """
    from colabdesign.shared.protein import _np_rmsd
    return _np_rmsd(coords1, coords2, use_jax=False)


def analyze_ensemble(structures: List[np.ndarray],
                    reference_structure: Optional[np.ndarray] = None,
                    verbose: bool = True) -> Dict[str, Any]:
    """
    Analyze structural ensemble diversity.
    
    Args:
        structures: List of structure arrays
        reference_structure: Optional reference for RMSD calculation
        verbose: Print analysis results
        
    Returns:
        Dictionary with ensemble statistics
    """
    from colabdesign.shared.protein import _np_rmsd
    
    n_structures = len(structures)
    
    # Extract CA coordinates
    ca_coords = [s[:, 1, :] for s in structures]
    
    # Calculate pairwise RMSDs
    pairwise_rmsds = []
    for i in range(n_structures):
        for j in range(i + 1, n_structures):
            rmsd = _np_rmsd(ca_coords[i], ca_coords[j], use_jax=False)
            pairwise_rmsds.append(rmsd)
    
    # Statistics
    stats = {
        'n_structures': n_structures,
        'mean_pairwise_rmsd': np.mean(pairwise_rmsds) if pairwise_rmsds else 0.0,
        'std_pairwise_rmsd': np.std(pairwise_rmsds) if pairwise_rmsds else 0.0,
        'max_pairwise_rmsd': np.max(pairwise_rmsds) if pairwise_rmsds else 0.0,
        'min_pairwise_rmsd': np.min(pairwise_rmsds) if pairwise_rmsds else 0.0,
    }
    
    # Reference RMSDs if provided
    if reference_structure is not None:
        ref_ca = reference_structure[:, 1, :]
        ref_rmsds = [_np_rmsd(ca, ref_ca, use_jax=False) for ca in ca_coords]
        stats['ref_rmsds'] = ref_rmsds
        stats['mean_ref_rmsd'] = np.mean(ref_rmsds)
        stats['std_ref_rmsd'] = np.std(ref_rmsds)
    
    if verbose:
        print("Ensemble Analysis:")
        print(f"  - Number of structures: {n_structures}")
        print(f"  - Mean pairwise RMSD: {stats['mean_pairwise_rmsd']:.2f} ± {stats['std_pairwise_rmsd']:.2f} Å")
        print(f"  - RMSD range: {stats['min_pairwise_rmsd']:.2f} - {stats['max_pairwise_rmsd']:.2f} Å")
        if reference_structure is not None:
            print(f"  - Mean RMSD to reference: {stats['mean_ref_rmsd']:.2f} ± {stats['std_ref_rmsd']:.2f} Å")
    
    return stats


def get_chain_metrics(outputs: Dict[str, Any], asym_id: np.ndarray) -> Dict[str, Any]:
    """
    Calculate chain and interface pTM metrics for multimers.
    
    Args:
        outputs: Model output dictionary (from debug mode)
        asym_id: Asymmetric unit IDs
        
    Returns:
        Dictionary with pairwise metrics
    """
    try:
        import colabdesign.af.contrib.extended_ptm as extended_ptm
        return extended_ptm.get_chain_and_interface_metrics(outputs, asym_id)
    except Exception as e:
        warnings.warn(f"Failed to calculate extended metrics: {e}")
        return {}



# =============================================================================
# SECTION 7: VISUALIZATION FUNCTIONS
# =============================================================================

def plot_3d_structure(atom_positions: np.ndarray,
                     plddt: np.ndarray,
                     lengths: Optional[List[int]] = None,
                     save_path: Optional[str] = None,
                     show: bool = True) -> plt.Figure:
    """
    Create 3D visualization of predicted structure.
    
    Args:
        atom_positions: Atom coordinates (L, 37, 3)
        plddt: Per-residue pLDDT scores
        lengths: Chain lengths for multi-chain structures
        save_path: Path to save figure
        show: Display figure
        
    Returns:
        Matplotlib figure
    """
    from colabdesign.shared.protein import _np_kabsch
    from colabdesign.shared.plot import plot_pseudo_3D, pymol_cmap
    
    if lengths is None:
        lengths = [len(plddt)]
    
    fig = plt.figure(figsize=(10, 5))
    
    # Extract CA coordinates
    xyz = atom_positions[:, 1]
    xyz = xyz @ _np_kabsch(xyz, xyz, return_v=True, use_jax=False)
    
    # Plot 1: Chain coloring (if multimer)
    ax = plt.subplot(1, 2, 1)
    if len(lengths) > 1:
        plt.title("Chain")
        c = np.concatenate([[n] * L for n, L in enumerate(lengths)])
        plot_pseudo_3D(xyz=xyz, c=c, cmap=pymol_cmap, cmin=0, cmax=39, Ls=lengths, ax=ax)
    else:
        plt.title("Length")
        plot_pseudo_3D(xyz=xyz, Ls=lengths, ax=ax)
    plt.axis(False)
    
    # Plot 2: pLDDT coloring
    ax = plt.subplot(1, 2, 2)
    plt.title("pLDDT")
    plot_pseudo_3D(xyz=xyz, c=plddt, cmin=0.5, cmax=0.9, Ls=lengths, ax=ax)
    plt.axis(False)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig


def plot_confidence(plddt: np.ndarray,
                   pae: Optional[np.ndarray] = None,
                   lengths: Optional[List[int]] = None,
                   save_path: Optional[str] = None,
                   show: bool = True) -> plt.Figure:
    """
    Plot confidence metrics (pLDDT and optionally PAE).
    
    Args:
        plddt: Per-residue pLDDT scores (0-1 or 0-100)
        pae: Predicted aligned error matrix (optional)
        lengths: Chain lengths for tick marks
        save_path: Path to save figure
        show: Display figure
        
    Returns:
        Matplotlib figure
    """
    from colabdesign.af.contrib import predict
    
    # Convert pLDDT to 0-100 scale if needed
    if plddt.max() <= 1.0:
        plddt = plddt * 100
    
    fig = predict.plot_confidence(plddt, pae, lengths)
    
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig


def plot_msa(msa: np.ndarray,
            lengths: Optional[List[int]] = None,
            save_path: Optional[str] = None,
            show: bool = True) -> plt.Figure:
    """
    Visualize MSA coverage and statistics.
    
    Args:
        msa: MSA array
        lengths: Sequence lengths for tick marks
        save_path: Path to save figure
        show: Display figure
        
    Returns:
        Matplotlib figure
    """
    from colabdesign.af.contrib import predict
    
    fig = predict.plot_msa(msa, lengths)
    
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig


def plot_coevolution(msa: np.ndarray,
                    contact_map: Optional[np.ndarray] = None,
                    save_path: Optional[str] = None,
                    show: bool = True) -> plt.Figure:
    """
    Plot coevolution matrix from MSA.
    
    Args:
        msa: MSA array
        contact_map: Optional contact map for overlay
        save_path: Path to save figure
        show: Display figure
        
    Returns:
        Matplotlib figure
    """
    coev = get_coevolution(msa)
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Plot coevolution
    im = ax.imshow(coev, cmap='RdBu_r', vmin=-np.max(np.abs(coev)), vmax=np.max(np.abs(coev)))
    ax.set_xlabel('Position')
    ax.set_ylabel('Position')
    ax.set_title('Coevolution Matrix')
    plt.colorbar(im, ax=ax, label='Coevolution Score')
    
    # Overlay contact map if provided
    if contact_map is not None:
        L = contact_map.shape[0]
        i, j = np.triu_indices(L, 6)
        contacts = contact_map[i, j] > 0.5
        ax.scatter(j[contacts], i[contacts], c='green', s=1, alpha=0.3, label='Contacts')
        ax.scatter(i[contacts], j[contacts], c='green', s=1, alpha=0.3)
        ax.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig


def plot_ensemble_analysis(structures: List[np.ndarray],
                          labels: Optional[List[str]] = None,
                          reference_structures: Optional[List[np.ndarray]] = None,
                          reference_labels: Optional[List[str]] = None,
                          save_path: Optional[str] = None,
                          show: bool = True) -> plt.Figure:
    """
    Create comprehensive visualization of ensemble diversity.
    
    Args:
        structures: List of predicted structures
        labels: Labels for each structure
        reference_structures: Optional reference structures for comparison
        reference_labels: Labels for reference structures
        save_path: Path to save figure
        show: Display figure
        
    Returns:
        Matplotlib figure
    """
    from colabdesign.shared.protein import _np_rmsd
    
    n_structures = len(structures)
    
    # Extract CA coordinates
    ca_coords = [s[:, 1, :] for s in structures]
    
    # Calculate pairwise RMSD matrix
    rmsd_matrix = np.zeros((n_structures, n_structures))
    for i in range(n_structures):
        for j in range(n_structures):
            if i != j:
                rmsd_matrix[i, j] = _np_rmsd(ca_coords[i], ca_coords[j], use_jax=False)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: RMSD matrix
    ax = axes[0]
    im = ax.imshow(rmsd_matrix, cmap='viridis')
    ax.set_xlabel('Structure Index')
    ax.set_ylabel('Structure Index')
    ax.set_title('Pairwise RMSD Matrix (Å)')
    plt.colorbar(im, ax=ax)
    
    # Plot 2: RMSD distribution
    ax = axes[1]
    upper_tri = rmsd_matrix[np.triu_indices(n_structures, k=1)]
    ax.hist(upper_tri, bins=20, alpha=0.7, edgecolor='black')
    ax.set_xlabel('RMSD (Å)')
    ax.set_ylabel('Count')
    ax.set_title('Pairwise RMSD Distribution')
    ax.axvline(np.mean(upper_tri), color='red', linestyle='--', label=f'Mean: {np.mean(upper_tri):.2f}Å')
    ax.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig



# =============================================================================
# SECTION 8: HIGH-LEVEL API
# =============================================================================

def quick_predict(sequence: str,
                 msa_mode: str = "mmseqs2",
                 num_recycles: int = 3,
                 jobname: str = "prediction",
                 verbose: bool = True) -> Dict[str, Any]:
    """
    Simple one-line prediction interface.
    
    Args:
        sequence: Amino acid sequence
        msa_mode: MSA generation mode ("mmseqs2" or "single_sequence")
        num_recycles: Number of recycling iterations
        jobname: Job name for output files
        verbose: Print progress information
        
    Returns:
        Dictionary with prediction results
    """
    # Check and setup environment
    if not check_colabdesign():
        if verbose:
            print("ColabDesign not found. Installing dependencies...")
        install_dependencies(verbose=verbose)
    
    # Generate MSA
    if msa_mode == "mmseqs2":
        msa, deletion_matrix = get_msa([sequence], jobname, verbose=verbose)
    else:
        msa, deletion_matrix = create_single_sequence_msa(sequence)
    
    # Setup model
    model = setup_model(sequence, verbose=verbose)
    
    # Run prediction
    results = predict_structure(
        model,
        msa=msa,
        deletion_matrix=deletion_matrix,
        num_recycles=num_recycles,
        verbose=verbose
    )
    
    return results


def predict_conformational_ensemble(sequence: str,
                                   msa_modes: List[str] = ["mmseqs2", "single_sequence"],
                                   num_seeds: int = 3,
                                   num_recycles: int = 3,
                                   use_dropout: bool = True,
                                   jobname: str = "ensemble",
                                   verbose: bool = True) -> List[Dict[str, Any]]:
    """
    Generate conformational ensemble with different MSA conditions.
    
    Args:
        sequence: Amino acid sequence
        msa_modes: List of MSA modes to try
        num_seeds: Seeds per MSA mode
        num_recycles: Recycling iterations
        use_dropout: Enable dropout for diversity
        jobname: Job name
        verbose: Print progress
        
    Returns:
        List of all predictions
    """
    all_predictions = []
    
    for msa_mode in msa_modes:
        if verbose:
            print(f"\n{'='*60}")
            print(f"Generating predictions with MSA mode: {msa_mode}")
            print(f"{'='*60}")
        
        # Generate MSA
        if msa_mode == "mmseqs2":
            msa, deletion_matrix = get_msa([sequence], f"{jobname}_{msa_mode}", verbose=verbose)
        else:
            msa, deletion_matrix = create_single_sequence_msa(sequence)
        
        # Setup model
        model = setup_model(sequence, verbose=False)
        
        # Generate ensemble
        predictions = predict_ensemble(
            model,
            msa=msa,
            deletion_matrix=deletion_matrix,
            num_seeds=num_seeds,
            num_recycles=num_recycles,
            use_dropout=use_dropout,
            verbose=verbose
        )
        
        # Add MSA mode to each prediction
        for pred in predictions:
            pred['msa_mode'] = msa_mode
        
        all_predictions.extend(predictions)
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Total predictions generated: {len(all_predictions)}")
        print(f"{'='*60}")
    
    return all_predictions


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def save_pdb(atom_positions: np.ndarray,
            sequence: str,
            output_path: str,
            plddt: Optional[np.ndarray] = None) -> None:
    """
    Save structure to PDB file.
    
    Args:
        atom_positions: Atom coordinates (L, 37, 3)
        sequence: Amino acid sequence
        output_path: Output PDB file path
        plddt: Optional per-residue pLDDT scores
    """
    from Bio.PDB import PDBIO, Structure, Model, Chain, Residue, Atom
    
    # Create structure
    structure = Structure.Structure("prediction")
    model = Model.Model(0)
    chain = Chain.Chain("A")
    
    # Amino acid mapping
    aa_map = {
        'A': 'ALA', 'C': 'CYS', 'D': 'ASP', 'E': 'GLU', 'F': 'PHE',
        'G': 'GLY', 'H': 'HIS', 'I': 'ILE', 'K': 'LYS', 'L': 'LEU',
        'M': 'MET', 'N': 'ASN', 'P': 'PRO', 'Q': 'GLN', 'R': 'ARG',
        'S': 'SER', 'T': 'THR', 'V': 'VAL', 'W': 'TRP', 'Y': 'TYR'
    }
    
    atom_names = ['N', 'CA', 'C', 'O']  # Simplified backbone
    
    for res_idx, aa in enumerate(sequence):
        resname = aa_map.get(aa, 'UNK')
        residue = Residue.Residue((' ', res_idx + 1, ' '), resname, '')
        
        # Add CA atom (index 1 in atom_positions)
        ca_coord = atom_positions[res_idx, 1, :]
        ca_atom = Atom.Atom('CA', ca_coord, 1.0, 1.0 if plddt is None else plddt[res_idx], 
                           ' ', 'CA', 0, 'C')
        residue.add(ca_atom)
        
        chain.add(residue)
    
    model.add(chain)
    structure.add(model)
    
    # Save
    io = PDBIO()
    io.set_structure(structure)
    io.save(output_path)


def load_pdb_coords(pdb_path: str) -> np.ndarray:
    """
    Load CA coordinates from PDB file.
    
    Args:
        pdb_path: Path to PDB file
        
    Returns:
        CA coordinates array (N, 3)
    """
    from Bio import PDB
    
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_path)
    
    coords = []
    for model in structure:
        for chain in model:
            for residue in chain:
                if 'CA' in residue:
                    coords.append(residue['CA'].coord)
    
    return np.array(coords)


# =============================================================================
# MODULE INITIALIZATION
# =============================================================================

def _try_import_colabdesign():
    """Try to import ColabDesign components."""
    global _COLABDESIGN_AVAILABLE
    try:
        from colabdesign import mk_af_model, clear_mem
        from colabdesign.af.contrib import predict
        from colabdesign.shared.protein import _np_rmsd, _np_kabsch
        from colabdesign.shared.plot import plot_pseudo_3D, pymol_cmap
        _COLABDESIGN_AVAILABLE = True
        return True
    except ImportError:
        _COLABDESIGN_AVAILABLE = False
        return False


# Try to import ColabDesign on module load
_try_import_colabdesign()

# Print module info
if __name__ != "__main__":
    print(f"AF2 Utils v{__version__} loaded")
    if _COLABDESIGN_AVAILABLE:
        print("  - ColabDesign: available")
    else:
        print("  - ColabDesign: not found (run install_dependencies())")


