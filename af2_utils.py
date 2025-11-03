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
import logging
import hashlib
import json
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple, Union, Sequence
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp

# Configure logging for MSA utilities
logger = logging.getLogger(__name__)

# Version info
__version__ = "1.0.0"
__all__ = [
    # Setup functions
    'setup_environment', 'install_dependencies', 'check_installation',
    # Core utilities
    'clear_memory', 'get_pdb', 'run_mmseqs2_wrapper', 'get_hash', 'create_job_folder',
    # MSA functions
    'run_hhalign', 'run_hhfilter', 'get_msa', 'parse_a3m', 'create_single_sequence_msa',
    # MSA Visualization classes
    'MSAData', 'MSACoevolutionVisualizer',
    # Template functions
    'get_template_feats', 'process_templates',
    # Prediction functions
    'setup_model', 'predict_structure', 'predict_with_recycling', 'predict_ensemble',
    'predict_with_logmd',
    # Analysis functions
    'get_coevolution', 'calculate_rmsd', 'analyze_ensemble', 'get_chain_metrics',
    # GPU-accelerated RMSD functions
    'calculate_batch_rmsd_gpu', 'calculate_batch_rmsd_to_references', 'calculate_all_vs_all_rmsd',
    # Visualization functions
    'plot_3d_structure', 'plot_confidence', 'plot_msa', 'plot_coevolution', 'plot_ensemble_analysis',
    'plot_msa_interactive', 'plot_coevolution_interactive', 'compare_coevolution_conditions',
    # LogMD functions
    'check_logmd', 'create_trajectory_from_ensemble', 'create_recycle_trajectory', 'save_pdb_string',
]

# Global state for checking if environment is setup
_ENVIRONMENT_SETUP = False
_COLABDESIGN_AVAILABLE = False
_LOGMD_AVAILABLE = False
_LOGMD_INTEGRATION = None


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
                        install_logmd: bool = True,
                        verbose: bool = True) -> None:
    """
    Install required dependencies for AlphaFold2 predictions.
    
    Args:
        install_colabdesign: Install ColabDesign package
        install_hhsuite: Install HH-suite for alignments
        download_params: Download AlphaFold2 parameters
        install_logmd: Install LogMD for interactive visualization
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
    
    # Install LogMD
    if install_logmd and not check_logmd():
        if verbose:
            print("  - Installing LogMD for interactive visualization...")
        os.system("pip -q install logmd")
        
        # Check if installation succeeded
        if check_logmd():
            if verbose:
                print("  - LogMD installed successfully")
    
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
        'logmd': check_logmd(),
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


def check_logmd() -> bool:
    """
    Check if LogMD is available for visualization.
    
    Returns:
        True if LogMD is available, False otherwise
    """
    global _LOGMD_AVAILABLE, _LOGMD_INTEGRATION
    
    try:
        import logmd_utils
        _LOGMD_INTEGRATION = logmd_utils.LogMDIntegration()
        _LOGMD_AVAILABLE = _LOGMD_INTEGRATION.is_available()
        return _LOGMD_AVAILABLE
    except ImportError:
        _LOGMD_AVAILABLE = False
        return False



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


def get_hash(sequence: str, length: int = 5) -> str:
    """
    Generate hash of sequence for folder naming (uses ColabDesign's predict.get_hash).
    
    Args:
        sequence: Protein sequence
        length: Length of hash to return
        
    Returns:
        Truncated hash of sequence
    """
    try:
        from colabdesign.af.contrib import predict
        return predict.get_hash(sequence)[:length]
    except ImportError:
        # Fallback if ColabDesign not available
        return hashlib.md5(sequence.encode()).hexdigest()[:length]


def create_job_folder(sequence: str, jobname: str = "prediction") -> str:
    """
    Create folder with sequence hash for organizing predictions.
    
    This follows the same pattern as predict.py to ensure workshop attendees
    can easily identify their predictions by sequence.
    
    Args:
        sequence: Protein sequence
        jobname: Base name for the job
        
    Returns:
        Path to created folder
        
    Example:
        >>> folder = create_job_folder("MKTAY...", "my_protein")
        >>> # Creates folder like: my_protein_a3f2b/
        >>> #   with subdirectories: pdb/ and pdb/recycles/
    """
    seq_hash = get_hash(sequence)
    folder_name = f"{jobname}_{seq_hash}"
    
    # Handle existing folders by appending a number
    if os.path.exists(folder_name):
        n = 0
        while os.path.exists(f"{folder_name}_{n}"):
            n += 1
        folder_name = f"{folder_name}_{n}"
    
    # Create folder structure
    os.makedirs(folder_name, exist_ok=True)
    os.makedirs(f"{folder_name}/pdb", exist_ok=True)
    os.makedirs(f"{folder_name}/pdb/recycles", exist_ok=True)
    
    return folder_name



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
                           save_pdbs: bool = False,
                           job_folder: Optional[str] = None,
                           sequence: Optional[str] = None,
                           model_name: str = "model",
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
        save_pdbs: Save PDB file for each recycle (default: False)
        job_folder: Folder to save PDBs (required if save_pdbs=True)
        sequence: Protein sequence (required if save_pdbs=True)
        model_name: Model name for PDB filenames (default: "model")
        verbose: Print progress
        
    Returns:
        Dictionary with prediction results and convergence info
    """
    from colabdesign.shared.protein import _np_rmsd
    
    # Validate PDB saving parameters
    if save_pdbs and (job_folder is None or sequence is None):
        raise ValueError("job_folder and sequence are required when save_pdbs=True")
    
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
        
        # Save PDB for this recycle
        if save_pdbs:
            pdb_path = f"{job_folder}/pdb/recycles/{model_name}_r{recycle}_seed{seed}.pdb"
            save_pdb(
                atom_positions=model.aux['atom_positions'],
                sequence=sequence,
                output_path=pdb_path,
                plddt=model.aux['plddt']
            )
        
        if verbose:
            log_str = f"  Recycle {recycle}: pLDDT={result['metrics']['plddt']:.3f}"
            if rmsd_change is not None:
                log_str += f", RMSD change={rmsd_change:.3f}Å"
            if save_pdbs:
                log_str += f" [PDB saved]"
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
    if save_pdbs:
        final_result['job_folder'] = job_folder
    
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

# GPU-Accelerated RMSD Functions (from AlphaMask)
# These functions use JAX for GPU acceleration and parallel processing

# JAX-compiled Kabsch algorithm for optimal superposition
@jax.jit
def _kabsch_jax(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    """
    Kabsch algorithm using JAX for GPU acceleration.
    
    Args:
        a: First coordinate set [N, 3]
        b: Second coordinate set [N, 3]
    
    Returns:
        Rotation matrix [3, 3]
    """
    u, s, vh = jnp.linalg.svd(a.T @ b, full_matrices=False)
    u = jnp.where(jnp.linalg.det(u @ vh) < 0, u.at[:,-1].set(-u[:,-1]), u)
    return u @ vh

# JAX-compiled RMSD calculation with alignment
@jax.jit
def _rmsd_jax(true: jnp.ndarray, pred: jnp.ndarray) -> float:
    """
    Calculate RMSD using JAX with GPU acceleration.
    
    Performs automatic alignment using Kabsch algorithm and computes RMSD.
    
    Args:
        true: Reference coordinates [N, 3]
        pred: Predicted coordinates [N, 3]
    
    Returns:
        RMSD value (Angstroms)
    """
    # Center coordinates
    p = true - true.mean(0, keepdims=True)
    q = pred - pred.mean(0, keepdims=True)
    
    # Align pred to true using Kabsch
    p = p @ _kabsch_jax(p, q)
    
    # Calculate RMSD
    return jnp.sqrt(jnp.square(p-q).sum(-1).mean())

# Vectorized RMSD calculation for parallel processing
_rmsd_parallel_jax = jax.jit(jax.vmap(_rmsd_jax, (None, 0)))

# All-vs-all RMSD matrix computation
@jax.jit
def _pairwise_rmsd_matrix_jax(coords: jnp.ndarray) -> jnp.ndarray:
    """
    Compute an N×N RMSD matrix for a batch of structures using GPU.
    
    Args:
        coords: Array of shape [N, P, 3], where N is number of structures,
                P is number of atoms per structure.
    
    Returns:
        rmsd_matrix: Array of shape [N, N], where element [i, j] is RMSD between
                     coords[i] and coords[j].
    
    Example:
        >>> ca_coords = np.stack([pred[:, 1, :] for pred in predictions])
        >>> rmsd_matrix = _pairwise_rmsd_matrix_jax(jnp.array(ca_coords))
    """
    # For each structure as reference, compute RMSD to all in the batch
    return jax.vmap(lambda ref: _rmsd_parallel_jax(ref, coords), in_axes=(0,))(coords)


def calculate_batch_rmsd_gpu(ref_coords: np.ndarray, 
                              pred_coords_list: list, 
                              use_gpu: bool = True) -> np.ndarray:
    """
    Calculate RMSD between reference and multiple predictions using GPU acceleration.
    
    This function leverages JAX's vectorization to compute RMSDs in parallel on GPU,
    which is significantly faster than computing them sequentially on CPU.
    
    Args:
        ref_coords: Reference CA coordinates, shape [N_residues, 3]
        pred_coords_list: List of predicted CA coordinates, each shape [N_residues, 3]
        use_gpu: Whether to use GPU acceleration (default: True)
    
    Returns:
        Array of RMSD values, shape [N_predictions]
    
    Example:
        >>> ref_ca = ref_structure[:, 1, :]  # Extract CA atoms
        >>> pred_ca_list = [pred[:, 1, :] for pred in predictions]
        >>> rmsds = calculate_batch_rmsd_gpu(ref_ca, pred_ca_list)
    """
    if not use_gpu:
        # Fall back to CPU calculation
        try:
            from colabdesign.shared.protein import _np_rmsd
            return np.array([_np_rmsd(ref_coords, pred, use_jax=False) 
                            for pred in pred_coords_list])
        except ImportError:
            # Use our own CPU implementation if ColabDesign not available
            return np.array([calculate_rmsd(ref_coords, pred, align=True) 
                            for pred in pred_coords_list])
    
    try:
        # Convert to JAX arrays
        ref_jax = jnp.array(ref_coords)
        pred_batch = jnp.array(np.stack(pred_coords_list))
        
        # Calculate RMSDs in parallel on GPU
        rmsds = _rmsd_parallel_jax(ref_jax, pred_batch)
        
        return np.array(rmsds)
    
    except Exception as e:
        warnings.warn(f"GPU calculation failed ({e}), falling back to CPU")
        try:
            from colabdesign.shared.protein import _np_rmsd
            return np.array([_np_rmsd(ref_coords, pred, use_jax=False) 
                            for pred in pred_coords_list])
        except ImportError:
            # Use our own CPU implementation if ColabDesign not available
            return np.array([calculate_rmsd(ref_coords, pred, align=True) 
                            for pred in pred_coords_list])


def calculate_batch_rmsd_to_references(pred_coords_list: List[np.ndarray],
                                       ref1_path: str = "state1.pdb",
                                       ref2_path: str = "state2.pdb",
                                       use_gpu: bool = True) -> List[Dict[str, float]]:
    """
    Calculate RMSD to both reference states for multiple predictions using GPU acceleration.
    
    This function is optimized for batch processing many structures at once.
    
    Args:
        pred_coords_list: List of predicted atom coordinates, each with shape [L, 37, 3]
        ref1_path: Path to first reference PDB file
        ref2_path: Path to second reference PDB file
        use_gpu: Use GPU-accelerated RMSD calculation (default: True)
    
    Returns:
        List of dictionaries, each with 'rmsd_state1' and 'rmsd_state2' values
    
    Example:
        >>> predictions = [pred1['structure'], pred2['structure'], ...]
        >>> rmsds = calculate_batch_rmsd_to_references(predictions)
    """
    # Load reference structures CA coordinates
    ref1_coords = load_pdb_coords(ref1_path)
    ref2_coords = load_pdb_coords(ref2_path)
    
    # Extract CA coordinates from predictions
    pred_ca_list = [pred_coords[:, 1, :] for pred_coords in pred_coords_list]
    
    # Ensure all predictions have same length as references (trim if needed)
    min_len = min(len(ref1_coords), len(pred_ca_list[0]) if pred_ca_list else 0)
    ref1_trimmed = ref1_coords[:min_len]
    ref2_trimmed = ref2_coords[:min_len]
    pred_ca_trimmed = [pred_ca[:min_len] for pred_ca in pred_ca_list]
    
    # Calculate RMSDs in batch
    if use_gpu and len(pred_ca_list) > 1:
        try:
            # Use GPU batch calculation for efficiency
            rmsds1 = calculate_batch_rmsd_gpu(ref1_trimmed, pred_ca_trimmed, use_gpu=True)
            rmsds2 = calculate_batch_rmsd_gpu(ref2_trimmed, pred_ca_trimmed, use_gpu=True)
            
            # Package results
            results = []
            for rmsd1, rmsd2 in zip(rmsds1, rmsds2):
                results.append({'rmsd_state1': float(rmsd1), 'rmsd_state2': float(rmsd2)})
            return results
            
        except Exception as e:
            warnings.warn(f"GPU batch calculation failed ({e}), falling back to CPU")
            use_gpu = False
    
    # Fall back to sequential CPU calculation
    try:
        from colabdesign.shared.protein import _np_rmsd
        results = []
        for pred_ca in pred_ca_trimmed:
            rmsd1 = _np_rmsd(pred_ca, ref1_trimmed, use_jax=False)
            rmsd2 = _np_rmsd(pred_ca, ref2_trimmed, use_jax=False)
            results.append({'rmsd_state1': rmsd1, 'rmsd_state2': rmsd2})
    except ImportError:
        # Use our own CPU implementation if ColabDesign not available
        results = []
        for pred_ca in pred_ca_trimmed:
            rmsd1 = calculate_rmsd(pred_ca, ref1_trimmed, align=True)
            rmsd2 = calculate_rmsd(pred_ca, ref2_trimmed, align=True)
            results.append({'rmsd_state1': rmsd1, 'rmsd_state2': rmsd2})
    
    return results


def calculate_all_vs_all_rmsd(structures: List[np.ndarray],
                              chunk_size: int = 100,
                              use_gpu: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate all-vs-all RMSD matrix for a list of structures.
    
    Efficiently computes pairwise RMSDs between all structures using GPU acceleration.
    Supports multi-GPU via JAX pmap for large-scale computations.
    
    Args:
        structures: List of coordinate arrays, each shape [N_atoms, 3]
        chunk_size: Number of structures to process at once (for memory management)
        use_gpu: Whether to use GPU acceleration
    
    Returns:
        Tuple of (rmsd_matrix, mean_pairwise_rmsd)
        - rmsd_matrix: N x N numpy array of RMSD values
        - mean_pairwise_rmsd: Mean of upper triangular matrix (excluding diagonal)
    
    Example:
        >>> ca_coords = [struct[:, 1, :] for struct in ensemble_structures]
        >>> rmsd_mat, mean_rmsd = calculate_all_vs_all_rmsd(ca_coords)
    """
    n_structures = len(structures)
    
    if n_structures == 0:
        return np.zeros((0, 0)), 0.0
    
    # Stack and convert to JAX array
    try:
        coords = jnp.array(np.stack(structures))
        
        if use_gpu:
            # Check for multiple GPUs
            gpus = jax.devices("gpu")
            num_gpus = len(gpus)
            
            if num_gpus > 1:
                # Multi-GPU support via pmap
                warnings.warn(f"Using {num_gpus} GPUs for all-vs-all RMSD calculation")
                
                # Process in chunks for memory efficiency
                rmsd_matrix = np.zeros((n_structures, n_structures))
                
                for start in range(0, n_structures, chunk_size):
                    end = min(start + chunk_size, n_structures)
                    block = coords[start:end]
                    
                    # Calculate RMSD for this chunk vs all structures
                    sub_rmsd = jax.vmap(lambda ref: _rmsd_parallel_jax(ref, coords), 
                                       in_axes=(0,))(block)
                    rmsd_matrix[start:end] = np.array(sub_rmsd)
            else:
                # Single GPU or CPU
                rmsd_matrix = np.array(_pairwise_rmsd_matrix_jax(coords))
        else:
            # CPU fallback
            try:
                from colabdesign.shared.protein import _np_rmsd
                rmsd_matrix = np.zeros((n_structures, n_structures))
                for i in range(n_structures):
                    for j in range(i+1, n_structures):
                        rmsd = _np_rmsd(structures[i], structures[j], use_jax=False)
                        rmsd_matrix[i, j] = rmsd
                        rmsd_matrix[j, i] = rmsd
            except ImportError:
                # Use our own CPU implementation if ColabDesign not available
                rmsd_matrix = np.zeros((n_structures, n_structures))
                for i in range(n_structures):
                    for j in range(i+1, n_structures):
                        rmsd = calculate_rmsd(structures[i], structures[j], align=True)
                        rmsd_matrix[i, j] = rmsd
                        rmsd_matrix[j, i] = rmsd
    
    except Exception as e:
        warnings.warn(f"GPU calculation failed ({e}), using CPU fallback")
        # CPU fallback
        try:
            from colabdesign.shared.protein import _np_rmsd
            rmsd_matrix = np.zeros((n_structures, n_structures))
            for i in range(n_structures):
                for j in range(i+1, n_structures):
                    rmsd = _np_rmsd(structures[i], structures[j], use_jax=False)
                    rmsd_matrix[i, j] = rmsd
                    rmsd_matrix[j, i] = rmsd
        except ImportError:
            # Use our own CPU implementation if ColabDesign not available
            rmsd_matrix = np.zeros((n_structures, n_structures))
            for i in range(n_structures):
                for j in range(i+1, n_structures):
                    rmsd = calculate_rmsd(structures[i], structures[j], align=True)
                    rmsd_matrix[i, j] = rmsd
                    rmsd_matrix[j, i] = rmsd
    
    # Calculate mean pairwise RMSD (upper triangular, excluding diagonal)
    upper_tri_indices = np.triu_indices(n_structures, k=1)
    mean_pairwise_rmsd = np.mean(rmsd_matrix[upper_tri_indices]) if n_structures > 1 else 0.0
    
    return rmsd_matrix, mean_pairwise_rmsd


# =============================================================================
# MSA Visualization Classes and Utilities
# =============================================================================

@dataclass(frozen=True)
class MSAData:
    """
    Container for MSA data and metadata.
    
    Attributes:
        array: Numeric MSA array (N_sequences, L_positions)
        deletion_matrix: Deletion matrix (N_sequences, L_positions)
        sequences: Original list of sequences from parse_a3m
        neff: Number of effective sequences
        length: Sequence length (L)
        masked_positions: List of masked positions (1-based indexing)
        mutated_positions: List of mutated positions (1-based indexing)
        condition_name: Name of the experimental condition
    """
    array: np.ndarray
    deletion_matrix: np.ndarray
    sequences: Sequence[np.ndarray]
    neff: int
    length: int
    masked_positions: List[int] = field(default_factory=list)
    mutated_positions: List[int] = field(default_factory=list)
    condition_name: str = ""


class MSACoevolutionVisualizer:
    """
    Interactive MSA and coevolution visualization utilities.
    
    This class provides methods to load, analyze, and visualize MSAs with
    interactive Plotly-based plots. It supports mutations, masking, and
    comparative analysis of different MSA conditions.
    
    Example:
        >>> vis = MSACoevolutionVisualizer()
        >>> msa_data = vis.load_msa("path/to/msa.a3m")
        >>> coev = vis.compute_coevolution(msa_data)
        >>> fig = vis.plot_heatmap(coev, msa_data=msa_data)
        >>> fig.show()
    """
    
    def __init__(self):
        """Initialize MSA visualizer with residue type mappings."""
        self.restypes = [
            "A", "R", "N", "D", "C", "Q", "E", "G", "H", "I",
            "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V"
        ]
        self.restypes_with_x_and_gap = self.restypes + ["X", "-"]
        self._coev_cache: Dict[str, np.ndarray] = {}
    
    def load_msa(
        self,
        msa_path: Union[str, Path],
        *,
        mutations: Optional[List[str]] = None,
        masking_mode: str = "none",
        cols: Optional[List[int]] = None,
        mask_identity: str = "X",
        condition_name: str = "",
    ) -> MSAData:
        """
        Load and process an MSA file.
        
        Args:
            msa_path: Path to MSA file (a3m format)
            mutations: List of mutations to apply (e.g., ["I89N", "S74S"])
            masking_mode: Masking mode - "none", "list", or "ranges"
            cols: List of column positions to mask (1-based)
            mask_identity: Character to use for masking (default "X")
            condition_name: Human-readable condition name
            
        Returns:
            MSAData object containing processed MSA and metadata
        """
        msa_path = Path(msa_path)
        if not msa_path.exists():
            raise FileNotFoundError(f"MSA file not found: {msa_path}")
        
        # Import parse_a3m from colabdesign
        from colabdesign.af.contrib import predict
        
        # Parse MSA
        sequences, deletion_matrix = predict.parse_a3m(str(msa_path))
        msa_arr = np.asarray(sequences)
        
        logger.info(
            f"Loaded MSA {msa_path.name} - {msa_arr.shape[0]} sequences, "
            f"length={msa_arr.shape[1] if msa_arr.size else 0}"
        )
        
        # Apply mutations
        mutated_positions: List[int] = []
        if mutations:
            logger.debug(f"Applying mutations: {mutations}")
            msa_arr = self._mutate_first_sequence(msa_arr, mutations)
            mutated_positions = [int(mut[1:-1]) for mut in mutations]
        
        # Apply masking
        masked_positions: List[int] = []
        if masking_mode == "list" and cols:
            logger.debug(f"Masking columns: {cols}")
            msa_arr = self._mask_columns_list(msa_arr, cols, mask_identity)
            masked_positions = cols
        
        return MSAData(
            array=msa_arr,
            deletion_matrix=deletion_matrix,
            sequences=sequences,
            neff=msa_arr.shape[0],
            length=msa_arr.shape[1] if msa_arr.size else 0,
            masked_positions=masked_positions,
            mutated_positions=mutated_positions,
            condition_name=condition_name,
        )
    
    def _mutate_first_sequence(
        self, 
        arr: np.ndarray, 
        mutations: List[str]
    ) -> np.ndarray:
        """Apply mutations to first sequence in MSA."""
        arr = arr.copy()
        
        for mutation in mutations:
            if not re.match(r"^[A-Z]\d+[A-Z]$", mutation):
                raise ValueError(f"Invalid mutation format: {mutation}")
            
            original_residue = mutation[0]
            position = int(mutation[1:-1]) - 1  # Convert to 0-based
            new_residue = mutation[-1]
            
            if position < 0 or position >= arr.shape[1]:
                raise ValueError(f"Invalid position: {position + 1}")
            
            if self.restypes_with_x_and_gap[int(arr[0, position])] != original_residue:
                raise ValueError(
                    f"Original residue mismatch at position {position + 1}"
                )
            
            if new_residue not in self.restypes:
                raise ValueError(f"Invalid new residue: {new_residue}")
            
            arr[0, position] = self.restypes.index(new_residue)
            logger.info(f"Mutated residue {original_residue}{position + 1}{new_residue}")
        
        return arr
    
    def _mask_columns_list(
        self, 
        arr: np.ndarray, 
        cols: List[int], 
        mask_identity: str
    ) -> np.ndarray:
        """Mask specific columns in MSA (skips first sequence)."""
        arr = arr.copy()
        
        if not cols:
            raise ValueError("The list of columns cannot be empty")
        if mask_identity not in self.restypes_with_x_and_gap:
            raise ValueError(f"Invalid mask_identity: {mask_identity}")
        
        mask = self.restypes_with_x_and_gap.index(mask_identity)
        cols_zero_based = [col - 1 for col in cols]  # Convert to 0-based
        arr[1:, cols_zero_based] = mask  # Skip first sequence
        
        return arr
    
    def compute_coevolution(
        self, 
        msa: MSAData, 
        *, 
        force_recompute: bool = False
    ) -> np.ndarray:
        """
        Compute coevolution matrix with caching.
        
        Args:
            msa: MSAData object
            force_recompute: Force recomputation even if cached
            
        Returns:
            Coevolution matrix (L x L)
        """
        # Create cache key
        cache_key = hashlib.md5(
            json.dumps(
                {
                    "condition": msa.condition_name,
                    "masked_pos": msa.masked_positions,
                    "mutated_pos": msa.mutated_positions,
                    "sha": hashlib.md5(msa.array.tobytes()).hexdigest(),
                },
                sort_keys=True,
            ).encode()
        ).hexdigest()
        
        if not force_recompute and cache_key in self._coev_cache:
            logger.debug(f"Using cached coevolution for {msa.condition_name}")
            return self._coev_cache[cache_key]
        
        logger.debug(f"Computing coevolution for {msa.condition_name}")
        coev = get_coevolution(msa.array)
        # Convert JAX array to numpy for caching and return
        coev = np.array(coev)
        self._coev_cache[cache_key] = coev
        
        return coev
    
    def plot_heatmap(
        self,
        coev_matrix: np.ndarray,
        *,
        title: str = "Coevolution",
        msa_data: Optional[MSAData] = None,
        show_colorbar: bool = True,
    ):
        """
        Create interactive Plotly heatmap of coevolution matrix.
        
        Args:
            coev_matrix: Coevolution matrix to plot
            title: Plot title
            msa_data: Optional MSAData for enhanced hover information
            show_colorbar: Whether to show colorbar
            
        Returns:
            Plotly Figure object
        """
        try:
            import plotly.graph_objects as go
        except ImportError:
            raise ImportError("Plotly is required for interactive plots. Install with: pip install plotly")
        
        if msa_data is not None:
            # Enhanced hover with amino acid identities
            customdata = self._create_customdata_matrix(msa_data)
            stats = {
                "mean": np.mean(coev_matrix),
                "max": np.max(coev_matrix),
                "std": np.std(coev_matrix)
            }
            hovertemplate = self._create_enhanced_hovertemplate(
                msa_data, title, stats, is_difference=False
            )
            
            fig = go.Figure(
                data=[
                    go.Heatmap(
                        z=coev_matrix,
                        colorscale="Viridis",
                        showscale=show_colorbar,
                        customdata=customdata,
                        hovertemplate=hovertemplate,
                        x=list(range(1, coev_matrix.shape[1] + 1)),
                        y=list(range(1, coev_matrix.shape[0] + 1)),
                    )
                ]
            )
        else:
            # Basic hover
            hovertemplate = (
                "Position i: %{x}<br>Position j: %{y}<br>"
                "Coevolution: %{z:.3f}<extra></extra>"
            )
            
            fig = go.Figure(
                data=[
                    go.Heatmap(
                        z=coev_matrix,
                        colorscale="Viridis",
                        showscale=show_colorbar,
                        hovertemplate=hovertemplate,
                        x=list(range(1, coev_matrix.shape[1] + 1)),
                        y=list(range(1, coev_matrix.shape[0] + 1)),
                    )
                ]
            )
        
        fig.update_layout(
            title=title,
            xaxis_title="Residue Position (1-indexed)",
            yaxis_title="Residue Position (1-indexed)",
            width=800,
            height=800,
        )
        
        return fig
    
    def plot_difference(
        self,
        matrix_a: np.ndarray,
        matrix_b: np.ndarray,
        *,
        title: str = "Difference (A-B)",
        msa_data: Optional[MSAData] = None,
    ):
        """
        Create difference plot comparing two coevolution matrices.
        
        Args:
            matrix_a: First coevolution matrix
            matrix_b: Second coevolution matrix
            title: Plot title
            msa_data: Optional MSAData for enhanced hover
            
        Returns:
            Plotly Figure object
        """
        try:
            import plotly.graph_objects as go
        except ImportError:
            raise ImportError("Plotly is required for interactive plots")
        
        diff = matrix_a - matrix_b
        max_abs = float(np.max(np.abs(diff))) or 1.0
        
        if msa_data is not None:
            customdata = self._create_customdata_matrix(msa_data)
            hovertemplate = self._create_enhanced_hovertemplate(
                msa_data, title, is_difference=True
            )
            
            fig = go.Figure(
                data=[
                    go.Heatmap(
                        z=diff,
                        colorscale="RdBu_r",
                        zmid=0,
                        zmin=-max_abs,
                        zmax=max_abs,
                        customdata=customdata,
                        hovertemplate=hovertemplate,
                        x=list(range(1, diff.shape[1] + 1)),
                        y=list(range(1, diff.shape[0] + 1)),
                    )
                ]
            )
        else:
            fig = go.Figure(
                data=[
                    go.Heatmap(
                        z=diff,
                        colorscale="RdBu_r",
                        zmid=0,
                        zmin=-max_abs,
                        zmax=max_abs,
                        hovertemplate="Position i: %{x}<br>Position j: %{y}<br>Δ: %{z:.3f}<extra></extra>",
                        x=list(range(1, diff.shape[1] + 1)),
                        y=list(range(1, diff.shape[0] + 1)),
                    )
                ]
            )
        
        fig.update_layout(
            title=title,
            xaxis_title="Residue Position (1-indexed)",
            yaxis_title="Residue Position (1-indexed)",
            width=800,
            height=800,
        )
        
        return fig
    
    def print_diagnostics(self, msa: MSAData) -> None:
        """Print comprehensive diagnostics for an MSA."""
        print(f"Diagnostic Analysis for {msa.condition_name}:")
        print(f"   MSA shape: {msa.array.shape}")
        print(f"   Sequence length: {msa.length}")
        print(f"   Number of sequences (Neff): {msa.neff}")
        
        # Check first sequence for mutations and masking
        first_seq = msa.array[0]
        
        # Identify masked positions (X = index 20)
        x_positions_zero_based = np.where(first_seq == 20)[0]
        x_positions = x_positions_zero_based + 1  # Convert to 1-based
        
        if len(x_positions) > 0:
            print(f"   Masked positions in first sequence (X): {list(x_positions)}")
            
            # Check if other sequences are also masked
            for pos_idx, pos in enumerate(x_positions[:5]):  # Check first 5
                pos_zero_based = pos - 1
                masked_count = np.sum(msa.array[:, pos_zero_based] == 20)
                total_seqs = len(msa.array)
                masked_percent = (masked_count / total_seqs) * 100
                print(f"   Position {pos}: {masked_count}/{total_seqs} sequences masked ({masked_percent:.1f}%)")
        else:
            print(f"   No masked positions (X) found in first sequence")
        
        # Check for mutations if expected
        if msa.mutated_positions:
            print(f"   Expected mutations at positions: {msa.mutated_positions}")
            for pos in msa.mutated_positions:
                pos_zero_based = pos - 1
                if pos_zero_based < len(first_seq):
                    residue = self.restypes_with_x_and_gap[first_seq[pos_zero_based]]
                    print(f"      Position {pos}: {residue}")
        
        # Report masking/mutation status
        if msa.masked_positions:
            print(f"   Masking applied at positions: {msa.masked_positions}")
        if msa.mutated_positions:
            print(f"   Mutations applied at positions: {msa.mutated_positions}")
    
    def _create_customdata_matrix(self, msa_data: MSAData) -> np.ndarray:
        """Create customdata matrix for hover templates with amino acid pairs."""
        first_seq = msa_data.array[0]
        seq_length = len(first_seq)
        
        # Create matrix where each cell contains [residue_i, residue_j]
        customdata = np.empty((seq_length, seq_length), dtype=object)
        
        for i in range(seq_length):
            for j in range(seq_length):
                res_i = self.restypes_with_x_and_gap[first_seq[i]]
                res_j = self.restypes_with_x_and_gap[first_seq[j]]
                customdata[i, j] = [res_i, res_j]
        
        return customdata
    
    def _create_enhanced_hovertemplate(
        self,
        msa_data: MSAData,
        condition_name: str,
        stats: Optional[Dict] = None,
        is_difference: bool = False
    ) -> str:
        """Create enhanced hover template with amino acid identities."""
        if is_difference:
            template = (
                f"<b>{condition_name}</b><br>" +
                "Position i: %{x}<br>" +
                "Position j: %{y}<br>" +
                "Residue i: %{customdata[0]}<br>" +
                "Residue j: %{customdata[1]}<br>" +
                "Difference: %{z:.3f}<br>" +
                "<extra></extra>"
            )
        else:
            base_template = (
                f"<b>{condition_name}</b><br>" +
                "Position i: %{x}<br>" +
                "Position j: %{y}<br>" +
                "Residue i: %{customdata[0]}<br>" +
                "Residue j: %{customdata[1]}<br>" +
                "Coevolution: %{z:.3f}<br>" +
                f"Neff: {msa_data.neff}<br>"
            )
            
            if stats:
                base_template += (
                    f"Mean: {stats['mean']:.3f}<br>" +
                    f"Max: {stats['max']:.3f}<br>" +
                    f"Std: {stats['std']:.3f}<br>"
                )
            
            template = base_template + "<extra></extra>"
        
        return template


def get_coevolution(msa: np.ndarray) -> np.ndarray:
    """
    Compute coevolution matrix from MSA using direct coupling analysis.
    
    Args:
        msa: MSA array with shape (N_sequences, L_residues)
        
    Returns:
        Coevolution matrix with shape (L, L)
    """
    # Check for insufficient sequences - return zeros as coevolution needs multiple sequences
    # Need at least 2 sequences for meaningful covariance
    if msa.shape[0] < 2:
        logger.debug(f"MSA has only {msa.shape[0]} sequence(s), returning zero coevolution matrix")
        return np.zeros((msa.shape[1], msa.shape[1]))
    
    # For very small MSAs, the coevolution might not be reliable but we'll compute it anyway
    if msa.shape[0] < 10:
        logger.warning(f"MSA has only {msa.shape[0]} sequences, coevolution may not be reliable")
    
    # Use the JIT-compiled version for multiple sequences
    return _get_coevolution_jit(msa)

@jax.jit
def _get_coevolution_jit(msa: np.ndarray):
    """JIT-compiled coevolution calculation for multiple sequences."""
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
    
    return coev  # Return JAX array directly, conversion happens outside JIT


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
    try:
        from colabdesign.shared.protein import _np_rmsd
        return _np_rmsd(coords1, coords2, use_jax=False)
    except ImportError:
        # Standalone CPU implementation when ColabDesign is not available
        coords1 = np.array(coords1)
        coords2 = np.array(coords2)
        
        if align:
            # Center coordinates
            c1 = coords1 - coords1.mean(axis=0)
            c2 = coords2 - coords2.mean(axis=0)
            
            # Kabsch algorithm for optimal rotation
            H = c1.T @ c2
            U, S, Vt = np.linalg.svd(H)
            R = Vt.T @ U.T
            
            # Ensure proper rotation (det(R) = 1)
            if np.linalg.det(R) < 0:
                Vt[-1, :] *= -1
                R = Vt.T @ U.T
            
            # Apply rotation
            c1 = c1 @ R
            
            # Calculate RMSD
            return np.sqrt(np.mean(np.sum((c1 - c2)**2, axis=1)))
        else:
            # Simple RMSD without alignment
            return np.sqrt(np.mean(np.sum((coords1 - coords2)**2, axis=1)))


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
    coev = np.array(get_coevolution(msa))  # Convert JAX array to numpy
    
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


def plot_msa_interactive(
    msa_data: MSAData,
    *,
    show_coverage: bool = True,
    show_identity: bool = True,
    title: str = "MSA Analysis"
):
    """
    Create interactive MSA visualizations with Plotly.
    
    Args:
        msa_data: MSAData object from MSACoevolutionVisualizer
        show_coverage: Show coverage plot
        show_identity: Show sequence identity distribution
        title: Plot title
        
    Returns:
        Plotly Figure object
    """
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        raise ImportError("Plotly is required. Install with: pip install plotly")
    
    # Calculate MSA statistics
    msa_arr = msa_data.array
    
    # Coverage: percentage of sequences with non-gap residues at each position
    # In ColabDesign, gaps are typically represented as index 21 or high values
    coverage = np.sum(msa_arr < 21, axis=0) / len(msa_arr) * 100
    
    # Sequence identity to query (first sequence)
    query_seq = msa_arr[0]
    identities = []
    for seq in msa_arr[1:]:
        identity = np.mean(seq == query_seq) * 100
        identities.append(identity)
    
    # Create subplots
    n_plots = sum([show_coverage, show_identity])
    if n_plots == 0:
        raise ValueError("At least one plot type must be enabled")
    
    fig = make_subplots(
        rows=n_plots, cols=1,
        subplot_titles=["MSA Coverage" if show_coverage else None, 
                       "Sequence Identity Distribution" if show_identity else None],
        vertical_spacing=0.15
    )
    
    plot_idx = 1
    
    if show_coverage:
        fig.add_trace(
            go.Scatter(
                x=list(range(1, len(coverage) + 1)),
                y=coverage,
                mode='lines',
                line=dict(color='steelblue', width=2),
                name='Coverage',
                hovertemplate="Position: %{x}<br>Coverage: %{y:.1f}%<extra></extra>"
            ),
            row=plot_idx, col=1
        )
        fig.update_xaxes(title_text="Residue Position", row=plot_idx, col=1)
        fig.update_yaxes(title_text="Coverage (%)", range=[0, 105], row=plot_idx, col=1)
        plot_idx += 1
    
    if show_identity:
        fig.add_trace(
            go.Histogram(
                x=identities,
                nbinsx=20,
                marker=dict(color='coral', line=dict(color='black', width=1)),
                name='Identity',
                hovertemplate="Identity: %{x:.1f}%<br>Count: %{y}<extra></extra>"
            ),
            row=plot_idx, col=1
        )
        fig.update_xaxes(title_text="Sequence Identity to Query (%)", row=plot_idx, col=1)
        fig.update_yaxes(title_text="Count", row=plot_idx, col=1)
    
    fig.update_layout(
        title=f"{title}<br><sub>Neff: {msa_data.neff}, Length: {msa_data.length}</sub>",
        height=400 * n_plots,
        showlegend=False
    )
    
    return fig


def plot_coevolution_interactive(
    msa: Union[np.ndarray, MSAData],
    *,
    title: str = "Coevolution Matrix",
    condition_name: str = ""
):
    """
    Create interactive coevolution heatmap with Plotly.
    
    Args:
        msa: MSA array or MSAData object
        title: Plot title
        condition_name: Condition name for enhanced display
        
    Returns:
        Plotly Figure object
    """
    # Create visualizer
    vis = MSACoevolutionVisualizer()
    
    # Handle both MSAData and raw arrays
    if isinstance(msa, MSAData):
        msa_data = msa
        coev = vis.compute_coevolution(msa_data)
    else:
        # Create minimal MSAData for raw array
        msa_data = MSAData(
            array=msa,
            deletion_matrix=np.zeros_like(msa),
            sequences=[msa],
            neff=msa.shape[0],
            length=msa.shape[1],
            condition_name=condition_name or "MSA"
        )
        coev = np.array(get_coevolution(msa))  # Convert JAX array to numpy
    
    # Create plot
    fig = vis.plot_heatmap(coev, title=title, msa_data=msa_data)
    
    return fig


def compare_coevolution_conditions(
    conditions: Dict[str, Union[str, Path, MSAData]],
    *,
    show_difference: bool = True,
    reference_condition: Optional[str] = None
):
    """
    Compare coevolution across multiple MSA conditions.
    
    Args:
        conditions: Dict mapping condition names to MSA paths or MSAData objects
        show_difference: Whether to show difference plots
        reference_condition: Reference condition for difference plots (default: first)
        
    Returns:
        Tuple of (main_figure, difference_figure) Plotly objects
    """
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        raise ImportError("Plotly is required. Install with: pip install plotly")
    
    vis = MSACoevolutionVisualizer()
    
    # Load all MSA data
    msa_store: Dict[str, MSAData] = {}
    coev_mats: Dict[str, np.ndarray] = {}
    
    for cond_name, msa_source in conditions.items():
        if isinstance(msa_source, MSAData):
            msa_data = msa_source
        else:
            # Load from path
            msa_data = vis.load_msa(msa_source, condition_name=cond_name)
        
        msa_store[cond_name] = msa_data
        coev_mats[cond_name] = vis.compute_coevolution(msa_data)
    
    # Create main comparison plot
    n = len(coev_mats)
    cols = min(2, n)
    rows = (n + cols - 1) // cols
    
    fig_main = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=list(coev_mats.keys()),
        vertical_spacing=0.08,
        horizontal_spacing=0.05,
    )
    
    for idx, (cond_name, coev_mat) in enumerate(coev_mats.items()):
        r = idx // cols + 1
        c = idx % cols + 1
        
        msa_data = msa_store[cond_name]
        customdata = vis._create_customdata_matrix(msa_data)
        stats = {"mean": np.mean(coev_mat), "max": np.max(coev_mat), "std": np.std(coev_mat)}
        hovertemplate = vis._create_enhanced_hovertemplate(
            msa_data, cond_name, stats, is_difference=False
        )
        
        hm = go.Heatmap(
            z=coev_mat,
            colorscale="Viridis",
            showscale=(idx == 0),
            name=f"{cond_name} (Neff: {msa_data.neff})",
            customdata=customdata,
            hovertemplate=hovertemplate,
            x=list(range(1, coev_mat.shape[1] + 1)),
            y=list(range(1, coev_mat.shape[0] + 1)),
        )
        fig_main.add_trace(hm, row=r, col=c)
    
    fig_main.update_layout(
        title="Coevolution Analysis - Multiple Conditions",
        height=600 * rows,
        width=1200,
        showlegend=False,
    )
    
    # Update axis labels
    for r in range(1, rows + 1):
        for c in range(1, cols + 1):
            fig_main.update_xaxes(title_text="Residue Position", row=r, col=c)
            fig_main.update_yaxes(title_text="Residue Position", row=r, col=c)
    
    fig_diff = None
    if show_difference and len(coev_mats) >= 2:
        # Create difference plots
        ref_name = reference_condition or list(coev_mats.keys())[0]
        ref_mat = coev_mats[ref_name]
        ref_msa = msa_store[ref_name]
        
        other_conditions = {k: v for k, v in coev_mats.items() if k != ref_name}
        n_diff = len(other_conditions)
        
        if n_diff > 0:
            diff_cols = min(2, n_diff)
            diff_rows = (n_diff + diff_cols - 1) // diff_cols
            
            fig_diff = make_subplots(
                rows=diff_rows,
                cols=diff_cols,
                subplot_titles=[f"{k} - {ref_name}" for k in other_conditions.keys()],
                vertical_spacing=0.12,
                horizontal_spacing=0.1,
            )
            
            for idx, (other_name, other_mat) in enumerate(other_conditions.items()):
                r = idx // diff_cols + 1
                c = idx % diff_cols + 1
                
                diff = other_mat - ref_mat
                max_abs = float(np.max(np.abs(diff))) or 1.0
                
                customdata = vis._create_customdata_matrix(ref_msa)
                hovertemplate = vis._create_enhanced_hovertemplate(
                    ref_msa, f"{other_name} - {ref_name}", is_difference=True
                )
                
                fig_diff.add_trace(
                    go.Heatmap(
                        z=diff,
                        colorscale="RdBu_r",
                        zmid=0,
                        zmin=-max_abs,
                        zmax=max_abs,
                        showscale=(idx == 0),
                        customdata=customdata,
                        hovertemplate=hovertemplate,
                        x=list(range(1, diff.shape[1] + 1)),
                        y=list(range(1, diff.shape[0] + 1)),
                    ),
                    row=r,
                    col=c,
                )
            
            fig_diff.update_layout(
                title=f"Coevolution Differences (relative to {ref_name})",
                height=600 * diff_rows,
                width=1200,
                showlegend=False,
            )
            
            # Update axis labels
            for r in range(1, diff_rows + 1):
                for c in range(1, diff_cols + 1):
                    fig_diff.update_xaxes(title_text="Residue Position", row=r, col=c)
                    fig_diff.update_yaxes(title_text="Residue Position", row=r, col=c)
    
    return fig_main, fig_diff


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
                                   save_all_pdbs: bool = False,
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
        save_all_pdbs: Save PDB files for each prediction (default: False)
        verbose: Print progress
        
    Returns:
        List of all predictions (includes 'job_folder' key if save_all_pdbs=True)
    """
    all_predictions = []
    
    # Create job folder if saving PDBs
    job_folder = None
    if save_all_pdbs:
        job_folder = create_job_folder(sequence, jobname)
        if verbose:
            print(f"Created job folder: {job_folder}")
    
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
        
        # Add MSA mode and save PDBs if requested
        for pred in predictions:
            pred['msa_mode'] = msa_mode
            
            # Save PDB for this prediction
            if save_all_pdbs and job_folder:
                seed = pred.get('seed', 0)
                pdb_path = f"{job_folder}/pdb/{msa_mode}_seed{seed}.pdb"
                save_pdb(
                    atom_positions=pred['structure'],
                    sequence=sequence,
                    output_path=pdb_path,
                    plddt=pred.get('plddt')
                )
        
        all_predictions.extend(predictions)
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Total predictions generated: {len(all_predictions)}")
        if save_all_pdbs:
            print(f"PDFs saved in: {job_folder}/pdb/")
        print(f"{'='*60}")
    
    # Add job folder to all predictions if saving
    if save_all_pdbs and job_folder:
        for pred in all_predictions:
            pred['job_folder'] = job_folder
        
        # Save best prediction
        best_pred = max(all_predictions, key=lambda x: x['metrics']['plddt'])
        best_pdb_path = f"{job_folder}/pdb/best.pdb"
        save_pdb(
            atom_positions=best_pred['structure'],
            sequence=sequence,
            output_path=best_pdb_path,
            plddt=best_pred.get('plddt')
        )
        if verbose:
            print(f"Best prediction saved to: {best_pdb_path}")
    
    return all_predictions


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def save_pdb(atom_positions: np.ndarray,
            sequence: str,
            output_path: str,
            plddt: Optional[np.ndarray] = None) -> None:
    """
    Save structure to PDB file with all atoms using ColabDesign's protein.to_pdb.
    
    Args:
        atom_positions: Atom coordinates (L, 37, 3)
        sequence: Amino acid sequence
        output_path: Output PDB file path
        plddt: Optional per-residue pLDDT scores
    """
    try:
        # Use ColabDesign's proper PDB generation with all atoms
        from colabdesign.af.alphafold.common import protein
        
        # Convert sequence to aatype array
        restype_order = {
            'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4,
            'G': 5, 'H': 6, 'I': 7, 'K': 8, 'L': 9,
            'M': 10, 'N': 11, 'P': 12, 'Q': 13, 'R': 14,
            'S': 15, 'T': 16, 'V': 17, 'W': 18, 'Y': 19
        }
        aatype = np.array([restype_order.get(aa, 0) for aa in sequence])
        
        # Create residue indices
        residue_index = np.arange(len(sequence))
        
        # Create atom mask (assuming all atoms are present)
        atom_mask = np.ones((len(sequence), 37))
        
        # Prepare protein dict
        p = {
            "aatype": aatype,
            "residue_index": residue_index,
            "atom_positions": atom_positions,
            "atom_mask": atom_mask
        }
        
        # Add B-factors if pLDDT is provided
        if plddt is not None:
            # Ensure plddt is in 0-100 range
            if plddt.max() <= 1.0:
                plddt = plddt * 100
            p["b_factors"] = atom_mask * plddt[..., None]
        else:
            p["b_factors"] = atom_mask * 50.0  # Default B-factor
        
        # Convert to PDB string using ColabDesign's method
        pdb_str = protein.to_pdb(protein.Protein(**p))
        
        # Save to file
        with open(output_path, 'w') as f:
            f.write(pdb_str)
            
    except ImportError:
        # Fallback to BioPython if ColabDesign not available
        warnings.warn("ColabDesign not available, saving CA-only PDB")
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
        
        for res_idx, aa in enumerate(sequence):
            resname = aa_map.get(aa, 'UNK')
            residue = Residue.Residue((' ', res_idx + 1, ' '), resname, '')
            
            # Add CA atom (index 1 in atom_positions)
            ca_coord = atom_positions[res_idx, 1, :]
            b_factor = plddt[res_idx] * 100 if plddt is not None else 50.0
            if plddt is not None and plddt.max() <= 1.0:
                b_factor = plddt[res_idx] * 100
            ca_atom = Atom.Atom('CA', ca_coord, 1.0, b_factor, 
                               ' ', 'CA', 0, 'C')
            residue.add(ca_atom)
            
            chain.add(residue)
        
        model.add(chain)
        structure.add(model)
        
        # Save
        io = PDBIO()
        io.set_structure(structure)
        io.save(output_path)


def save_pdb_string(atom_positions: np.ndarray,
                   sequence: str,
                   plddt: Optional[np.ndarray] = None) -> str:
    """
    Generate PDB format string from structure (for LogMD).
    
    Args:
        atom_positions: Atom coordinates (L, 37, 3)
        sequence: Amino acid sequence
        plddt: Optional per-residue pLDDT scores
        
    Returns:
        PDB format string
    """
    try:
        import logmd_utils
        return logmd_utils.create_pdb_string(atom_positions, sequence, plddt)
    except ImportError:
        # Fallback to simple implementation
        lines = []
        aa_map = {
            'A': 'ALA', 'C': 'CYS', 'D': 'ASP', 'E': 'GLU', 'F': 'PHE',
            'G': 'GLY', 'H': 'HIS', 'I': 'ILE', 'K': 'LYS', 'L': 'LEU',
            'M': 'MET', 'N': 'ASN', 'P': 'PRO', 'Q': 'GLN', 'R': 'ARG',
            'S': 'SER', 'T': 'THR', 'V': 'VAL', 'W': 'TRP', 'Y': 'TYR'
        }
        
        for res_idx, aa in enumerate(sequence):
            resname = aa_map.get(aa, 'UNK')
            ca_coord = atom_positions[res_idx, 1, :]
            b_factor = plddt[res_idx] if plddt is not None else 1.0
            line = (
                f"ATOM  {res_idx+1:5d}  CA  {resname:3s} A{res_idx+1:4d}    "
                f"{ca_coord[0]:8.3f}{ca_coord[1]:8.3f}{ca_coord[2]:8.3f}"
                f"  1.00{b_factor:6.2f}           C  "
            )
            lines.append(line)
        
        lines.append("END")
        return "\n".join(lines)


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


def create_trajectory_from_ensemble(
    predictions: List[Dict[str, Any]],
    sequence: str,
    project: str = "AF2_Tutorial",
    align_structures: bool = True,
    sort_by_rmsd: bool = False,
    reference_coords: Optional[np.ndarray] = None,
    max_structures: Optional[int] = None,
    verbose: bool = True
) -> Optional[Any]:
    """
    Create LogMD trajectory from ensemble predictions.
    
    Args:
        predictions: List of prediction dictionaries with 'structure' and 'plddt'
        sequence: Amino acid sequence
        project: LogMD project name
        align_structures: Align all structures to first frame
        sort_by_rmsd: Sort structures by RMSD to reference
        reference_coords: Reference CA coordinates for sorting
        max_structures: Maximum number of structures to include
        verbose: Print progress information
        
    Returns:
        LogMD trajectory instance or None if unavailable
    """
    if not check_logmd():
        if verbose:
            print("LogMD not available. Install with: pip install logmd")
        return None
    
    try:
        import logmd_utils
        trajectory = logmd_utils.create_trajectory_from_predictions(
            predictions=predictions,
            sequence=sequence,
            project=project,
            align_structures=align_structures,
            sort_by_rmsd=sort_by_rmsd,
            reference_coords=reference_coords,
            max_structures=max_structures
        )
        
        if verbose and trajectory:
            print(f"\nLogMD trajectory created with {len(predictions)} frames")
            print(f"View at: {trajectory.url}")
        
        return trajectory
    except Exception as e:
        if verbose:
            print(f"Failed to create trajectory: {e}")
        return None


def create_recycle_trajectory(
    job_folder: str,
    sequence: str,
    model_name: str = "model",
    seed: int = 0,
    align_to_first: bool = True,
    project: str = "recycle_trajectory",
    verbose: bool = True
) -> Optional[Any]:
    """
    Load recycle PDB files and create LogMD trajectory.
    
    This function loads all PDB files from recycling iterations and creates
    an interactive LogMD trajectory, with structures aligned to the first
    recycle (following AlphaMask pattern).
    
    Args:
        job_folder: Job folder containing pdb/recycles/ subdirectory
        sequence: Amino acid sequence
        model_name: Model name used in PDB filenames (default: "model")
        seed: Random seed used during prediction (default: 0)
        align_to_first: Align all structures to first recycle (default: True)
        project: LogMD project name
        verbose: Print progress information
        
    Returns:
        LogMD trajectory instance or None if unavailable/failed
        
    Example:
        >>> folder = create_job_folder("MKTAY...", "my_protein")
        >>> result = predict_with_recycling(model, msa, save_pdbs=True, 
        ...                                  job_folder=folder, sequence=seq)
        >>> traj = create_recycle_trajectory(folder, seq)
        >>> # View trajectory showing structure evolution through recycling
    """
    import glob
    
    if not check_logmd():
        if verbose:
            print("LogMD not available. Install with: pip install logmd")
        return None
    
    # Find all recycle PDB files
    recycle_dir = f"{job_folder}/pdb/recycles"
    pattern = f"{recycle_dir}/{model_name}_r*_seed{seed}.pdb"
    pdb_files = sorted(glob.glob(pattern))
    
    if not pdb_files:
        if verbose:
            print(f"No PDB files found matching pattern: {pattern}")
        return None
    
    if verbose:
        print(f"Found {len(pdb_files)} recycle PDB files")
    
    try:
        import logmd_utils
        
        # Load structures and pLDDT values
        structures = []
        plddts = []
        
        for pdb_file in pdb_files:
            # Load coordinates
            coords = load_pdb_coords(pdb_file)
            
            # Convert CA coords to full atom_positions format (N, 37, 3)
            # For now, we'll create a minimal representation with CA atoms
            full_structure = np.zeros((len(coords), 37, 3))
            full_structure[:, 1, :] = coords  # CA is index 1
            structures.append(full_structure)
            
            # Try to extract pLDDT from B-factor if available
            try:
                from Bio import PDB
                parser = PDB.PDBParser(QUIET=True)
                structure = parser.get_structure("protein", pdb_file)
                plddt_values = []
                for model in structure:
                    for chain in model:
                        for residue in chain:
                            if 'CA' in residue:
                                plddt_values.append(residue['CA'].bfactor / 100.0)
                plddts.append(np.array(plddt_values) if plddt_values else None)
            except:
                plddts.append(None)
        
        # Create predictions list for LogMD
        predictions = []
        for i, (struct, plddt) in enumerate(zip(structures, plddts)):
            pred = {
                'structure': struct,
                'plddt': plddt if plddt is not None else np.ones(len(struct)) * 0.9,
                'recycle': i,
                'seed': seed
            }
            predictions.append(pred)
        
        # Create trajectory with alignment to first recycle
        trajectory = logmd_utils.create_trajectory_from_predictions(
            predictions=predictions,
            sequence=sequence,
            project=project,
            align_structures=align_to_first,
            sort_by_rmsd=False,  # Already in recycle order
            reference_coords=None,
            max_structures=None
        )
        
        if verbose and trajectory:
            print(f"\nRecycle trajectory created with {len(predictions)} frames")
            print(f"View at: {trajectory.url}")
            if align_to_first:
                print("Structures aligned to first recycle (recycle 0)")
        
        return trajectory
        
    except Exception as e:
        if verbose:
            print(f"Failed to create recycle trajectory: {e}")
        return None


def predict_with_logmd(
    sequence: str,
    msa_mode: str = "mmseqs2",
    num_recycles: int = 3,
    project: str = "AF2_Tutorial",
    save_pdbs: bool = True,
    jobname: str = "prediction",
    show_viewer: bool = True,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Run prediction with real-time LogMD visualization and PDB saving.
    
    Args:
        sequence: Amino acid sequence
        msa_mode: MSA generation mode ("mmseqs2" or "single_sequence")
        num_recycles: Number of recycling iterations
        project: LogMD project name
        save_pdbs: Save PDB files for each recycle (default: True)
        jobname: Job name for folder creation (default: "prediction")
        show_viewer: Display LogMD viewer in notebook
        verbose: Print progress information
        
    Returns:
        Dictionary with prediction results, trajectory, and job_folder
    """
    if not check_logmd():
        if verbose:
            print("LogMD not available. Falling back to standard prediction.")
        return quick_predict(sequence, msa_mode, num_recycles, verbose=verbose)
    
    try:
        import logmd_utils
        
        # Create job folder if saving PDBs
        job_folder = None
        if save_pdbs:
            job_folder = create_job_folder(sequence, jobname)
            if verbose:
                print(f"Created job folder: {job_folder}")
        
        # Create LogMD trajectory
        integration = logmd_utils.LogMDIntegration()
        trajectory = integration.create_trajectory(project=project)
        
        if trajectory is None:
            if verbose:
                print("Failed to create LogMD trajectory. Falling back to standard prediction.")
            return quick_predict(sequence, msa_mode, num_recycles, verbose=verbose)
        
        if verbose:
            print(f"Real-time visualization: {trajectory.url}")
            print("\nRunning prediction with recycling...")
        
        # Setup model
        model = setup_model(sequence, verbose=False)
        
        # Generate MSA
        if msa_mode == "mmseqs2":
            msa, deletion_matrix = get_msa([sequence], "temp_logmd", verbose=False)
        else:
            msa, deletion_matrix = create_single_sequence_msa(sequence)
        
        # Set MSA
        model.set_msa(msa, deletion_matrix)
        model.set_seed(0)
        
        # Get reference CA for alignment
        reference_ca = None
        all_structures = []
        
        # Run prediction with recycling
        for recycle in range(num_recycles + 1):
            model.set_opt(num_recycles=recycle)
            model._inputs.pop("prev", None)
            model.predict(dropout=False, verbose=False)
            
            # Get structure
            atom_positions = model.aux['atom_positions'].copy()
            plddt = model.aux['plddt'].copy()
            
            # Save PDB if requested
            if save_pdbs and job_folder:
                pdb_path = f"{job_folder}/pdb/recycles/model_r{recycle}_seed0.pdb"
                save_pdb(
                    atom_positions=atom_positions,
                    sequence=sequence,
                    output_path=pdb_path,
                    plddt=plddt
                )
            
            # Align to first frame (AlphaMask style)
            if reference_ca is None:
                reference_ca = logmd_utils.get_ca_positions(atom_positions)
                aligned_positions = atom_positions
            else:
                aligned_positions = logmd_utils.superimpose_structures(
                    atom_positions, reference_ca
                )
            
            # Add to trajectory
            integration.add_structure(
                trajectory,
                aligned_positions,
                sequence,
                plddt=plddt,
                label=f"Recycle {recycle}",
                metadata={'recycle': recycle, 'mean_plddt': float(plddt.mean())}
            )
            
            all_structures.append({
                'structure': atom_positions,
                'plddt': plddt,
                'recycle': recycle
            })
            
            if verbose:
                pdb_status = " [PDB saved]" if save_pdbs else ""
                print(f"  Recycle {recycle}: pLDDT={plddt.mean():.3f}{pdb_status}")
        
        # Get final result
        final_result = {
            'structure': model.aux['atom_positions'],
            'plddt': model.aux['plddt'],
            'pae': model.aux.get('pae', None),
            'ptm': model.aux.get('ptm', 0.0),
            'metrics': {
                'plddt': model.aux['plddt'].mean(),
                'ptm': model.aux.get('ptm', 0.0),
            },
            'trajectory': trajectory,
            'all_structures': all_structures
        }
        
        if save_pdbs:
            final_result['job_folder'] = job_folder
        
        # Display in notebook if requested
        if show_viewer:
            try:
                logmd_utils.display_trajectory_in_notebook(trajectory)
            except:
                if verbose:
                    print(f"\nView trajectory at: {trajectory.url}")
        
        return final_result
        
    except Exception as e:
        if verbose:
            print(f"LogMD prediction failed: {e}")
            print("Falling back to standard prediction.")
        return quick_predict(sequence, msa_mode, num_recycles, verbose=verbose)


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


