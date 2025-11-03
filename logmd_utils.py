"""
LogMD Utilities for AF2 Tutorial

Provides real-time structure visualization and trajectory creation capabilities
for AlphaFold2 predictions using LogMD.

This module is adapted from AlphaMask's LogMD integration to work with the
AF2 Utils package for educational tutorials.

Author: Felipe Engelberger
Date: 2025
License: MIT
"""

import logging
import numpy as np
from typing import Optional, List, Dict, Any, Tuple
from pathlib import Path
import warnings

__version__ = "1.0.0"
__all__ = [
    'LogMDIntegration',
    'get_ca_positions',
    'kabsch_rotation',
    'superimpose_structures',
    'create_pdb_string',
    'create_trajectory_from_predictions',
]

logger = logging.getLogger(__name__)


# =============================================================================
# STRUCTURAL SUPERPOSITION UTILITIES
# =============================================================================

def get_ca_positions(atom_positions: np.ndarray) -> np.ndarray:
    """
    Extract CA atom coordinates from AlphaFold atom_positions array.
    
    Args:
        atom_positions: Array of shape (N, 37, 3) where index 1 is CA
        
    Returns:
        Array of shape (N, 3) containing CA coordinates
    """
    if atom_positions.ndim != 3 or atom_positions.shape[1] < 2:
        raise ValueError(
            f"Expected atom_positions with shape (N, 37, 3); got {atom_positions.shape}"
        )
    return atom_positions[:, 1, :].copy()


def kabsch_rotation(P: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """
    Compute optimal rotation matrix using Kabsch algorithm.
    
    Both P and Q must be centered on their centroids before calling.
    
    Args:
        P: Mobile coordinates (N, 3), centered
        Q: Reference coordinates (N, 3), centered
        
    Returns:
        Rotation matrix (3, 3)
    """
    # Covariance matrix
    C = P.T @ Q
    V, S, Wt = np.linalg.svd(C)
    d = np.sign(np.linalg.det(V @ Wt))
    D = np.diag([1.0, 1.0, d])
    U = V @ D @ Wt
    return U.astype(P.dtype, copy=False)


def superimpose_structures(
    atom_positions: np.ndarray,
    reference_ca: np.ndarray
) -> np.ndarray:
    """
    Align atom_positions onto reference_ca using Kabsch algorithm.
    
    Args:
        atom_positions: Structure to align (N, 37, 3)
        reference_ca: Reference CA coordinates (N, 3)
        
    Returns:
        Aligned atom_positions (N, 37, 3)
    """
    if atom_positions.shape[0] != reference_ca.shape[0]:
        raise ValueError(
            f"Reference and mobile structures have different residue counts "
            f"({reference_ca.shape[0]} vs {atom_positions.shape[0]})"
        )
    
    mobile_ca = get_ca_positions(atom_positions)
    
    # Center both structures
    mobile_centroid = mobile_ca.mean(axis=0)
    ref_centroid = reference_ca.mean(axis=0)
    P = mobile_ca - mobile_centroid
    Q = reference_ca - ref_centroid
    
    # Compute rotation matrix
    U = kabsch_rotation(P, Q)
    
    # Rotate all atoms
    aligned = (atom_positions - mobile_centroid) @ U + ref_centroid
    return aligned


# =============================================================================
# PDB STRING GENERATION
# =============================================================================

def create_pdb_string(
    atom_positions: np.ndarray,
    sequence: str,
    plddt: Optional[np.ndarray] = None,
    chain_id: str = 'A'
) -> str:
    """
    Generate PDB format string from atom positions.
    
    Args:
        atom_positions: Atom coordinates (N, 37, 3)
        sequence: Amino acid sequence
        plddt: Optional per-residue pLDDT scores
        chain_id: Chain identifier
        
    Returns:
        PDB format string
    """
    # Amino acid 3-letter codes
    aa_map = {
        'A': 'ALA', 'C': 'CYS', 'D': 'ASP', 'E': 'GLU', 'F': 'PHE',
        'G': 'GLY', 'H': 'HIS', 'I': 'ILE', 'K': 'LYS', 'L': 'LEU',
        'M': 'MET', 'N': 'ASN', 'P': 'PRO', 'Q': 'GLN', 'R': 'ARG',
        'S': 'SER', 'T': 'THR', 'V': 'VAL', 'W': 'TRP', 'Y': 'TYR'
    }
    
    # Atom names for standard amino acids (backbone atoms)
    atom_names = ['N', 'CA', 'C', 'O', 'CB']
    
    lines = []
    atom_serial = 1
    
    for res_idx, aa in enumerate(sequence):
        resname = aa_map.get(aa, 'UNK')
        res_num = res_idx + 1
        b_factor = plddt[res_idx] if plddt is not None else 1.0
        
        # Add CA atom (most important for visualization)
        ca_coord = atom_positions[res_idx, 1, :]
        line = (
            f"ATOM  {atom_serial:5d}  CA  {resname:3s} {chain_id}{res_num:4d}    "
            f"{ca_coord[0]:8.3f}{ca_coord[1]:8.3f}{ca_coord[2]:8.3f}"
            f"  1.00{b_factor:6.2f}           C  "
        )
        lines.append(line)
        atom_serial += 1
    
    lines.append("END")
    return "\n".join(lines)


# =============================================================================
# LOGMD INTEGRATION
# =============================================================================

class LogMDIntegration:
    """
    Integration class for LogMD visualization in AF2 tutorials.
    
    This class manages LogMD instances, handles structure superposition,
    and creates interactive 3D visualizations of predicted structures.
    """
    
    def __init__(self):
        """Initialize LogMD integration."""
        self.logmd_instances = {}
        self.LogMD = None
        self._check_logmd_availability()
    
    def _check_logmd_availability(self) -> bool:
        """Check if LogMD is available."""
        try:
            from logmd import LogMD
            self.LogMD = LogMD
            logger.info("LogMD successfully imported")
            return True
        except ImportError:
            logger.warning("LogMD not available. Install with: pip install logmd")
            self.LogMD = None
            return False
    
    def is_available(self) -> bool:
        """Check if LogMD is available for use."""
        return self.LogMD is not None
    
    def create_trajectory(
        self,
        project: str = "",
        trajectory_id: Optional[str] = None
    ) -> Optional[Any]:
        """
        Create a new LogMD trajectory.
        
        Args:
            project: Project name for organization
            trajectory_id: Optional identifier for the trajectory
            
        Returns:
            LogMD trajectory instance or None if unavailable
        """
        if not self.is_available():
            logger.warning("LogMD not available - cannot create trajectory")
            return None
        
        try:
            trajectory = self.LogMD(project=project)
            
            if trajectory_id:
                self.logmd_instances[trajectory_id] = trajectory
            
            logger.info(f"Created LogMD trajectory: {trajectory.url}")
            return trajectory
        except Exception as e:
            logger.error(f"Failed to create LogMD trajectory: {e}")
            return None
    
    def add_structure(
        self,
        trajectory: Any,
        atom_positions: np.ndarray,
        sequence: str,
        plddt: Optional[np.ndarray] = None,
        label: str = "",
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Add a structure to LogMD trajectory.
        
        Args:
            trajectory: LogMD trajectory instance
            atom_positions: Atom coordinates (N, 37, 3)
            sequence: Amino acid sequence
            plddt: Optional pLDDT scores
            label: Frame label
            metadata: Additional metadata
            
        Returns:
            True if successful, False otherwise
        """
        if trajectory is None or not self.is_available():
            return False
        
        try:
            # Generate PDB string
            pdb_string = create_pdb_string(atom_positions, sequence, plddt)
            
            # Create frame info
            frame_info = {"label": label}
            if metadata:
                frame_info.update(metadata)
            
            # Add to trajectory
            trajectory.add_frame(pdb_string, **frame_info)
            return True
        except Exception as e:
            logger.error(f"Failed to add structure to trajectory: {e}")
            return False
    
    def display_url(self, trajectory: Any, enhanced: bool = True) -> str:
        """
        Get display URL for trajectory.
        
        Args:
            trajectory: LogMD trajectory instance
            enhanced: Add enhanced visualization parameters
            
        Returns:
            URL string
        """
        if trajectory is None:
            return ""
        
        url = trajectory.url
        
        if enhanced and url:
            # Add visualization parameters
            params = [
                "fps=10",
                "preset=polymer-cartoon",
                "plddt",
                "label=conf"
            ]
            separator = "&" if "?" in url else "?"
            url = url + separator + "&".join(params)
        
        return url


# =============================================================================
# HIGH-LEVEL TRAJECTORY CREATION
# =============================================================================

def create_trajectory_from_predictions(
    predictions: List[Dict[str, Any]],
    sequence: str,
    project: str = "AF2_Tutorial",
    align_structures: bool = True,
    sort_by_rmsd: bool = False,
    reference_coords: Optional[np.ndarray] = None,
    max_structures: Optional[int] = None
) -> Optional[Any]:
    """
    Create LogMD trajectory from list of predictions.
    
    Args:
        predictions: List of prediction dictionaries with 'structure' and 'plddt'
        sequence: Amino acid sequence
        project: LogMD project name
        align_structures: Align all structures to first frame
        sort_by_rmsd: Sort structures by RMSD to reference
        reference_coords: Reference CA coordinates for sorting
        max_structures: Maximum number of structures to include
        
    Returns:
        LogMD trajectory instance or None if unavailable
    """
    integration = LogMDIntegration()
    
    if not integration.is_available():
        logger.warning("LogMD not available - returning None")
        return None
    
    # Create trajectory
    trajectory = integration.create_trajectory(project=project)
    if trajectory is None:
        return None
    
    # Sort by RMSD if requested
    if sort_by_rmsd and reference_coords is not None:
        from af2_utils import calculate_rmsd
        rmsd_values = []
        for pred in predictions:
            ca_coords = pred['structure'][:, 1, :]
            rmsd = calculate_rmsd(ca_coords, reference_coords)
            rmsd_values.append(rmsd)
        
        # Sort predictions by RMSD
        sorted_indices = np.argsort(rmsd_values)
        predictions = [predictions[i] for i in sorted_indices]
    
    # Limit number of structures
    if max_structures is not None:
        predictions = predictions[:max_structures]
    
    # Get reference for alignment
    reference_ca = None
    if align_structures and len(predictions) > 0:
        reference_ca = get_ca_positions(predictions[0]['structure'])
    
    # Add structures to trajectory
    for i, pred in enumerate(predictions):
        atom_positions = pred['structure']
        
        # Align if requested
        if align_structures and reference_ca is not None:
            atom_positions = superimpose_structures(atom_positions, reference_ca)
        
        # Create metadata
        metadata = {
            'frame': i + 1,
            'mean_plddt': float(pred.get('plddt', np.array([0])).mean())
        }
        
        # Add seed and model info if available
        if 'seed' in pred:
            metadata['seed'] = pred['seed']
        if 'msa_mode' in pred:
            metadata['msa_mode'] = pred['msa_mode']
        if 'ensemble_idx' in pred:
            metadata['ensemble_idx'] = pred['ensemble_idx']
        
        # Create label
        label = f"Frame {i+1}"
        if 'msa_mode' in pred:
            label += f" ({pred['msa_mode']})"
        if 'seed' in pred:
            label += f" seed={pred['seed']}"
        
        # Add to trajectory
        integration.add_structure(
            trajectory,
            atom_positions,
            sequence,
            plddt=pred.get('plddt'),
            label=label,
            metadata=metadata
        )
    
    logger.info(f"Created trajectory with {len(predictions)} frames")
    logger.info(f"View at: {integration.display_url(trajectory)}")
    
    return trajectory


# =============================================================================
# NOTEBOOK DISPLAY HELPERS
# =============================================================================

def display_trajectory_in_notebook(trajectory: Any) -> None:
    """
    Display LogMD trajectory in Jupyter notebook.
    
    Args:
        trajectory: LogMD trajectory instance
    """
    if trajectory is None:
        print("No trajectory to display (LogMD not available)")
        return
    
    try:
        from IPython.display import display, HTML
        
        integration = LogMDIntegration()
        url = integration.display_url(trajectory, enhanced=True)
        
        # Create iframe for embedding
        iframe_html = f'''
        <div style="margin: 20px 0;">
            <h3>Interactive 3D Structure Viewer</h3>
            <iframe 
                src="{url}" 
                width="100%" 
                height="600px" 
                frameborder="0"
                style="border: 1px solid #ccc; border-radius: 5px;">
            </iframe>
            <p style="margin-top: 10px;">
                <a href="{url}" target="_blank">Open in new window</a>
            </p>
        </div>
        '''
        
        display(HTML(iframe_html))
        print(f"\nTrajectory URL: {url}")
    except ImportError:
        print(f"IPython not available. View trajectory at: {trajectory.url}")
    except Exception as e:
        logger.error(f"Failed to display trajectory: {e}")
        print(f"View trajectory at: {trajectory.url}")


# =============================================================================
# MODULE INITIALIZATION
# =============================================================================

if __name__ != "__main__":
    print(f"LogMD Utils v{__version__} loaded")
    integration = LogMDIntegration()
    if integration.is_available():
        print("  - LogMD: available")
    else:
        print("  - LogMD: not available (install with: pip install logmd)")

