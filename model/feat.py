import pandas as pd
import torch, os
import numpy as np
import pandas as pd
import freesasa as fs
import torch.nn.functional as F
import warnings
import contextlib
import sys
from io import StringIO
try:
    from torch_geometric.data import Data
except Exception:
    Data = None

amino_acid_mapping = {'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
                      'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
                      'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
                      'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V',}

amino_acid_mapping_reverse = {v: k for k, v in amino_acid_mapping.items()}
amino_acid_1_to_int = { k: i for i, k in enumerate(sorted(amino_acid_mapping_reverse.keys())) }
amino_acid_3_to_int = { amino_acid_mapping_reverse[k]: i for i, k in enumerate( sorted( amino_acid_mapping_reverse.keys() ) ) }

amino_acid_1_to_int['X'] = 20
amino_acid_3_to_int['UNK'] = 20


@contextlib.contextmanager
def suppress_freesasa_warnings():
    """
    Context manager to suppress FreeSASA warnings about unknown atoms.
    Captures both Python warnings and stderr output.
    """
    # Suppress Python warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        # Capture stderr to suppress FreeSASA warnings
        old_stderr = sys.stderr
        sys.stderr = StringIO()
        
        try:
            yield
        finally:
            sys.stderr = old_stderr


STANDARD_ATOMS = {
    'ALA': ['N', 'CA', 'C', 'O', 'CB'],
    'ARG': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'NE', 'CZ', 'NH1', 'NH2'],
    'ASN': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'OD1', 'ND2'],
    'ASP': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'OD1', 'OD2'],
    'CYS': ['N', 'CA', 'C', 'O', 'CB', 'SG'],
    'GLN': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'OE1', 'NE2'],
    'GLU': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'OE1', 'OE2'],
    'GLY': ['N', 'CA', 'C', 'O'],
    'HIS': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'ND1', 'CD2', 'CE1', 'NE2'],
    'ILE': ['N', 'CA', 'C', 'O', 'CB', 'CG1', 'CG2', 'CD1'],
    'LEU': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD1', 'CD2'],
    'LYS': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'CE', 'NZ'],
    'MET': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'SD', 'CE'],
    'PHE': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ'],
    'PRO': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD'],
    'SER': ['N', 'CA', 'C', 'O', 'CB', 'OG'],
    'THR': ['N', 'CA', 'C', 'O', 'CB', 'OG1', 'CG2'],
    'TRP': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD1', 'CD2', 'NE1', 'CE2', 'CE3', 'CZ2', 'CZ3', 'CH2'],
    'TYR': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ', 'OH'],
    'VAL': ['N', 'CA', 'C', 'O', 'CB', 'CG1', 'CG2'],
    'UNK': ['N', 'CA', 'C', 'O', 'CB']
}

nucleic_acid_residues = {
        'DA', 'DT', 'DG', 'DC', 'DI', 'DU',
        'A', 'U', 'G', 'C', 'I',
        'ADE', 'THY', 'GUA', 'CYT', 'URA',
        '1MA', '2MG', '4SU', '5MC', '5MU', 'PSU', 'H2U', 'M2G', 'OMC', 'OMG'
    }

def standardize_pdb_file(input_pdb_path, output_pdb_path, remove_hydrogens=True):
    with open(input_pdb_path, 'r') as f:
        lines = f.readlines()
    
    new_lines = []

    for line in lines:
        if line.startswith('ATOM') or line.startswith('HETATM'):
            start_line = line[:6]  # ATOM or HETATM
            atom_num = line[6:11]
            atom_name =  line[12:16]
            res_name = line[17:20]
            chain_id = line[21:22]
            res_num = line[22:26]
            insertion_code = line[26:27]
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
            occupancy = line[54:60]
            temp_factor = line[60:66]
            segment_id = line[72:76]
            element = line[76:78]
            charge = line[78:80]

            if res_name in nucleic_acid_residues or res_name in ['HOH']:
                continue

            if occupancy.strip() == '':
                occupancy = '1.00'
            if temp_factor.strip() == '':
                temp_factor = '0.00'

            if remove_hydrogens and (atom_name.startswith('H') or element.upper() == 'H'):
                continue

            new_lines.append(f"{start_line}{atom_num:>5s}{atom_name:>4s}{res_name:>3s}{chain_id:>1s}{res_num:>4s}{insertion_code:>1s}{x:>8.3f}{y:>8.3f}{z:>8.3f}{occupancy:>6.2f}{temp_factor:>6.2f}{segment_id:>4s}{element:>2s}{charge:>2s}\n")

    with open(output_pdb_path, 'w') as f:
        f.writelines(new_lines)

    return output_pdb_path




            
def standardize_pdb(input_pdb_path, output_pdb_path, remove_hydrogens=True):
    nucleic_acid_residues = {
        'DA', 'DT', 'DG', 'DC', 'DI', 'DU',
        'A', 'U', 'G', 'C', 'I',
        'ADE', 'THY', 'GUA', 'CYT', 'URA',
        '1MA', '2MG', '4SU', '5MC', '5MU', 'PSU', 'H2U', 'M2G', 'OMC', 'OMG'
    }
    
    element_symbols = {
        'H', 'C', 'N', 'O', 'S', 'P', 'F', 'CL', 'BR', 'I',
        'FE', 'ZN', 'MG', 'CA', 'MN', 'CU', 'NI', 'CO', 'K', 'NA'
    }
    
    def get_element_symbol(atom_name, residue_name):
        atom_name = atom_name.strip()
        
        if atom_name.startswith('C'):
            return 'C'
        elif atom_name.startswith('N'):
            return 'N'
        elif atom_name.startswith('O'):
            return 'O'
        elif atom_name.startswith('S'):
            return 'S'
        elif atom_name.startswith('P'):
            return 'P'
        elif atom_name.startswith('H'):
            return 'H'
        elif atom_name.startswith('F'):
            return 'F'
        
        true_2char_elements = {'CL', 'BR', 'FE', 'ZN', 'MG', 'MN', 'CU', 'NI', 'CO'}
        if len(atom_name) >= 2:
            two_char = atom_name[:2].upper()
            if two_char in true_2char_elements:
                return two_char
        
        if atom_name.upper() == 'CA' and residue_name not in ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL']:
            return 'CA'
        
        return atom_name[0].upper()
    
    def format_atom_name(atom_name):
        atom_name = atom_name.strip()
        
        true_2char_elements = {'CL', 'BR', 'FE', 'ZN', 'MG', 'MN', 'CU', 'NI', 'CO'}
        
        if len(atom_name) >= 2 and atom_name[:2].upper() in true_2char_elements:
            return f"{atom_name:<4s}"
        
        else:
            return f" {atom_name:<3s}"
    
    def format_coordinates(x, y, z):
        """Format coordinates to PDB specification: 8.3 format."""
        return f"{x:8.3f}{y:8.3f}{z:8.3f}"
    
    def format_occupancy_bfactor(occupancy, temp_factor):
        """Format occupancy and B-factor to PDB specification: 6.2 format."""
        try:
            occ = float(occupancy) if occupancy.strip() else 1.00
            bf = float(temp_factor) if temp_factor.strip() else 0.00
        except ValueError:
            occ, bf = 1.00, 0.00
        
        return f"{occ:6.2f}{bf:6.2f}"
    
    output_dir = os.path.dirname(output_pdb_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    with open(input_pdb_path, 'r') as f:
        lines = f.readlines()
    
    protein_residues = {}
    hetatm_residues = {}
    
    # Parse input file
    for line in lines:
        if line.startswith('ATOM') or line.startswith('HETATM'):
            try:
                atom_name = line[12:16].strip()
                res_name = line[17:20].strip()
                chain_id = line[21] if len(line) > 21 else ' '
                res_num_str = line[22:27].strip()  # Include insertion code
                
                # Get element symbol from existing PDB or derive it
                element = line[76:78].strip() if len(line) > 76 else get_element_symbol(atom_name, res_name)
                
                # Filter hydrogens if requested
                if remove_hydrogens and (atom_name.startswith('H') or element.upper() == 'H'):
                    continue
                
                # Skip water molecules
                if res_name in ['HOH', 'WAT']:
                    continue
                    
                # Skip DNA and RNA residues
                if res_name in nucleic_acid_residues:
                    continue
                
                residue_key = (chain_id, res_num_str, res_name)
                
                if line.startswith('ATOM'):
                    if residue_key not in protein_residues:
                        protein_residues[residue_key] = {}
                    protein_residues[residue_key][atom_name] = line
                elif line.startswith('HETATM'):
                    if residue_key not in hetatm_residues:
                        hetatm_residues[residue_key] = {}
                    hetatm_residues[residue_key][atom_name] = line
                    
            except (ValueError, IndexError):
                continue  # Skip malformed lines
    
    standardized_lines = []
    atom_counter = 1
    
    # Sort residues by chain and residue number (handling insertion codes)
    def sort_key(residue_key):
        chain_id, res_num_str, res_name = residue_key
        # Extract numeric part and insertion code
        numeric_part = ''.join(filter(str.isdigit, res_num_str))
        res_num = int(numeric_part) if numeric_part else 0
        insertion_code = ''.join(filter(str.isalpha, res_num_str))
        return (chain_id, res_num, insertion_code)
    
    # Group protein residues by chain
    protein_by_chain = {}
    for residue_key in protein_residues.keys():
        chain_id = residue_key[0]
        if chain_id not in protein_by_chain:
            protein_by_chain[chain_id] = []
        protein_by_chain[chain_id].append(residue_key)
    
    # Process protein residues chain by chain
    for chain_id in sorted(protein_by_chain.keys()):
        sorted_residues = sorted(protein_by_chain[chain_id], key=sort_key)
        
        for residue_key in sorted_residues:
            chain_id, res_num_str, res_name = residue_key
            residue_atoms = protein_residues[residue_key]
            
            # Process atoms in standard order if available, otherwise alphabetically
            if res_name in STANDARD_ATOMS:
                atom_order = STANDARD_ATOMS[res_name]
            else:
                atom_order = sorted(residue_atoms.keys())
            
            for atom_name in atom_order:
                if atom_name in residue_atoms:
                    line = residue_atoms[atom_name]
                    
                    try:
                        # Extract coordinates and other data
                        x = float(line[30:38])
                        y = float(line[38:46])
                        z = float(line[46:54])
                        occupancy = line[54:60].strip() if len(line) > 54 else "1.00"
                        temp_factor = line[60:66].strip() if len(line) > 60 else "0.00"
                        element = line[76:78].strip() if len(line) > 76 else get_element_symbol(atom_name, res_name)
                        
                        # Format components
                        formatted_atom_name = format_atom_name(atom_name)
                        coord_str = format_coordinates(x, y, z)
                        occ_bf_str = format_occupancy_bfactor(occupancy, temp_factor)
                        
                        # Create properly formatted ATOM line
                        new_line = (f"ATOM  {atom_counter:5d} {formatted_atom_name} {res_name} {chain_id}{res_num_str:>4s}   "
                                   f"{coord_str}{occ_bf_str}          {element:>2s}\n")
                        
                        standardized_lines.append(new_line)
                        atom_counter += 1
                        
                    except (ValueError, IndexError):
                        continue  # Skip malformed atom lines
        
        # Add TER record at the end of each protein chain
        if sorted_residues:
            last_residue = sorted_residues[-1]
            chain_id = last_residue[0]
            last_res_num = last_residue[1]
            ter_line = f"TER   {atom_counter:5d}      {last_residue[2]} {chain_id}{last_res_num:>4s}\n"
            standardized_lines.append(ter_line)
            atom_counter += 1
    
    # Group HETATM residues by chain
    hetatm_by_chain = {}
    for residue_key in hetatm_residues.keys():
        chain_id = residue_key[0]
        if chain_id not in hetatm_by_chain:
            hetatm_by_chain[chain_id] = []
        hetatm_by_chain[chain_id].append(residue_key)
    
    # Process HETATM residues chain by chain
    for chain_id in sorted(hetatm_by_chain.keys()):
        sorted_residues = sorted(hetatm_by_chain[chain_id], key=sort_key)
        
        for residue_key in sorted_residues:
            chain_id, res_num_str, res_name = residue_key
            residue_atoms = hetatm_residues[residue_key]
            
            for atom_name in sorted(residue_atoms.keys()):
                line = residue_atoms[atom_name]
                
                try:
                    # Extract coordinates and other data
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    occupancy = line[54:60].strip() if len(line) > 54 else "1.00"
                    temp_factor = line[60:66].strip() if len(line) > 60 else "0.00"
                    element = line[76:78].strip() if len(line) > 76 else get_element_symbol(atom_name, res_name)
                    
                    # Format components
                    formatted_atom_name = format_atom_name(atom_name)
                    coord_str = format_coordinates(x, y, z)
                    occ_bf_str = format_occupancy_bfactor(occupancy, temp_factor)
                    
                    # Create properly formatted HETATM line
                    new_line = (f"HETATM{atom_counter:5d} {formatted_atom_name} {res_name} {chain_id}{res_num_str:>4s}   "
                               f"{coord_str}{occ_bf_str}          {element:>2s}\n")
                    
                    standardized_lines.append(new_line)
                    atom_counter += 1
                    
                except (ValueError, IndexError):
                    continue  # Skip malformed atom lines
    
    # Add END record
    standardized_lines.append("END\n")
    
    # Write standardized PDB
    with open(output_pdb_path, 'w') as f:
        f.writelines(standardized_lines)
    
    return output_pdb_path

class PDBParserSim:
    def __init__(self, pdb):
        self.pdb = pdb
        self.protein_indices, self.hetero_indices, self.protein_atom_info, self.hetero_atom_info = self._get_data_from_pdb(pdb)

    def _get_data_from_pdb(self, pdb):
        with open(pdb, 'r') as f:
            lines = f.read().split('\n')

        protein_index, hetero_index = [], []
        protein_data,  hetero_data = { 'coord': [] }, { 'coord': [] }

        for line in lines:
            record_type = line[:6].strip()  # Changed to handle HETATM properly
            if record_type in ['ATOM', 'HETATM'] and len(line) > 13 and line[13] != 'H' and line[17:20].strip() != 'HOH':
                atom_type = line[12:17].strip()
                res_type  = amino_acid_3_to_int.get( line[17:20].strip(), 20 )
                chain_id  = line[21]
                res_num   = int( line[22:26] )
                # res_num   = line[22:27].strip()
                try:
                    xyz = [ float( line[idx:idx + 8] ) for idx in range(30, 54, 8) ]
                except ValueError:
                    # Skip malformed coordinate lines (e.g., missing column spacing)
                    continue

                if record_type == 'ATOM':
                    protein_index.append( (chain_id, res_num, res_type, atom_type ) )
                    protein_data['coord'].append( xyz )

                elif record_type == 'HETATM':
                    hetero_index.append( ('HETERO', res_num, res_type, atom_type ) )
                    hetero_data['coord'].append( xyz )

        protein_index = pd.MultiIndex.from_tuples(protein_index, names=['chain', 'res_num', 'AA', 'atom'])
        hetero_index  = pd.MultiIndex.from_tuples(hetero_index,  names=['chain', 'res_num', 'AA', 'atom'])

        protein_atom_info = pd.DataFrame(protein_data, index=protein_index)
        hetero_atom_info  = pd.DataFrame(hetero_data,  index=hetero_index)

        return (protein_index, hetero_index, protein_atom_info, hetero_atom_info)

    def get_atom(self, prot=True, hetero=False):
        if prot and hetero:
            return list( self.protein_indices ) + list( self.hetero_indices )
        elif prot:
            return list( self.protein_indices )
        elif hetero:
            return list( self.hetero_indices )

    def get_all_atom_info(self, prot=True, hetero=False):
        if prot and hetero:
            return pd.concat( (self.protein_atom_info, self.hetero_atom_info) )
        elif prot:
            return self.protein_atom_info
        elif hetero:
            return self.hetero_atom_info

    def get_residue(self):
        return sorted( set( list( [ (chain, num, res) for chain, num, res, atom in self.protein_indices ] ) ) )

    def get_residue_coord(self, index):
        return self.get_all_atom_info(prot=True, hetero=False).coord.xs(index)

    def get_atom_with_name(self, name):
        return sorted( set( list( [ (chain, num, res) for chain, num, res, atom in self.protein_indices if atom == name ]  ) ) )
    
    def get_terminal_flag(self):
        """
        Get terminal flags for each residue.
        Returns N-terminal and C-terminal flags separately.
        """
        # Get unique residues (chain, res_num, res_type)
        residues = self.get_residue()
        
        n_terminal = []
        c_terminal = []
        
        # Group residues by chain
        chain_residues = {}
        for chain, num, res in residues:
            
            if chain not in chain_residues:
                chain_residues[chain] = []
            chain_residues[chain].append((num, res))

        # Sort residues within each chain by residue number
        for chain in chain_residues:
            chain_residues[chain].sort(key=lambda x: x[0])

        for chain, num, res in residues:
            chain_res_list = chain_residues[chain]
            
            # Check if this is N-terminal (first residue in chain)
            is_n_terminal = (num, res) == chain_res_list[0]
            # Check if this is C-terminal (last residue in chain)
            is_c_terminal = (num, res) == chain_res_list[-1]
            
            n_terminal.append(is_n_terminal)
            c_terminal.append(is_c_terminal)
        
        return torch.tensor(n_terminal, dtype=torch.bool), torch.tensor(c_terminal, dtype=torch.bool)

    def get_relative_position(self, cutoff=32, onehot=True):
        """
        Get relative position of each residue within each chain.
        Returns a tensor where same chain residues have relative positions and different chains have -1.
        """
        residues = self.get_residue()
        num_residues = len(residues)
        
        # Initialize with -1 (for different chains)
        relative_positions = torch.ones((num_residues, num_residues)) * 33
        
        # Group residues by chain with their indices
        chain_residue_indices = {}
        for idx, (chain, num, res) in enumerate(residues):
            if chain not in chain_residue_indices:
                chain_residue_indices[chain] = []
            chain_residue_indices[chain].append(idx)
        
        # Calculate relative positions within each chain
        for chain, indices in chain_residue_indices.items():
            num_chain_residues = len(indices)
            arrange = torch.arange(num_chain_residues)
            chain_relative_positions = (arrange[:, None] - arrange[None, :]).abs()
            chain_relative_positions = torch.where(chain_relative_positions > cutoff, 33, chain_relative_positions)
            
            # Fill in the relative positions for same chain residues
            for i, idx_i in enumerate(indices):
                for j, idx_j in enumerate(indices):
                    relative_positions[idx_i, idx_j] = chain_relative_positions[i, j]

        if onehot:
            relative_positions = relative_positions.long()
            relative_positions = F.one_hot(relative_positions, num_classes=cutoff + 2)
            relative_positions = relative_positions.float()

        return relative_positions

    def _dihedral(self, X, eps=1e-8):
        shape_X = X.shape
        X = X.reshape( shape_X[0] * shape_X[1], shape_X[2] )

        U = F.normalize( X[1:, :] - X[:-1,:], dim=-1)
        u_2 = U[:-2,:]
        u_1 = U[1:-1,:]
        u_0 = U[2:,:]

        n_2 = F.normalize(torch.cross(u_2, u_1, dim=1), dim=-1)
        n_1 = F.normalize(torch.cross(u_1, u_0, dim=1), dim=-1)

        cosD = (n_2 * n_1).sum(-1)
        cosD = torch.clamp(cosD, -1+eps, 1-eps)

        D = torch.sign((u_2 * n_1).sum(-1)) * torch.acos(cosD)
        D = F.pad(D, (1,2), 'constant', 0)

        return D.view( (int(D.size(0)/shape_X[1]), shape_X[1] ) )

    def get_SASA(self):
        # Suppress FreeSASA warnings about unknown atoms
        with suppress_freesasa_warnings():
            sasas = [
                [
                    values.total / 350,
                    values.polar / 350,
                    values.apolar / 350,
                    values.mainChain / 350,
                    values.sideChain / 350,
                    values.relativeTotal,
                    values.relativePolar,
                    values.relativeApolar,
                    values.relativeMainChain,
                    values.relativeSideChain
                ]
                for chain, residues in fs.calc( fs.Structure( self.pdb ) ).residueAreas().items()
                for residue, values in residues.items()
            ]

        return torch.nan_to_num( torch.as_tensor( sasas ) )

    def get_backbone_curvature(self, ca_coords, terminal_flags, eps=1e-8):
        """
        Calculate backbone curvature from CA coordinates.
        Curvature is the angle between vectors formed by 3 consecutive CA atoms.

        Args:
            ca_coords: (num_residues, 3) CA coordinates
            terminal_flags: tuple of (n_terminal, c_terminal) boolean tensors
            eps: small value for numerical stability

        Returns:
            curvature: (num_residues, 1) tensor of curvature angles in radians
        """
        num_residues = ca_coords.shape[0]
        curvature = torch.zeros(num_residues, 1)

        if num_residues < 3:
            return curvature

        n_terminal, c_terminal = terminal_flags

        # For each residue i, compute vectors to neighbors
        # v1 = CA[i-1] - CA[i], v2 = CA[i+1] - CA[i]
        for i in range(1, num_residues - 1):
            # Skip if at chain boundary
            if n_terminal[i] or c_terminal[i]:
                continue
            if c_terminal[i-1] or n_terminal[i+1]:
                continue

            v1 = ca_coords[i-1] - ca_coords[i]
            v2 = ca_coords[i+1] - ca_coords[i]

            # Normalize vectors
            v1_norm = F.normalize(v1.unsqueeze(0), dim=-1).squeeze(0)
            v2_norm = F.normalize(v2.unsqueeze(0), dim=-1).squeeze(0)

            # Compute angle (curvature)
            cos_angle = torch.clamp((v1_norm * v2_norm).sum(), -1 + eps, 1 - eps)
            angle = torch.acos(cos_angle)
            curvature[i, 0] = angle

        return torch.nan_to_num(curvature)

    def get_backbone_torsion(self, ca_coords, terminal_flags, eps=1e-8):
        """
        Calculate backbone torsion from CA coordinates.
        Torsion is the dihedral angle formed by 4 consecutive CA atoms.

        Args:
            ca_coords: (num_residues, 3) CA coordinates
            terminal_flags: tuple of (n_terminal, c_terminal) boolean tensors
            eps: small value for numerical stability

        Returns:
            torsion: (num_residues, 1) tensor of torsion angles in radians
        """
        num_residues = ca_coords.shape[0]
        torsion = torch.zeros(num_residues, 1)

        if num_residues < 4:
            return torsion

        n_terminal, c_terminal = terminal_flags

        # For each residue i, compute torsion using CA[i-1], CA[i], CA[i+1], CA[i+2]
        for i in range(1, num_residues - 2):
            # Skip if at chain boundary
            if n_terminal[i] or c_terminal[i] or c_terminal[i+1]:
                continue
            if c_terminal[i-1] or n_terminal[i+1] or n_terminal[i+2]:
                continue

            # Get 4 consecutive CA positions
            p0 = ca_coords[i-1]
            p1 = ca_coords[i]
            p2 = ca_coords[i+1]
            p3 = ca_coords[i+2]

            # Compute bond vectors
            b1 = p1 - p0
            b2 = p2 - p1
            b3 = p3 - p2

            # Compute normal vectors
            n1 = torch.linalg.cross(b1, b2)
            n2 = torch.linalg.cross(b2, b3)

            # Normalize
            n1 = F.normalize(n1.unsqueeze(0), dim=-1).squeeze(0)
            n2 = F.normalize(n2.unsqueeze(0), dim=-1).squeeze(0)
            b2_norm = F.normalize(b2.unsqueeze(0), dim=-1).squeeze(0)

            # Compute torsion angle using atan2
            m1 = torch.linalg.cross(n1, b2_norm)
            x = (n1 * n2).sum()
            y = (m1 * n2).sum()
            angle = torch.atan2(y, x)
            torsion[i, 0] = angle

        return torch.nan_to_num(torsion)

    def get_curvature_torsion(self, coords):
        """
        Extract curvature and torsion features from coordinates.

        Args:
            coords: (num_residues, 15, 3) full atom coordinates

        Returns:
            features: (num_residues, 4) tensor [cos(curv), sin(curv), cos(tors), sin(tors)]
        """
        # Get CA coordinates (index 1)
        ca_coords = coords[:, 1, :]

        # Get terminal flags
        terminal_flags = self.get_terminal_flag()

        # Calculate curvature and torsion
        curvature = self.get_backbone_curvature(ca_coords, terminal_flags)
        torsion = self.get_backbone_torsion(ca_coords, terminal_flags)

        # Convert to sin/cos representation
        features = torch.cat([
            torch.cos(curvature),
            torch.sin(curvature),
            torch.cos(torsion),
            torch.sin(torsion)
        ], dim=1)

        return features

    def get_dihedral_angle(self, coords, res_type):
        """
        coords: (num_residues, 15, 3)
        res_type: (num_residues, )

        chi1: N A B G == [0, 1, 4, 5]
        chi2: A B G D == [1, 4, 5, 6]
        chi3: B G D E == [4, 5, 6, 7]
        chi4: G D E Z == [5, 6, 7, 8]
        chi5: D E Z H == [6, 7, 8, 9]
        return: (num_residues, 8[:3 == bb, 3: == sc]), (num_residues, 5)
        """ 

        chi_indices = {
            'chi1': torch.tensor([1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]),
            'chi2': torch.tensor([2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 18, 19]),
            'chi3': torch.tensor([3, 8, 10, 13, 14]),
            'chi4': torch.tensor([8, 14]),
            'chi5': torch.tensor([14])
        }

        is_ILE = torch.isin(res_type, torch.tensor([7])).int().unsqueeze(1).unsqueeze(2)
        is_not_ILE = 1 - is_ILE

        has_chi = torch.stack([torch.isin(res_type, chi_indices[f'chi{i}']).int() for i in range(1, 6)], dim=1)
        
        N_CA_C = coords[:, :3, :]
        backbone_dihedrals = self._dihedral(N_CA_C)

        N_A_B_G_D_E_Z_ILE = torch.cat( [coords[:, :2, :], coords[:, 4:6, :], coords[:, 7:11, :]], dim=1 ) * is_ILE
        N_A_B_G_D_E_Z_no_ILE = torch.cat( [coords[:, :2, :], coords[:, 4:10, :]], dim=1 ) * is_not_ILE

        N_A_B_G_D_E_Z = N_A_B_G_D_E_Z_ILE + N_A_B_G_D_E_Z_no_ILE
        
        side_chain_dihedrals = self._dihedral(N_A_B_G_D_E_Z)[:, 1:-2] * has_chi
        
        dihedrals = torch.cat( [backbone_dihedrals, side_chain_dihedrals], dim=1 )

        return dihedrals, has_chi

    def _self_distance(self, coords):
        coords = torch.cat( [ coords[:, :4, :], coords[:, -1:, :] ], dim=1 )
        distance = torch.cdist( coords, coords )
        mask_sca = torch.triu( torch.ones_like( distance ), diagonal=1 ).bool()
        distance = torch.masked_select( distance, mask_sca ).view( distance.shape[0], -1 )
        
        return torch.nan_to_num(distance)

    def _self_vector(self, coords):
        coords = torch.cat( [ coords[:, :4, :], coords[:, -1:, :] ], dim=1 )
        vectors = coords[:, None] - coords[:, :, None]
        vectors = vectors.view( coords.shape[0], 25, 3 )
        vectors = torch.index_select(vectors, 1, torch.tensor([ 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 19, 20, 21, 22, 23]))

        return torch.nan_to_num( vectors )

    def _forward_reverse(self, coord, terminal_flags):
        ca_coords = coord[:, 1, :]  # CA coordinates
        sc_coords = coord[:, -1, :] # SC coordinates
        
        n_terminal, c_terminal = terminal_flags
        
        forward_vector = torch.zeros(coord.shape[0], 4, 3)
        forward_distance = torch.zeros(coord.shape[0], 4)
        reverse_vector = torch.zeros(coord.shape[0], 4, 3)
        reverse_distance = torch.zeros(coord.shape[0], 4)
        
        if coord.shape[0] > 1:
            # Forward vectors (current to next residue)
            ca_diff = ca_coords[1:] - ca_coords[:-1]
            sc_diff = sc_coords[1:] - sc_coords[:-1]
            ca_sc_diff = sc_coords[1:] - ca_coords[:-1]
            sc_ca_diff = ca_coords[1:] - sc_coords[:-1]
            
            forward_vector[:-1] = torch.stack([ca_diff, sc_diff, ca_sc_diff, sc_ca_diff], dim=1)
            forward_distance[:-1] = torch.norm(forward_vector[:-1], dim=-1)
            
            # Mask out forward vectors for C-terminal residues
            c_mask = ~c_terminal[:-1]
            forward_vector[:-1] *= c_mask[:, None, None]
            forward_distance[:-1] *= c_mask[:, None]
            
            # Reverse vectors (current to previous residue)
            reverse_vector[1:] = torch.stack([-ca_diff, -sc_diff, ca_coords[:-1] - sc_coords[1:], sc_coords[:-1] - ca_coords[1:]], dim=1)
            reverse_distance[1:] = torch.norm(reverse_vector[1:], dim=-1)
            
            # Mask out reverse vectors for N-terminal residues
            n_mask = (~n_terminal[1:])
            reverse_vector[1:] *= n_mask[:, None, None]
            reverse_distance[1:] *= n_mask[:, None]

        forward_vector = torch.nan_to_num(forward_vector)
        reverse_vector = torch.nan_to_num(reverse_vector)
        forward_distance = torch.nan_to_num(forward_distance)
        reverse_distance = torch.nan_to_num(reverse_distance)

        return forward_vector, forward_distance, reverse_vector, reverse_distance

    def _interaction_distance(self, coords, cutoff=8):
        coord_CA = coords[:, 1:2, :].transpose(0, 1)
        coord_SC = coords[:, -1:, :].transpose(0, 1)
        mask = ( 1 - torch.eye( coords.shape[0] ) ).int()

        dm_CA_CA = torch.cdist( coord_CA, coord_CA )[0]
        dm_SC_SC = torch.cdist( coord_SC, coord_SC )[0]
        dm_CA_SC = torch.cdist( coord_CA, coord_SC )[0]
        dm_SC_CA = torch.cdist( coord_SC, coord_CA )[0]

        adj_CA_CA = (dm_CA_CA < cutoff) * mask
        adj_SC_SC = (dm_SC_SC < cutoff) * mask
        adj_CA_SC = (dm_CA_SC < cutoff) * mask
        adj_SC_CA = (dm_SC_CA < cutoff) * mask

        adj = adj_CA_CA | adj_SC_SC | adj_CA_SC | adj_SC_CA

        dm_all = torch.stack( (dm_CA_CA, dm_SC_SC, dm_CA_SC, dm_SC_CA), dim=-1 )
        dm_select = dm_all * adj[:, :, None]

        return torch.nan_to_num(dm_select), adj

    def _interaction_vectors(self, coords, adj):
        coord_CA_SC = torch.cat( [coords[:, 1:2, :], coords[:, -1:, :]], dim=1 )
        coord_SC_CA = torch.cat( [coords[:, -1:, :], coords[:, 1:2, :]], dim=1 )

        vector1 = coord_CA_SC[:, None, :] - coord_CA_SC[:, :, :]
        vector3 = coord_CA_SC[:, None, :] - coord_SC_CA[:, :, :]
        vectors = torch.cat( [vector1, -vector1, vector3, -vector3], dim=2).nan_to_num()
        vectors = vectors * adj[:, :, None, None]

        return vectors

    def _residue_features(self, coords, residue_types):
        residue_one_hot = F.one_hot(residue_types, num_classes=21)

        self_distance = self._self_distance(coords)
        self_vector = self._self_vector(coords)

        dihedrals, has_chi_angles = self.get_dihedral_angle(coords, residue_types)
        dihedrals = torch.cat([torch.cos(dihedrals), torch.sin(dihedrals)], dim=-1)
        
        sasa = self.get_SASA()

        terminal_flags = self.get_terminal_flag()

        forward_vector, forward_distance, reverse_vector, reverse_distance = self._forward_reverse(coords, terminal_flags)

        rf_vector = torch.cat([forward_vector, reverse_vector], dim=1) # 8
        rf_distance = torch.cat([forward_distance, reverse_distance], dim=1)

        node_scalar_features = (
            residue_one_hot, 
            self_distance, 
            dihedrals, 
            has_chi_angles, 
            sasa, 
            rf_distance, 
            torch.stack(terminal_flags, dim=1)
            )
        
        node_vector_features = (
            self_vector,
            rf_vector,
            )

        return node_scalar_features, node_vector_features

    def _interaction_features(self, coords, distance_cutoff=8, relative_position_cutoff=32):
        relative_position = self.get_relative_position(cutoff=relative_position_cutoff, onehot=True)
        distance_adj, adj = self._interaction_distance(coords, cutoff=distance_cutoff)
        interaction_vectors = self._interaction_vectors(coords, adj)

        sparse = distance_adj.to_sparse(sparse_dim=2)
        src, dst = sparse.indices()
        distance = sparse.values()
            
        relative_position = relative_position[src, dst]
        vectors = interaction_vectors[src, dst, :]

        edges = (src, dst)
        edge_scalar_features = (distance, relative_position)
        edge_vector_features = (vectors, )
        
        return edges, edge_scalar_features, edge_vector_features

    def get_features(self):
        residues = self.get_residue()
        coords = torch.zeros( len(residues), 15, 3 )
        residue_types = torch.from_numpy( np.array( residues )[:, 2].astype(int) )

        for idx, residue in enumerate( residues ):
            residue_coord = torch.as_tensor( self.get_residue_coord(residue).tolist() )
            coords[idx, :residue_coord.shape[0], :] = residue_coord
            coords[idx, -1, :] = residue_coord[4:, :].mean(0)

        coords_CA = coords[:, 1:2, :]
        coords_SC = coords[:, -1:, :]

        coord = torch.cat( [coords_CA, coords_SC], dim=1 )
        node_scalar_features, node_vector_features = self._residue_features( coords, residue_types )
        edges, edge_scalar_features, edge_vector_features = self._interaction_features( coords, distance_cutoff=8, relative_position_cutoff=32 )
        
        node = {'coord': coord, 'node_scalar_features': node_scalar_features, 'node_vector_features': node_vector_features}
        edge = {'edges': edges, 'edge_scalar_features': edge_scalar_features, 'edge_vector_features': edge_vector_features}
        return node, edge
    

    
def get_ligand_coord_from_pdb(ligand_pdb_path):
    """
    Extract heavy atom coordinates from ligand PDB file (HETATM lines only).
    Excludes hydrogen atoms.
    """
    ligand_coords = []
    
    with open(ligand_pdb_path, 'r') as f:
        for line in f:
            if line.startswith('HETATM'):
                atom_name = line[12:16].strip()
                element = line[76:78].strip() if len(line) > 76 else atom_name[0]
                
                # Skip hydrogen atoms
                if atom_name.startswith('H') or element.upper() == 'H':
                    continue
                
                # Extract coordinates
                x = float(line[30:38])
                y = float(line[38:46]) 
                z = float(line[46:54])
                ligand_coords.append([x, y, z])
    
    return torch.tensor(ligand_coords).float() if ligand_coords else torch.empty(0, 3)

def find_binding_sites_from_ligands(protein_ca_coords, protein_sc_coords, ligand_coords_list, cutoff=6.0, verbose=False, residue_info=None, ligand_names=None):
    """
    Find binding site residues based on distance to ligand atoms.
    Considers both CA and SC coordinates - if either is within cutoff, residue is binding site.
    
    Args:
        protein_ca_coords: CA coordinates of protein residues [num_residues, 3]
        protein_sc_coords: SC coordinates of protein residues [num_residues, 3]
        ligand_coords_list: List of ligand coordinate tensors
        cutoff: Distance cutoff in Angstroms
        verbose: If True, print detailed statistics
        residue_info: List of residue information (chain, res_num, res_type) for verbose output
        ligand_names: List of ligand names for verbose output
        
    Returns:
        is_binding_site: Boolean tensor indicating binding site residues
    """
    num_residues = protein_ca_coords.shape[0]
    is_binding_site = torch.zeros(num_residues, dtype=torch.bool)
    total_ca_binding = torch.zeros(num_residues, dtype=torch.bool)
    total_sc_binding = torch.zeros(num_residues, dtype=torch.bool)
    
    for ligand_idx, ligand_coords in enumerate(ligand_coords_list):
        if ligand_coords.numel() == 0:  # Skip empty ligand
            continue
            
        # Calculate distances between CA atoms and ligand atoms
        ca_distances = torch.cdist(protein_ca_coords, ligand_coords)  # [num_residues, num_ligand_atoms]
        ca_min_distances, _ = ca_distances.min(dim=1)  # [num_residues]
        ca_binding_residues = ca_min_distances < cutoff
        
        # Calculate distances between SC atoms and ligand atoms
        sc_distances = torch.cdist(protein_sc_coords, ligand_coords)  # [num_residues, num_ligand_atoms]
        sc_min_distances, _ = sc_distances.min(dim=1)  # [num_residues]
        sc_binding_residues = sc_min_distances < cutoff
        
        # Combine CA and SC binding sites (OR operation)
        binding_residues = ca_binding_residues | sc_binding_residues
        
        # Update binding site mask (OR operation for multiple ligands)
        is_binding_site = is_binding_site | binding_residues
        total_ca_binding = total_ca_binding | ca_binding_residues
        total_sc_binding = total_sc_binding | sc_binding_residues
        
        if verbose and residue_info is not None:
            ligand_name = ligand_names[ligand_idx] if ligand_names else f"Ligand_{ligand_idx+1}"
            print(f"\n--- {ligand_name} (총 {ligand_coords.shape[0]}개 원자) ---")
            
            # CA binding sites
            ca_indices = torch.where(ca_binding_residues)[0]
            print(f"CA 바인딩 사이트 ({len(ca_indices)}개):")
            for idx in ca_indices:
                chain, res_num, res_type = residue_info[idx]
                ca_coord = protein_ca_coords[idx]
                min_dist = ca_min_distances[idx].item()
                print(f"  {chain}:{res_num} ({amino_acid_mapping_reverse.get(res_type, 'UNK')}) CA=({ca_coord[0]:.1f},{ca_coord[1]:.1f},{ca_coord[2]:.1f}) dist={min_dist:.2f}Å")
            
            # SC binding sites  
            sc_indices = torch.where(sc_binding_residues)[0]
            print(f"SC 바인딩 사이트 ({len(sc_indices)}개):")
            for idx in sc_indices:
                chain, res_num, res_type = residue_info[idx]
                sc_coord = protein_sc_coords[idx]
                min_dist = sc_min_distances[idx].item()
                print(f"  {chain}:{res_num} ({amino_acid_mapping_reverse.get(res_type, 'UNK')}) SC=({sc_coord[0]:.1f},{sc_coord[1]:.1f},{sc_coord[2]:.1f}) dist={min_dist:.2f}Å")
            
            # 통계
            ca_only_ligand = ca_binding_residues & ~sc_binding_residues
            sc_only_ligand = sc_binding_residues & ~ca_binding_residues
            both_ligand = ca_binding_residues & sc_binding_residues
            print(f"통계: CA만={ca_only_ligand.sum().item()}개, SC만={sc_only_ligand.sum().item()}개, 둘다={both_ligand.sum().item()}개, 합계={binding_residues.sum().item()}개")
    
    if verbose:
        print(f"\n=== 전체 바인딩 사이트 통계 ===")
        ca_only_total = total_ca_binding & ~total_sc_binding
        sc_only_total = total_sc_binding & ~total_ca_binding
        both_total = total_ca_binding & total_sc_binding
        
        print(f"CA만 탐지: {ca_only_total.sum().item()}개")
        print(f"SC만 탐지: {sc_only_total.sum().item()}개") 
        print(f"CA+SC 둘다 탐지: {both_total.sum().item()}개")
        print(f"총 바인딩 사이트 (합집합): {is_binding_site.sum().item()}개")
        print(f"CA 탐지 총합: {total_ca_binding.sum().item()}개")
        print(f"SC 탐지 총합: {total_sc_binding.sum().item()}개")
        print(f"교집합 (CA∩SC): {both_total.sum().item()}개")
        print(f"합집합 (CA∪SC): {is_binding_site.sum().item()}개")
    
    return is_binding_site

def get_protein_graph_torch_new(pdb_path, ligand_coords_list, verbose=False):
    """
    Create protein graph with binding site labels from multiple ligands.
    
    Args:
        pdb_path: Path to protein PDB file
        ligand_coords_list: List of ligand coordinate tensors
        verbose: If True, print detailed binding site statistics
        
    Returns:
        data: torch_geometric.data.Data object
    """
    parser = PDBParserSim(pdb_path)
    node, edge = parser.get_features()

    # Get CA and SC coordinates
    ca_coord = node['coord'][:, 0, :]
    sc_coord = node['coord'][:, 1, :]
    
    # Get residue information for verbose output
    residue_info = parser.get_residue() if verbose else None
    
    # Find binding sites from all ligands (using both CA and SC coords)
    is_binding_site = find_binding_sites_from_ligands(
        ca_coord, sc_coord, ligand_coords_list, cutoff=6.0, 
        verbose=verbose, residue_info=residue_info, ligand_names=None
    )
    
    # Calculate individual ligand centers
    ligand_centers = []
    for coords in ligand_coords_list:
        if coords.numel() > 0:
            center = coords.mean(0)
            ligand_centers.append(center)
        else:
            ligand_centers.append(torch.zeros(3))

    # Node features - concatenate all scalar features
    node_scalar_features = torch.cat([
        node['node_scalar_features'][0],  # residue_one_hot [N, 21]
        node['node_scalar_features'][1],  # self_distance [N, 10]
        node['node_scalar_features'][2],  # dihedrals [N, 16]
        node['node_scalar_features'][3],  # has_chi_angles [N, 5]
        node['node_scalar_features'][4],  # sasa [N, 10]
        node['node_scalar_features'][5],  # rf_distance [N, 8]
        node['node_scalar_features'][6]   # terminal_flags [N, 2]
    ], dim=-1)  # Total: [N, 72]
    
    # Node vector features - keep as [N, num_vectors, 3] format
    node_vector_features = torch.cat([
        node['node_vector_features'][0],  # self_vector: [N, 20, 3]
        node['node_vector_features'][1],  # rf_vector: [N, 8, 3]
    ], dim=1)  # Total: [N, 28, 3]
    
    # Edge indices and features
    src, dst = edge['edges']
    edge_index = torch.stack([src, dst], dim=0)
    edge_scalar_features = torch.cat([
        edge['edge_scalar_features'][0],  # distance [E, 4]
        edge['edge_scalar_features'][1]   # relative_distance [E, 34]
    ], dim=-1)  # Total: [E, 38]
    
    # Create torch_geometric Data object with minimal essential features
    data = Data(
        # Essential coordinates
        coord_CA=ca_coord,  # CA coordinates [N, 3]
        coord_SC=sc_coord,  # SC coordinates [N, 3] 
        ligand_centers=torch.stack(ligand_centers) if ligand_centers else torch.empty(0, 3),  # Individual ligand centers [num_ligands, 3]
        
        # Labels and connectivity
        y=is_binding_site.float(),  # Binding site labels [N]
        edge_index=edge_index,  # Edge connectivity [2, E]
        
        # Features
        node_scalar=node_scalar_features,  # Node scalar features [N, 72]
        node_vector=node_vector_features,  # Node vector features [N, 28, 3]
        edge_scalar=edge_scalar_features,  # Edge scalar features [E, 38]
    )

    return data

def process_bsite_pdb_dataset(input_dir, output_dir, test_mode=True, start_idx=None, end_idx=None):
    """
    Process the new dataset structure with protein.pdb and multiple ligand_{???}.pdb files.
    
    Args:
        input_dir: Root directory containing PDB subdirectories
        output_dir: Directory to save processed data
        test_mode: If True, process only the first PDB with detailed output
        start_idx: Starting index for processing (for parallel processing)
        end_idx: Ending index for processing (for parallel processing)
    """
    import glob
    from tqdm import tqdm
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all subdirectories containing PDB files
    pdb_dirs = [d for d in glob.glob(os.path.join(input_dir, "*")) if os.path.isdir(d)]
    pdb_dirs = sorted(pdb_dirs)  # Sort for consistent ordering across jobs
    
    total_dirs = len(pdb_dirs)
    print(f"Found {total_dirs} PDB directories to process")
    
    # Handle parallel processing parameters
    if start_idx is not None and end_idx is not None:
        pdb_dirs = pdb_dirs[start_idx:end_idx]
        print(f"Processing subset: {start_idx} to {end_idx-1} ({len(pdb_dirs)} directories)")
    elif test_mode:
        pdb_dirs = pdb_dirs[:5]  # Process first 5 for testing
        print(f"Test mode: Processing first 5 directories: {[os.path.basename(d) for d in pdb_dirs]}")
    
    processed_count = 0
    skipped_count = 0
    error_count = 0
    
    # Use tqdm for progress bar
    pdb_dirs_iter = tqdm(pdb_dirs, desc="Processing PDB directories", unit="dir")
    
    for idx, pdb_dir in enumerate(pdb_dirs_iter):
        pdb_id = os.path.basename(pdb_dir)
        output_pt = os.path.join(output_dir, f"{pdb_id}.pt")
        
        # Update tqdm description
        pdb_dirs_iter.set_postfix({
            'current': pdb_id,
            'processed': processed_count,
            'skipped': skipped_count,
            'errors': error_count
        })
        
        # Skip if already processed
        if os.path.exists(output_pt):
            if test_mode:
                print(f"⏭️ Skipping {pdb_id} (already processed)")
            skipped_count += 1
            continue
        
        # In test mode, only show detailed analysis for first unprocessed file
        is_first_test = test_mode and processed_count == 0
        verbose_mode = is_first_test
        
        if test_mode:
            print(f"\n=== Processing {pdb_id} ({processed_count+1}/{len(pdb_dirs)-skipped_count}) ===")
        
        # Find protein.pdb file
        protein_pdb = os.path.join(pdb_dir, "protein.pdb")
        if not os.path.exists(protein_pdb):
            print(f"Warning: protein.pdb not found in {pdb_dir}")
            error_count += 1
            continue
        
        # Find all ligand PDB files
        ligand_pattern = os.path.join(pdb_dir, "ligand_*.pdb")
        ligand_files = glob.glob(ligand_pattern)
        
        if not ligand_files:
            print(f"Warning: No ligand files found in {pdb_dir}")
            error_count += 1
            continue
        
        if verbose_mode:
            print(f"Found {len(ligand_files)} ligand files: {[os.path.basename(f) for f in ligand_files]}")
        elif test_mode:
            print(f"  Found {len(ligand_files)} ligand files")
        
        try:
            # Standardize protein PDB file
            standardized_protein_pdb = f"/tmp/{pdb_id}_protein_standardized.pdb"
            standardize_pdb(protein_pdb, standardized_protein_pdb)
            
            if verbose_mode:
                print(f"✓ Standardized protein PDB: {standardized_protein_pdb}")
            
            # Extract coordinates from all ligand files
            ligand_coords_list = []
            for ligand_file in ligand_files:
                ligand_coords = get_ligand_coord_from_pdb(ligand_file)
                if verbose_mode:
                    print(f"  {os.path.basename(ligand_file)}: {ligand_coords.shape[0]} heavy atoms")
                ligand_coords_list.append(ligand_coords)
            
            # Filter out empty ligand coordinates
            valid_ligand_coords = [coords for coords in ligand_coords_list if coords.numel() > 0]
            
            if not valid_ligand_coords:
                print(f"Warning: No valid ligand coordinates found in {pdb_dir}")
                error_count += 1
                continue
            
            # Generate protein graph with binding site labels using standardized PDB
#data = get_protein_graph_torch_new(standardized_protein_pdb, valid_ligand_coords, verbose=verbose_mode)
            data = get_protein_graph_torch_new(protein_pdb, valid_ligand_coords, verbose=verbose_mode)

            
            # Add metadata
            data.pdb_id = pdb_id
            data.ligand_files = [os.path.basename(f) for f in ligand_files]
            
            # Print summary information
            if verbose_mode:
                print(f"✓ Graph created - Nodes: {data.coord_CA.shape[0]}, Edges: {data.edge_index.shape[1]}")
                print(f"✓ Binding sites: {int(data.y.sum())}/{data.y.shape[0]} ({data.y.sum()/data.y.shape[0]*100:.1f}%)")
                print(f"✓ Features: Node_scalar={data.node_scalar.shape}, Node_vector={data.node_vector.shape}, Edge_scalar={data.edge_scalar.shape}")
                print(f"✓ Ligand centers ({data.ligand_centers.shape[0]} ligands):")
                for i, center in enumerate(data.ligand_centers):
                    ligand_name = data.ligand_files[i] if hasattr(data, 'ligand_files') else f"Ligand_{i+1}"
                    print(f"   {ligand_name}: ({center[0]:.1f}, {center[1]:.1f}, {center[2]:.1f})")
            elif test_mode:
                print(f"  ✓ {data.coord_CA.shape[0]} nodes, {int(data.y.sum())} binding sites ({data.y.sum()/data.y.shape[0]*100:.1f}%)")
            
            # Save processed data
            torch.save(data, output_pt)
            processed_count += 1
            
            if verbose_mode:
                print(f"Saved to: {output_pt}")
            elif test_mode:
                print(f"  Saved: {pdb_id}.pt")
            
            # Clean up temporary standardized file
            if os.path.exists(standardized_protein_pdb):
                # os.remove(standardized_protein_pdb)
                pass
                
        except Exception as e:
            print(f"Error processing {pdb_id}: {e}")
            error_count += 1
            if verbose_mode:
                import traceback
                traceback.print_exc()
            continue
    
    # Print final statistics
    print(f"\n=== 처리 완료 ===")
    print(f"새로 처리됨: {processed_count}개")
    print(f"이미 존재함: {skipped_count}개") 
    print(f"오류 발생: {error_count}개")
    print(f"총 디렉토리: {len(pdb_dirs)}개")

if __name__ == "__main__":
    import sys
    
    # Configuration
    input_dir = "PDB_downloads"
    output_dir = "data/processed"
    
    # Parse command line arguments
    test_mode = "--test" in sys.argv or "-t" in sys.argv
    start_idx = None
    end_idx = None
    
    # Parse start and end indices for parallel processing
    for i, arg in enumerate(sys.argv):
        if arg == "--start" and i + 1 < len(sys.argv):
            start_idx = int(sys.argv[i + 1])
        elif arg == "--end" and i + 1 < len(sys.argv):
            end_idx = int(sys.argv[i + 1])
    
    if test_mode:
        print("🔍 TEST MODE: 상세 분석과 함께 첫 5개 파일만 처리합니다")
        print("   전체 데이터셋 처리하려면: python data/feat.py")
        print("   테스트 모드: python data/feat.py --test 또는 python data/feat.py -t")
        print("   병렬 처리: python data/feat.py --start 0 --end 1000")
        print()
    elif start_idx is not None and end_idx is not None:
        print(f"🚀 PARALLEL MODE: {start_idx}~{end_idx-1} 범위를 처리합니다")
        print()
    else:
        print("🚀 FULL MODE: 전체 데이터셋을 처리합니다")
        print("   테스트 모드로 실행하려면: python data/feat.py --test")
        print("   병렬 처리: python data/feat.py --start 0 --end 1000")
        print()
    
    process_bsite_pdb_dataset(input_dir=input_dir, output_dir=output_dir, test_mode=test_mode, start_idx=start_idx, end_idx=end_idx)
