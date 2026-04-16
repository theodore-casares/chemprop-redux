from .data import (
    MoleculeDatapoint,
    MoleculeDataset,
    MoleculeDataLoader,
    MoleculeSampler,
    construct_molecule_batch,
    make_mols,
    cache_graph,
    set_cache_graph,
    cache_mol,
    set_cache_mol,
    empty_cache,
)
from .scaffold import (
    generate_scaffold,
    scaffold_to_smiles,
    scaffold_split,
    log_scaffold_stats,
)
from .scaler import StandardScaler


def preprocess_smiles_columns(path, smiles_columns):
    """Return [smiles_columns] wrapped as a list; default to first CSV column."""
    import csv
    if smiles_columns is None:
        with open(path) as f:
            header = next(csv.reader(f))
        return [header[0]]
    if isinstance(smiles_columns, str):
        return [smiles_columns]
    return list(smiles_columns)


def get_task_names(path, smiles_columns=None, target_columns=None, ignore_columns=None):
    """Header columns that are not smiles / ignored → task names."""
    import csv
    if target_columns is not None:
        return list(target_columns)
    with open(path) as f:
        header = next(csv.reader(f))
    smiles_columns = preprocess_smiles_columns(path, smiles_columns)
    ignore = set(smiles_columns) | set(ignore_columns or [])
    return [c for c in header if c not in ignore]
