import numpy as np
np.random.seed(42)
# sources 
# Electron affinity : https://periodictable.com/Properties/A/ElectronAffinity.v.log.html
# Electronegativity :https://periodictable.com/Properties/A/Electronegativity.al.html
# Valence electron: https://periodictable.com/Properties/A/Valence.al.html
# covalent radius: https://periodictable.com/Properties/A/CovalentRadius.an.html
# First ionization energy: https://periodictable.com/Properties/A/IonizationEnergies.an.html
# Atomic volume: http://hyperphysics.phy-astr.gsu.edu/hbase/pertab/H.html
# Dictionary containing properties for each atom type
atom_properties = {
    "H": {
        "electronegativity": 2.20, "atomic_mass": 1.008, "valence_electrons": 1,
        "group_number": 1, "covalent_radius": 31, "first_ionization_energy": 1312,
        "electron_affinity": 73, "atomic_volume": 14.0
    },
    "C": {
        "electronegativity": 2.55, "atomic_mass": 12.01, "valence_electrons": 4,
        "group_number": 14, "covalent_radius": 76, "first_ionization_energy": 1086,
        "electron_affinity": 153.9, "atomic_volume": 4.58
    },
    "N": {
        "electronegativity": 3.04, "atomic_mass": 14.007, "valence_electrons": 5,
        "group_number": 15, "covalent_radius": 71, "first_ionization_energy": 1402,
        "electron_affinity": 7, "atomic_volume": 17.3
    },
    "O": {
        "electronegativity": 3.44, "atomic_mass": 15.999, "valence_electrons": 2,
        "group_number": 16, "covalent_radius": 66, "first_ionization_energy": 1314,
        "electron_affinity": 141, "atomic_volume": 14
    },
    "S": {
        "electronegativity": 2.58, "atomic_mass": 32.06, "valence_electrons": 6,
        "group_number": 16, "covalent_radius": 105, "first_ionization_energy": 999.6,
        "electron_affinity": 200, "atomic_volume": 15.5
    },
    "F": {
        "electronegativity": 3.98, "atomic_mass": 18.998, "valence_electrons": 7,
        "group_number": 17, "covalent_radius": 57, "first_ionization_energy": 1681,
        "electron_affinity": 328, "atomic_volume": 17.1
    },
    "P": {
        "electronegativity": 2.19, "atomic_mass": 30.974, "valence_electrons": 5,
        "group_number": 15, "covalent_radius": 107, "first_ionization_energy": 1011.8,
        "electron_affinity": 72, "atomic_volume": 17.0
    },
    "Cl": {
        "electronegativity": 3.16, "atomic_mass": 35.45, "valence_electrons": 7,
        "group_number": 17, "covalent_radius": 102, "first_ionization_energy": 1251.2,
        "electron_affinity": 349, "atomic_volume": 22.7
    },
    "Br": {
        "electronegativity": 2.96, "atomic_mass": 79.904, "valence_electrons": 7,
        "group_number": 17, "covalent_radius": 120, "first_ionization_energy": 1139.9,
        "electron_affinity": 324.6, "atomic_volume": 23.5
    },
    "I": {
        "electronegativity": 2.66, "atomic_mass": 126.90, "valence_electrons": 7,
        "group_number": 17, "covalent_radius": 139, "first_ionization_energy": 1008.4,
        "electron_affinity": 295.2, "atomic_volume": 25.7
    }
  

}

# Switch for atomic properties
single_atomic_property_switches = {
    "electronegativity": True,
    "atomic_mass": True,
    "valence_electrons": True,
    "group_number": True,
    "covalent_radius": True,
    "first_ionization_energy": True,
    "electron_affinity": True,
    "atomic_volume": True
}


atomic_number_to_type = {
    1: "H",
    6: "C",
    7: "N",
    8: "O",
    9: "F",
    15: "P",
    16: "S",
    17: "Cl",
    35: "Br",
    53: "I"
}


def get_embeddings(single_atom, atom_properties, single_atomic_property_switches, embedding_size=16):
    """
    Get fixed-size embeddings (length=16) for each atom using atomic number, 3D coordinates, and selected properties.
    Remaining positions are zero-padded.

    Parameters:
    - single_atom: numpy array (N_molecules, N_views, N_atoms, 4)
    - atom_properties: dictionary of properties per atom type
    - single_atomic_property_switches: which properties to include
    - embedding_size: desired fixed embedding length (default = 16)

    Returns:
    - atom_embeddings: numpy array (N_molecules, N_views, N_atoms, embedding_size)
    """
    number_of_molecules = single_atom.shape[0]
    number_of_views = single_atom.shape[1]
    number_of_atoms = single_atom.shape[2]
    
    property_keys = [prop for prop, is_on in single_atomic_property_switches.items() if is_on]
    num_features = 1 + 3 + len(property_keys)  # atomic_number + coordinates + properties

    if num_features > embedding_size:
        raise ValueError(f"Embedding size {embedding_size} is too small for selected features (requires at least {num_features}).")

    atom_embeddings = []

    for mol_id in range(number_of_molecules):
        molecule_extended = []
        for view_id in range(number_of_views):
            view = single_atom[mol_id, view_id]
            view_extended = []

            for atom_id in range(number_of_atoms):
                atomic_number = view[atom_id, 0]

                if atomic_number == 0.0:
                    extended_atom = np.zeros(embedding_size)
                else:
                    coord = view[atom_id, 1:4]
                    atom_type = atomic_number_to_type.get(atomic_number, "H")  # default to H if unknown
                    properties = [atom_properties[atom_type][prop] for prop in property_keys]
                    raw_features = np.concatenate([[atomic_number], coord, properties])

                    # Pad with zeros to reach desired embedding size
                    padded = np.zeros(embedding_size)
                    padded[:len(raw_features)] = raw_features
                    extended_atom = padded

                view_extended.append(extended_atom)
            molecule_extended.append(np.array(view_extended))
        atom_embeddings.append(np.array(molecule_extended))

    return np.array(atom_embeddings)