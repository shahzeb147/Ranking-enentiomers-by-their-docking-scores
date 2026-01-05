# Find Broken Views
import numpy as np
np.random.seed(42)

def small_views(vs, piece_size=4):
    """
    Split views into smaller pieces such that each small view contains only atomic number + 3D coordinates.

    Parameters:
    - vs: numpy array of views
    - piece_size: size of each small view (default = 4: atomic number + 3D coordinates)
    
    Returns:
    - new_view: numpy array with views split into atomic pieces
    """
    # Check that the view size is divisible by the piece size
    if vs.shape[-1] % piece_size != 0:
        raise ValueError(f"View size {vs.shape[-1]} is not divisible by piece size {piece_size}.")
    
    single_atom_piece = vs.shape[-1] // piece_size
    
    new_view = np.reshape(vs, (vs.shape[0], vs.shape[1], single_atom_piece, piece_size))
    
    return new_view

###################################################################################

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
        "electronegativity": 3.04, "atomic_mass": 14.007, "valence_electrons": 3,
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

#######################################################################################


# calcualte coulomb matrix

def coulomb_interaction(nuclear_charge_i, nuclear_charge_j, coord_i, coord_j):
    """
calculate coulomb interaction
Parameters:
- nuclear_charge_i: nuclear charge of atom i
- nuclear_charge_j: nuclear charge of atom j
-coord_i: 3D coordinates of atom i
-coord_j: 3D coordinates of atom j

Returns: 
- Coulomb interaction 

    """

    # calculate distance between two atoms

    distance = np.linalg.norm(np.array(coord_i)-np.array(coord_j))

    if distance == 0:
        return 0
    return(nuclear_charge_i * nuclear_charge_j)/ distance

##########################################################################################

def coulomb_interaction_broken(single_atom):
    """
    calculate coulomb interaction of an atom with all other atoms present in a view
    """
    number_of_molecules = single_atom.shape[0]
    number_of_views = single_atom.shape[1]
    number_of_atoms = single_atom.shape[2]

    interaction = np.zeros((number_of_molecules, number_of_views, number_of_atoms, number_of_atoms-1))

# extract a single view data first

    for mol_index in range(number_of_molecules):
        for view_index in range(number_of_views):
            view = single_atom[mol_index, view_index]

# extract specific atom and then find its coulomb interaction
            for i in range(number_of_atoms):
                nuclear_charge_i = view[i, 0]
                coord_i = view[i, 1:4]

                col_index = 0
                for j in range(number_of_atoms):
                    if i != j:
                        nuclear_charge_j = view[j, 0]
                        coord_j = view[j, 1:4]
                        interaction[mol_index, view_index, i, col_index] = coulomb_interaction(nuclear_charge_i, nuclear_charge_j, coord_i, coord_j)
                        col_index += 1
    return interaction

#################################################################################################

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


def get_embeddings(single_atom, coulomb_interaction_all, atom_properties, single_atomic_property_switches):

    """
    get features for each atom

    parameters: 
    - single_atom: broken views of size (80,17,,17,4) and (20,17,,17,4)
    - coulomb_interaction_all: coulomb interaction of size (80, 17,17,16) and (20, 17,17,16)
    - atom_properties: dictionary containing properties for each atom type
    - single_atomic_properties_switches: switches to add single atom properties

    Returns: 
    - array with atomic number, 3D coordinates, atomic properties and coulomb interactions
    """
    



    number_of_molecules = single_atom.shape[0]
    number_of_views = single_atom.shape[1]
    number_of_atoms = single_atom.shape[2]
    property_keys = [prop for prop, is_on in single_atomic_property_switches.items() if is_on]
    num_properties = len(property_keys)

    atom_embeddings = []

    for mol_id in range(number_of_molecules):
        molecule_extended = []
        for view_id in range(number_of_views):
            view1 = single_atom[mol_id, view_id]
            view_extended = []

            for atom_id in range(number_of_atoms):
                atomic_number = view1[atom_id, 0]

                # handle atoms with atomic number 0.0
                if atomic_number == 0.0:
                    extended_atom = np.zeros(1 + 3 + num_properties + 22)  # fill with zeros
                else:
                    coord = view1[atom_id, 1:4]
                    atom_type = atomic_number_to_type[atomic_number]
                    properties = [atom_properties[atom_type][prop] for prop in property_keys]
                    coulomb_interaction = coulomb_interaction_all[mol_id, view_id, atom_id]

                    extended_atom = np.concatenate([[atomic_number], coord, properties, coulomb_interaction]) #concatenate atomic number, coord, properties and coulomb matrix of a single atom

                view_extended.append(extended_atom) #conatin feature for all atoms in a single view (23 in our case)
            molecule_extended.append(np.array(view_extended)) #all views in a molecule

        atom_embeddings.append(np.array(molecule_extended))

    return np.array(atom_embeddings)


##############################################################################