import torch

# The exact order from your Stage A (16 species)
PRETEXT_SPECIES = [
    "BF",
    "MR",
    "MH",
    "BY",
    "BW",
    "AB",
    "LA",
    "SW",
    "SB",
    "PJ",
    "PR",
    "PW",
    "PI",
    "PT",
    "OR",
    "CW",
]

# The exact order of labels in your .npz files
# I have added PW to NIF as requested.
SITE_LABELS = {
    "WRF": ["SB", "LA", "PJ", "BW", "PT", "BF", "CW", "SW"],
    "RMF": ["BF", "BW", "CE", "LA", "PT", "PJ", "PO", "SB", "SW"],
    "NIF": ["SB", "LA", "PW", "MH", "MR", "BY", "BW", "BF", "PT"],
    "OVF": ["AB", "PO", "MR", "BF", "CE", "PW", "MH", "BW", "SW", "OR", "PR"],
}

# Taxonomic logic: Mapping Site Codes to Pretext Species
# If a site uses a Genus code (PO, CE), we map all relevant pretext species to it.
GROUPS = {
    "PO": ["PI", "PT"],  # Poplar contains both Aspens
    "CE": ["CW"],  # Cedar contains Eastern White Cedar
}


def get_mapping_matrix(site_name):
    """
    Creates a matrix of size [16, num_site_labels].
    A 1.0 at matrix[i, j] means pretext species i belongs to site label j.
    """
    site_list = SITE_LABELS[site_name.upper()]
    matrix = torch.zeros(len(PRETEXT_SPECIES), len(site_list))

    for site_idx, site_code in enumerate(site_list):
        # 1. Check if the site code is a group (PO, CE)
        if site_code in GROUPS:
            for pretext_code in GROUPS[site_code]:
                if pretext_code in PRETEXT_SPECIES:
                    pretext_idx = PRETEXT_SPECIES.index(pretext_code)
                    matrix[pretext_idx, site_idx] = 1.0
        # 2. Otherwise, check for a direct match (e.g., SB -> SB)
        elif site_code in PRETEXT_SPECIES:
            pretext_idx = PRETEXT_SPECIES.index(site_code)
            matrix[pretext_idx, site_idx] = 1.0
        else:
            # Species in site data but NOT in pretext knowledge
            print(f"Warning: Site species {site_code} has no pretext equivalent.")

    return matrix
