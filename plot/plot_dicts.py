target_keywords = {
    "decorrelation.txt": "Decorrelation:",
    "distance.txt": "Distance",
    "interference.txt": "Interference:",
    "linear_probing_xy.txt": "Percentage error",
    "linear_probing_color.txt": "Percentage error",
    "linear_probing_count.txt": "Percentage error",
    "orthogonality.txt": "Orthogonality:",
    "sparsity_instance.txt": "sparsity:",
}

violin_colors = {
    "DQN": "C4",

    "DQN+XY": "C1",
    "DQN+XY+color": "goldenrod",
    "DQN+XY+count": "C8",
    "DQN+Decoder": "C3",
    "DQN+NAS": "C5",
    "DQN+SF": "C6",
    "DQN+AuxControl": "C0",
    "DQN+AuxControl1g": "C0",
    "DQN+AuxControl5g": "C9",

    "Sparse0.1": "C2",
    "Sparse0.2": "springgreen",
    "Sparse0.4": "lightgreen",

    "DQN+LTA": "C2",

    "Laplace": "gray",
    "Scratch DQN": "black",
    "Random": "navy",
    "Input": "gold",
}

mc_learn = [
    {"label": "DQN", "property": "data/output/test/",
     "control": "data/output/test/mountaincar/representations/dqn/sweep/",
     },
    {"label": "DQN+LTA", "property": "data/output/test/",
     "control": "data/output/test/mountaincar/representations/dqn_lta/sweep/",
     },
]
