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
    "DQN+LTA(no target)": "springgreen",

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
    {"label": "DQN+LTA(no target)", "property": "data/output/test/",
     "control": "data/output/test/mountaincar/representations/dqn_lta/no_target/",
     },
]

gh_learn = [
    {"label": "DQN", "property": "data/output/test/",
     "control": "data/output/test/gridhard/representations/dqn/best/",
    },
    {"label": "DQN+LTA", "property": "data/output/test/",
     "control": "data/output/test/gridhard/representations/dqn_lta/best/",
    },
]
gh_same = [
    {"label": "DQN", "property": "data/output/test/",
     "control": "data/output/test/gridhard/control/same_task/fix_rep/dqn/best/",
    },
    # {"label": "DQN+LTA", "property": "data/output/test/",
    #  "control": "data/output/test/gridhard/control/same_task/fix_rep/dqn_lta/best/",
    # },
    {"label": "Random", "property": "data/output/test/",
     "control": "data/output/test/gridhard/control/same_task/fix_rep/random/best/",
     },
    {"label": "Input", "property": "data/output/test/",
     "control": "data/output/test/gridhard/control/same_task/fix_rep/input/best/",
     },
]

gh_similar = [
    {"label": "DQN", "property": "data/output/test/",
     "control": "data/output/test/gridhard/control/similar_task/fix_rep/dqn/best/",
    },
    # {"label": "DQN+LTA", "property": "data/output/test/",
    #  "control": "data/output/test/gridhard/control/similar_task/fix_rep/dqn_lta/best/",
    # },
    {"label": "Random", "property": "data/output/test/",
     "control": "data/output/test/gridhard/control/similar_task/fix_rep/random/best/",
     },
    {"label": "Input", "property": "data/output/test/",
     "control": "data/output/test/gridhard/control/similar_task/fix_rep/input/best/",
     },
]

gh_diff = [
    {"label": "DQN", "property": "data/output/test/",
     "control": "data/output/test/gridhard/control/different_task/fix_rep/dqn/best/",
    },
    # {"label": "DQN+LTA", "property": "data/output/test/",
    #  "control": "data/output/test/gridhard/control/different_task/fix_rep/dqn_lta/best/",
    # },
    {"label": "Random", "property": "data/output/test/",
     "control": "data/output/test/gridhard/control/different_task/fix_rep/random/best/",
     },
    {"label": "Input", "property": "data/output/test/",
     "control": "data/output/test/gridhard/control/different_task/fix_rep/input/best/",
     },
]

gh_diff_tune = [
    {"label": "DQN", "property": "data/output/test/",
     "control": "data/output/test/gridhard/control/difference_task/fine_tune/dqn/best/",
    },
    {"label": "DQN+LTA", "property": "data/output/test/",
     "control": "data/output/test/gridhard/control/difference_task/fine_tune/dqn_lta/best/",
    },
    {"label": "Random", "property": "data/output/test/",
     "control": "data/output/test/gridhard/control/difference_task/fine_tune/random/best/",
     },
    {"label": "Input", "property": "data/output/test/",
     "control": "data/output/test/gridhard/control/difference_task/fine_tune/input/best/",
     },
]
