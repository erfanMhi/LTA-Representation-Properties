import matplotlib
cmap = matplotlib.cm.get_cmap('cool')

target_keywords = {
    "decorrelation.txt": "Decorrelation:",
    "distance.txt": "Distance",
    "noninterference.txt": "Noninterference:",
    "linear_probing_xy.txt": "Percentage error",
    "linear_probing_color.txt": "Percentage error",
    "linear_probing_count.txt": "Percentage error",
    "orthogonality.txt": "Orthogonality:",
    "sparsity_instance.txt": "sparsity:",
}
target_files = {
    "decorr": "decorrelation.txt",
    "distance": "distance.txt",
    "noninterf": "noninterference.txt",
    "ortho": "orthogonality.txt",
    "sparsity": "sparsity_instance.txt"
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
    "DQN+Reward": "C10",

    "Sparse0.1": "C2",
    "Sparse0.2": "springgreen",
    "Sparse0.4": "lightgreen",

    "DQN+LTA": "C2",
    "DQN+LTA(no target)": "springgreen",
    "DQN+LTA+Reward": "C7",

    "DQN+LTA eta=0.2": "C1",
    "DQN+LTA eta=0.4": "C2",
    "DQN+LTA eta=0.6": "C3",
    "DQN+LTA eta=0.8": "C4",

    "Laplace": "gray",
    "Scratch DQN": "black",
    "Random": "navy",
    "Input": "gold",
}

mc_learn_sweep = [
    {"label": "DQN",
     "control": "data/output/test/mountaincar/representations/dqn/sweep/",
     },
    {"label": "DQN+LTA",
     "control": "data/output/test/mountaincar/representations/dqn_lta/sweep/",
     },
    {"label": "DQN+LTA(no target)",
     "control": "data/output/test/mountaincar/representations/dqn_lta/no_target/",
     },
]

gh_learn_sweep = [
    {"label": "DQN",
     "control": "data/output/test/gridhard/representations/dqn/sweep/",
     },
    {"label": "DQN+LTA",
     "control": "data/output/test/gridhard/representations/dqn_lta/sweep/",
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

gh_same_sweep = [
    {"label": "DQN",
     "control": "data/output/test/gridhard/control/same_task/fix_rep/dqn/sweep/",
     },
    {"label": "DQN+Reward",
     "control": "data/output/test/gridhard/control/same_task/fix_rep/dqn_aux/reward/sweep/",
     },
    {"label": "DQN+LTA",
     "control": "data/output/test/gridhard/control/same_task/fix_rep/dqn_lta/sweep/",
     },
    {"label": "DQN+LTA+Reward",
     "control": "data/output/test/gridhard/control/same_task/fix_rep/dqn_lta_aux/reward/sweep/",
     },
    {"label": "Random",
     "control": "data/output/test/gridhard/control/same_task/fix_rep/random/sweep/",
     },
    {"label": "Input",
     "control": "data/output/test/gridhard/control/same_task/fix_rep/input/sweep/",
     },
]
gh_same = [
    {"label": "DQN", "property": "data/output/test/",
     "control": "data/output/test/gridhard/control/same_task/fix_rep/dqn/best/",
     "online_measure": "data/output/test/gridhard/online_property/dqn/best/"
     },
    {"label": "DQN+Reward",
     "control": "data/output/test/gridhard/control/same_task/fix_rep/dqn_aux/reward/best/",
     "online_measure": "data/output/test/gridhard/online_property/dqn_aux/reward/best/"
     },
    {"label": "DQN+LTA", "property": "data/output/test/",
     "control": "data/output/test/gridhard/control/same_task/fix_rep/dqn_lta/best/",
     "online_measure": "data/output/test/gridhard/online_property/dqn_lta/best/"
     },
    {"label": "DQN+LTA+Reward",
     "control": "data/output/test/gridhard/control/same_task/fix_rep/dqn_lta_aux/reward/best/",
     "online_measure": "data/output/test/gridhard/online_property/dqn_lta_aux/reward/best/"
     },
    {"label": "Random",
     "control": "data/output/test/gridhard/control/same_task/fix_rep/random/best/",
     "property": "data/output/test/gridhard/property/random/best",
     },
    {"label": "Input",
     "control": "data/output/test/gridhard/control/same_task/fix_rep/input/best/",
     "property": "data/output/test/gridhard/property/input/best",
     },
]

gh_similar_sweep = [
    {"label": "DQN",
     "control": "data/output/test/gridhard/control/similar_task/fix_rep/dqn/sweep/",
     },
    {"label": "DQN+Reward",
     "control": "data/output/test/gridhard/control/similar_task/fix_rep/dqn_aux/reward/sweep/",
     },
    {"label": "DQN+LTA",
     "control": "data/output/test/gridhard/control/similar_task/fix_rep/dqn_lta/sweep/",
     },
    {"label": "DQN+LTA+Reward",
     "control": "data/output/test/gridhard/control/similar_task/fix_rep/dqn_lta_aux/reward/sweep/",
     },
    {"label": "Random",
     "control": "data/output/test/gridhard/control/similar_task/fix_rep/random/sweep/",
     },
    {"label": "Input",
     "control": "data/output/test/gridhard/control/similar_task/fix_rep/input/sweep/",
     },
]
gh_similar = [
    {"label": "DQN", "property": "data/output/test/",
     "control": "data/output/test/gridhard/control/similar_task/fix_rep/dqn/best/",
     "online_measure": "data/output/test/gridhard/online_property/dqn/best/"
     },
    {"label": "DQN+Reward",
     "control": "data/output/test/gridhard/control/similar_task/fix_rep/dqn_aux/reward/best/",
     "online_measure": "data/output/test/gridhard/online_property/dqn_aux/reward/best/"
     },
    {"label": "DQN+LTA", "property": "data/output/test/",
     "control": "data/output/test/gridhard/control/similar_task/fix_rep/dqn_lta/best/",
     "online_measure": "data/output/test/gridhard/online_property/dqn_lta/best/"
     },
    {"label": "DQN+LTA+Reward",
     "control": "data/output/test/gridhard/control/similar_task/fix_rep/dqn_lta_aux/reward/best/",
     "online_measure": "data/output/test/gridhard/online_property/dqn_lta_aux/reward/best/"
     },
    {"label": "Random",
     "control": "data/output/test/gridhard/control/similar_task/fix_rep/random/best/",
     "property": "data/output/test/gridhard/property/random/best",
     },
    {"label": "Input",
     "control": "data/output/test/gridhard/control/similar_task/fix_rep/input/best/",
     "property": "data/output/test/gridhard/property/input/best",
     },
]

gh_diff_sweep = [
    {"label": "DQN",
     "control": "data/output/test/gridhard/control/different_task/fix_rep/dqn/sweep/",
     },
    {"label": "DQN+Reward",
     "control": "data/output/test/gridhard/control/different_task/fix_rep/dqn_aux/reward/sweep/",
     },
    {"label": "DQN+LTA",
     "control": "data/output/test/gridhard/control/different_task/fix_rep/dqn_lta/sweep/",
     },
    {"label": "DQN+LTA+Reward",
     "control": "data/output/test/gridhard/control/different_task/fix_rep/dqn_lta_aux/reward/sweep/",
     },
    {"label": "Random",
     "control": "data/output/test/gridhard/control/different_task/fix_rep/random/sweep/",
     },
    {"label": "Input",
     "control": "data/output/test/gridhard/control/different_task/fix_rep/input/sweep/",
     },
]
gh_diff = [
    {"label": "DQN", "property": "data/output/test/",
     "control": "data/output/test/gridhard/control/different_task/fix_rep/dqn/best/",
     "online_measure": "data/output/test/gridhard/online_property/dqn/best/"
     },
    {"label": "DQN+Reward",
     "control": "data/output/test/gridhard/control/different_task/fix_rep/dqn_aux/reward/best/",
     "online_measure": "data/output/test/gridhard/online_property/dqn_aux/reward/best/"
     },
    {"label": "DQN+LTA", "property": "data/output/test/",
     "control": "data/output/test/gridhard/control/different_task/fix_rep/dqn_lta/best/",
     "online_measure": "data/output/test/gridhard/online_property/dqn_lta/best/"
     },
    {"label": "DQN+LTA+Reward",
     "control": "data/output/test/gridhard/control/different_task/fix_rep/dqn_lta_aux/reward/best/",
     "online_measure": "data/output/test/gridhard/online_property/dqn_lta_aux/reward/best/"
     },
    {"label": "Random",
     "control": "data/output/test/gridhard/control/different_task/fix_rep/random/best/",
     "property": "data/output/test/gridhard/property/random/best",
     },
    {"label": "Input",
     "control": "data/output/test/gridhard/control/different_task/fix_rep/input/best/",
     "property": "data/output/test/gridhard/property/input/best",
     },

    # {"label": "DQN+LTA eta=0.2",
    #  "control": "data/output/test/gridhard/control/different_task/fix_rep/dqn_lta/eta_study_0.2_best/",
    #  "online_measure": "data/output/test/gridhard/online_property/dqn_lta/eta_study_0.2_best/"
    #  },
    # {"label": "DQN+LTA eta=0.6",
    #  "control": "data/output/test/gridhard/control/different_task/fix_rep/dqn_lta/eta_study_0.6_best/",
    #  "online_measure": "data/output/test/gridhard/online_property/dqn_lta/eta_study_0.6_best/"
    #  },
    # {"label": "DQN+LTA eta=0.8",
    #  "control": "data/output/test/gridhard/control/different_task/fix_rep/dqn_lta/eta_study_0.8_best/",
    #  "online_measure": "data/output/test/gridhard/online_property/dqn_lta/eta_study_0.8_best/"
    #  }
]

gh_diff_tune_sweep = [
    {"label": "DQN",
     "control": "data/output/test/gridhard/control/different_task/fine_tune/dqn/sweep/",
     },
    {"label": "DQN+Reward",
     "control": "data/output/test/gridhard/control/different_task/fine_tune/dqn_aux/reward/sweep/",
     },
    {"label": "DQN+LTA",
     "control": "data/output/test/gridhard/control/different_task/fine_tune/dqn_lta/sweep/",
     },
    {"label": "DQN+LTA+Reward",
     "control": "data/output/test/gridhard/control/different_task/fine_tune/dqn_lta_aux/reward/sweep/",
     },
    {"label": "Random",
     "control": "data/output/test/gridhard/control/different_task/fine_tune/random/sweep/",
     },
]
gh_diff_tune = [
    {"label": "DQN", "property": "data/output/test/",
     "control": "data/output/test/gridhard/control/different_task/fine_tune/dqn/best/",
     "online_measure": "data/output/test/gridhard/online_property/dqn/best/"
     },
    {"label": "DQN+Reward",
     "control": "data/output/test/gridhard/control/different_task/fine_tune/dqn_aux/reward/best/",
     "online_measure": "data/output/test/gridhard/online_property/dqn_aux/reward/best/"
     },
    {"label": "DQN+LTA", "property": "data/output/test/",
     "control": "data/output/test/gridhard/control/different_task/fine_tune/dqn_lta/best/",
     "online_measure": "data/output/test/gridhard/online_property/dqn_lta/best/"
     },
    {"label": "DQN+LTA+Reward",
     "control": "data/output/test/gridhard/control/different_task/fine_tune/dqn_lta_aux/reward/best/",
     "online_measure": "data/output/test/gridhard/online_property/dqn_lta_aux/reward/best/"
     },
    {"label": "Random",
     "control": "data/output/test/gridhard/control/different_task/fine_tune/random/best/",
     "property": "data/output/test/gridhard/property/random/best",
     },
    #
    # {"label": "DQN+LTA eta=0.2",
    #  "control": "data/output/test/gridhard/control/different_task/fine_tune/dqn_lta/eta_study_0.2_best/",
    #  "online_measure": "data/output/test/gridhard/online_property/dqn_lta/eta_study_0.2_best/"
    #  },
    # {"label": "DQN+LTA eta=0.6",
    #  "control": "data/output/test/gridhard/control/different_task/fine_tune/dqn_lta/eta_study_0.6_best/",
    #  "online_measure": "data/output/test/gridhard/online_property/dqn_lta/eta_study_0.6_best/"
    #  },
    # {"label": "DQN+LTA eta=0.8",
    #  "control": "data/output/test/gridhard/control/different_task/fine_tune/dqn_lta/eta_study_0.8_best/",
    #  "online_measure": "data/output/test/gridhard/online_property/dqn_lta/eta_study_0.8_best/"
    #  }
]

gh_etaStudy_diff_fix_sweep = [
    {"label": "DQN+LTA eta=0.2",
     "control": "data/output/test/gridhard/control/different_task/fix_rep/dqn_lta/eta_study_0.2/",
    },
    {"label": "DQN+LTA eta=0.4",
     "control": "data/output/test/gridhard/control/different_task/fix_rep/dqn_lta/sweep/",
    },
    {"label": "DQN+LTA eta=0.6",
     "control": "data/output/test/gridhard/control/different_task/fix_rep/dqn_lta/eta_study_0.6/",
     },
    {"label": "DQN+LTA eta=0.8",
     "control": "data/output/test/gridhard/control/different_task/fix_rep/dqn_lta/eta_study_0.8/",
     },
]
gh_etaStudy_diff_tune_sweep = [
    {"label": "DQN+LTA eta=0.2",
     "control": "data/output/test/gridhard/control/different_task/fine_tune/dqn_lta/eta_study_0.2/",
     },
    {"label": "DQN+LTA eta=0.4",
     "control": "data/output/test/gridhard/control/different_task/fine_tune/dqn_lta/sweep/",
    },
    {"label": "DQN+LTA eta=0.6",
     "control": "data/output/test/gridhard/control/different_task/fine_tune/dqn_lta/eta_study_0.6/",
     },
    {"label": "DQN+LTA eta=0.8",
     "control": "data/output/test/gridhard/control/different_task/fine_tune/dqn_lta/eta_study_0.8/",
     },
]
gh_etaStudy_diff_fix = [
    {"label": "DQN+LTA eta=0.2",
     "control": "data/output/test/gridhard/control/different_task/fix_rep/dqn_lta/eta_study_0.2_best/",
     "online_measure": "data/output/test/gridhard/online_property/dqn_lta/eta_study_0.2_best/"
     },
    {"label": "DQN+LTA eta=0.4",
     "control": "data/output/test/gridhard/control/different_task/fix_rep/dqn_lta/best/",
     "online_measure": "data/output/test/gridhard/online_property/dqn_lta/best/"
     },
    {"label": "DQN+LTA eta=0.6",
     "control": "data/output/test/gridhard/control/different_task/fix_rep/dqn_lta/eta_study_0.6_best/",
     "online_measure": "data/output/test/gridhard/online_property/dqn_lta/eta_study_0.6_best/"
     },
    {"label": "DQN+LTA eta=0.8",
     "control": "data/output/test/gridhard/control/different_task/fix_rep/dqn_lta/eta_study_0.8_best/",
     "online_measure": "data/output/test/gridhard/online_property/dqn_lta/eta_study_0.8_best/"
     },
]
gh_etaStudy_diff_tune = [
    {"label": "DQN+LTA eta=0.2",
     "control": "data/output/test/gridhard/control/different_task/fine_tune/dqn_lta/eta_study_0.2_best/",
     "online_measure": "data/output/test/gridhard/online_property/dqn_lta/eta_study_0.2_best/"
     },
    {"label": "DQN+LTA eta=0.4",
     "control": "data/output/test/gridhard/control/different_task/fine_tune/dqn_lta/best/",
     "online_measure": "data/output/test/gridhard/online_property/dqn_lta/best/"
    },
    {"label": "DQN+LTA eta=0.6",
     "control": "data/output/test/gridhard/control/different_task/fine_tune/dqn_lta/eta_study_0.6_best/",
     "online_measure": "data/output/test/gridhard/online_property/dqn_lta/eta_study_0.6_best/"
     },
    {"label": "DQN+LTA eta=0.8",
     "control": "data/output/test/gridhard/control/different_task/fine_tune/dqn_lta/eta_study_0.8_best/",
     "online_measure": "data/output/test/gridhard/online_property/dqn_lta/eta_study_0.8_best/"
     },
]


gh_online_sweep = [
    # {"label": "DQN",
    #  "control": "data/output/test/gridhard/online_property/different_task/fine_tune/dqn/sweep/",
    #  },
    # {"label": "DQN+LTA",
    #  "control": "data/output/test/gridhard/online_property/dqn_lta/eta_study/",
    #  },
    {"label": "DQN+Reward",
     "control": "data/output/test/gridhard/online_property/dqn_aux/reward/sweep/",
     },
    {"label": "DQN+LTA",
     "control": "data/output/test/gridhard/online_property/dqn_lta_aux/reward/sweep/",
     },
]

gh_online = [
    {"label": "DQN",
     "control": "data/output/test/gridhard/online_property/dqn/best/",
     },
    {"label": "DQN+LTA",
     "control": "data/output/test/gridhard/online_property/dqn_lta/best/",
     },
    {"label": "DQN+Reward",
     "control": "data/output/test/gridhard/online_property/dqn_aux/reward/best/",
     },
    {"label": "DQN+LTA+Reward",
     "control": "data/output/test/gridhard/online_property/dqn_lta_aux/reward/best/",
     },
]

gh_etaStudy_online = [
    {"label": "DQN+LTA eta=0.2",
     "control": "data/output/test/gridhard/online_property/dqn_lta/eta_study_0.2_best/",
     },
    {"label": "DQN+LTA eta=0.4",
     "control": "data/output/test/gridhard/online_property/dqn_lta/best/",
     },
    {"label": "DQN+LTA eta=0.6",
     "control": "data/output/test/gridhard/online_property/dqn_lta/eta_study_0.6_best/",
     },
    {"label": "DQN+LTA eta=0.8",
     "control": "data/output/test/gridhard/online_property/dqn_lta/eta_study_0.8_best/",
     },

]