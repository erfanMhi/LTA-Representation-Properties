import matplotlib
# cmap = matplotlib.cm.get_cmap('hsv')
# c_default = ['#377eb8', '#ff7f00', '#4daf4a',
#              '#f781bf', '#a65628', '#984ea3',
#              '#999999', '#e41a1c', '#dede00']

# https://personal.sron.nl/~pault/#fig:scheme_bright
c_default = ["#332288","#88CCEE", "#4A7BB7", #"#0077BB", # purple, blue
             "#44AA99","#117733","#999933", #"#D9F0D3", # green
             "#DDCC77","#EE7733", # yellow
             "#CC6677","#882255", "#AA4499","#FEDA8B",#"#F4A582",#red
             "#BBBBBB",  # grey
             ]
s_default = ["-", "--"]
m_default = ["o", "^"]

def cmap(idx, total):
    if idx < len(c_default):
        return c_default[idx]
    else:
        getc = matplotlib.cm.get_cmap('cool')
        return getc(float(idx - len(c_default))/(total- len(c_default)))


target_keywords = {
    # "decorrelation.txt": "Decorrelation:",
    "distance.txt": "Distance",
    "noninterference.txt": "Noninterference:",
    "interference.txt": "Interference:",
    "linear_probing_xy.txt": "Percentage error",
    "linear_probing_color.txt": "Percentage error",
    "linear_probing_count.txt": "Percentage error",
    "orthogonality.txt": "Orthogonality:",
    "sparsity_instance.txt": "sparsity:",
    "diversity.txt": "Diversity:",
}
target_files = {
    "diversity": "diversity.txt",
    # "decorr": "decorrelation.txt",
    "distance": "distance.txt",
    "noninterf": "noninterference.txt",
    "interf": "interference.txt",
    "ortho": "orthogonality.txt",
    "sparsity": "sparsity_instance.txt"
}

violin_colors = {
    "ReLU": c_default[0],
    "ReLU+Control": c_default[2],
    "ReLU+Control1g": c_default[2],
    "ReLU+Control5g": c_default[3],
    "ReLU+XY": c_default[4],
    "ReLU+XY+color": c_default[5],
    "ReLU+XY+count": c_default[6],
    "ReLU+Decoder": c_default[7],
    "ReLU+NAS": c_default[8],
    "ReLU+Reward": c_default[9],
    "ReLU+SF": c_default[10],

    "ReLU+LTA": c_default[0],
    "ReLU+LTA(no target)": c_default[0],

    "LTA eta=0.2": c_default[1],
    "LTA eta=0.4": c_default[10],
    "LTA eta=0.6": c_default[11],
    "LTA eta=0.8": c_default[12],

    "LTA+Control": c_default[2],
    "LTA+Control1g": c_default[2],
    "LTA+Control5g": c_default[3],
    "LTA+XY": c_default[4],
    "LTA+XY+color": c_default[5],
    "LTA+XY+count": c_default[6],
    "LTA+Decoder": c_default[7],
    "LTA+NAS": c_default[8],
    "LTA+Reward": c_default[9],
    "LTA+SF": c_default[10],

    "Laplace": "gray",
    "Scratch DQN": "black",
    "Random": c_default[12],
    "Input": c_default[11],
}

curve_styles = {
    "ReLU": s_default[1],
    "ReLU+Control": s_default[1],
    "ReLU+Control1g": s_default[1],
    "ReLU+Control5g": s_default[1],
    "ReLU+XY": s_default[1],
    "ReLU+XY+color": s_default[1],
    "ReLU+XY+count": s_default[1],
    "ReLU+Decoder": s_default[1],
    "ReLU+NAS": s_default[1],
    "ReLU+Reward": s_default[1],
    "ReLU+SF": s_default[1],

    "ReLU+LTA": s_default[0],
    "ReLU+LTA(no target)": s_default[0],

    "LTA eta=0.2": s_default[0],
    "LTA eta=0.4": s_default[0],
    "LTA eta=0.6": s_default[0],
    "LTA eta=0.8": s_default[0],

    "LTA+Control": s_default[0],
    "LTA+Control1g": s_default[0],
    "LTA+Control5g": s_default[0],
    "LTA+XY": s_default[0],
    "LTA+XY+color": s_default[0],
    "LTA+XY+count": s_default[0],
    "LTA+Decoder": s_default[0],
    "LTA+NAS": s_default[0],
    "LTA+Reward": s_default[0],
    "LTA+SF": s_default[0],

    "Random": s_default[0],
    "Input": s_default[0],
}

marker_styles = {
    "ReLU": m_default[1],
    "ReLU+Control": m_default[1],
    "ReLU+Control1g": m_default[1],
    "ReLU+Control5g": m_default[1],
    "ReLU+XY": m_default[1],
    "ReLU+XY+color": m_default[1],
    "ReLU+XY+count": m_default[1],
    "ReLU+Decoder": m_default[1],
    "ReLU+NAS": m_default[1],
    "ReLU+Reward": m_default[1],
    "ReLU+SF": m_default[1],

    "ReLU+LTA": m_default[0],
    "ReLU+LTA(no target)": m_default[0],

    "LTA eta=0.2": m_default[0],
    "LTA eta=0.4": m_default[0],
    "LTA eta=0.6": m_default[0],
    "LTA eta=0.8": m_default[0],

    "LTA+Control": m_default[0],
    "LTA+Control1g": m_default[0],
    "LTA+Control5g": m_default[0],
    "LTA+XY": m_default[0],
    "LTA+XY+color": m_default[0],
    "LTA+XY+count": m_default[0],
    "LTA+Decoder": m_default[0],
    "LTA+NAS": m_default[0],
    "LTA+Reward": m_default[0],
    "LTA+SF": m_default[0],

    "Random": m_default[0],
    "Input": m_default[0],
}

mc_learn_sweep = [
    {"label": "ReLU",
     "control": "data/output/test/mountaincar/representations/dqn/sweep/",
     },
    {"label": "ReLU+LTA",
     "control": "data/output/test/mountaincar/representations/dqn_lta/sweep/",
     },
    {"label": "ReLU+LTA(no target)",
     "control": "data/output/test/mountaincar/representations/dqn_lta/no_target/",
     },
]

gh_online_sweep = [
    {"label": "ReLU",
     "control": "data/output/test_v3/gridhard/online_property/dqn/sweep/",
     },
    {"label": "ReLU+Control1g",
     "control": "data/output/test_v3/gridhard/online_property/dqn_aux/freq8/aux_control/sweep_1g/",
     # "control": "data/output/test_v3/gridhard/online_property/dqn_aux/freq64/aux_control/sweep_1g/",
     },
    {"label": "ReLU+Control5g",
     # "control": "data/output/test_v3/gridhard/online_property/dqn_aux/freq8/aux_control/sweep_5g/",
     # "control": "data/output/test_v3/gridhard/online_property/dqn_aux/freq64/aux_control/sweep_5g/",
     "control": "data/output/test_v3/gridhard/online_property/dqn_aux/freq128/aux_control/sweep_5g/",
     # "control": "data/output/test/gridhard/online_property/dqn_aux/freq256/aux_control/sweep_5g/",
     },
    {"label": "ReLU+XY",
     "control": "data/output/test_v3/gridhard/online_property/dqn_aux/freq8/info/sweep/",
     # "control": "data/output/test_v3/gridhard/online_property/dqn_aux/freq64/info/sweep/",
     },
    {"label": "ReLU+Decoder",
     "control": "data/output/test_v3/gridhard/online_property/dqn_aux/freq8/input_decoder/sweep/",
     # "control": "data/output/test_v3/gridhard/online_property/dqn_aux/freq64/input_decoder/sweep/",
     },
    {"label": "ReLU+NAS",
     "control": "data/output/test_v3/gridhard/online_property/dqn_aux/freq8/nas_v2_delta/sweep/",
     # "control": "data/output/test_v3/gridhard/online_property/dqn_aux/freq64/nas_v2_delta/sweep/",
     },
    {"label": "ReLU+Reward",
     "control": "data/output/test_v3/gridhard/online_property/dqn_aux/freq8/reward/sweep/",
     # "control": "data/output/test_v3/gridhard/online_property/dqn_aux/freq64/reward/sweep/",
     },
    {"label": "ReLU+SF",
     "control": "data/output/test_v3/gridhard/online_property/dqn_aux/freq8/successor_as/sweep/",
     # "control": "data/output/test_v3/gridhard/online_property/dqn_aux/freq64/successor_as/sweep/",
     },
    {"label": "LTA eta=0.2",
     "control": "data/output/test_v3/gridhard/online_property/dqn_lta/eta_study_0.2_sweep/",
     },
    {"label": "LTA eta=0.4",
     "control": "data/output/test_v3/gridhard/online_property/dqn_lta/eta_study_0.4_sweep/",
     },
    {"label": "LTA eta=0.6",
     "control": "data/output/test_v3/gridhard/online_property/dqn_lta/eta_study_0.6_sweep/",
     },
    {"label": "LTA eta=0.8",
     "control": "data/output/test_v3/gridhard/online_property/dqn_lta/eta_study_0.8_sweep/",
     },
    {"label": "LTA+Control1g",
     "control": "data/output/test_v3/gridhard/online_property/dqn_lta_aux/aux_control/sweep_1g/",
     },
    {"label": "LTA+Control5g",
     "control": "data/output/test_v3/gridhard/online_property/dqn_lta_aux/aux_control/sweep_5g/",
     },
    {"label": "LTA+XY",
     "control": "data/output/test_v3/gridhard/online_property/dqn_lta_aux/info/sweep/",
     },
    {"label": "LTA+Decoder",
     "control": "data/output/test_v3/gridhard/online_property/dqn_lta_aux/input_decoder/sweep/",
     },
    {"label": "LTA+NAS",
     "control": "data/output/test_v3/gridhard/online_property/dqn_lta_aux/nas_v2_delta/sweep/",
     },
    {"label": "LTA+Reward",
     "control": "data/output/test_v3/gridhard/online_property/dqn_lta_aux/reward/sweep/",
     },
    {"label": "LTA+SF",
     "control": "data/output/test_v3/gridhard/online_property/dqn_lta_aux/successor_as/sweep/",
     }
]

gh_same_early_sweep = [
    # {"label": "ReLU",
    #  "control": "data/output/test/gridhard/control/same_task/fix_rep/dqn/sweep/",
    #  },
    # {"label": "ReLU+Control1g",
    #  "control": "data/output/test/gridhard/control/same_task/fix_rep/dqn_aux/aux_control/sweep_1g/",
    #  },
    {"label": "ReLU+Control5g",
     "control": "data/output/test/gridhard/control/early_stop/same_task/fix_rep/dqn_aux/aux_control/sweep_5g/",
     },
    # {"label": "ReLU+XY",
    #  "control": "data/output/test/gridhard/control/same_task/fix_rep/dqn_aux/info/sweep/",
    #  },
    # {"label": "ReLU+Decoder",
    #  "control": "data/output/test/gridhard/control/same_task/fix_rep/dqn_aux/input_decoder/sweep/",
    #  },
    # {"label": "ReLU+NAS",
    #  "control": "data/output/test/gridhard/control/same_task/fix_rep/dqn_aux/nas_v2_delta/sweep/",
    #  },
    # {"label": "ReLU+Reward",
    #  "control": "data/output/test/gridhard/control/same_task/fix_rep/dqn_aux/reward/sweep/",
    #  },
    # {"label": "ReLU+SF",
    #  "control": "data/output/test/gridhard/control/same_task/fix_rep/dqn_aux/successor_as/sweep/",
    #  },
    # {"label": "LTA eta=0.2",
    #  "control": "data/output/test/gridhard/control/same_task/fix_rep/dqn_lta/eta_study_0.2_sweep/",
    #  },
    # {"label": "LTA eta=0.4",
    #  "control": "data/output/test/gridhard/control/same_task/fix_rep/dqn_lta/eta_study_0.4_sweep/",
    #  },
    # {"label": "LTA eta=0.6",
    #  "control": "data/output/test/gridhard/control/same_task/fix_rep/dqn_lta/eta_study_0.6_sweep/",
    #  },
    # {"label": "LTA eta=0.8",
    #  "control": "data/output/test/gridhard/control/same_task/fix_rep/dqn_lta/eta_study_0.8_sweep/",
    #  },
    # {"label": "LTA+Control1g",
    #  "control": "data/output/test/gridhard/control/same_task/fix_rep/dqn_lta_aux/aux_control/sweep_1g/",
    #  },
    # {"label": "LTA+Control5g",
    #  "control": "data/output/test/gridhard/control/same_task/fix_rep/dqn_lta_aux/aux_control/sweep_5g/",
    #  },
    # {"label": "LTA+XY",
    #  "control": "data/output/test/gridhard/control/same_task/fix_rep/dqn_lta_aux/info/sweep/",
    #  },
    # {"label": "LTA+Decoder",
    #  "control": "data/output/test/gridhard/control/same_task/fix_rep/dqn_lta_aux/input_decoder/sweep/",
    #  },
    # {"label": "LTA+NAS",
    #  "control": "data/output/test/gridhard/control/same_task/fix_rep/dqn_lta_aux/nas_v2_delta/sweep/",
    #  },
    # {"label": "LTA+Reward",
    #  "control": "data/output/test/gridhard/control/same_task/fix_rep/dqn_lta_aux/reward/sweep/",
    #  },
    # {"label": "LTA+SF",
    #  "control": "data/output/test/gridhard/control/same_task/fix_rep/dqn_lta_aux/successor_as/sweep/",
    #  },
    # {"label": "Random",
    #  "control": "data/output/test/gridhard/control/same_task/fix_rep/random/sweep/",
    #  },
    # {"label": "Input",
    #  "control": "data/output/test/gridhard/control/same_task/fix_rep/input/sweep/",
    #  },
]

gh_similar_early_sweep = [
    # {"label": "ReLU",
    #  "control": "data/output/test/gridhard/control/similar_task/fix_rep/dqn/sweep/",
    #  },
    # {"label": "ReLU+Control1g",
    #  "control": "data/output/test/gridhard/control/similar_task/fix_rep/dqn_aux/aux_control/sweep_1g/",
    #  },
    {"label": "ReLU+Control5g",
     "control": "data/output/test/gridhard/control/early_stop/similar_task/fix_rep/dqn_aux/aux_control/sweep_5g/",
     },
    # {"label": "ReLU+XY",
    #  "control": "data/output/test/gridhard/control/similar_task/fix_rep/dqn_aux/info/sweep/",
    #  },
    # {"label": "ReLU+Decoder",
    #  "control": "data/output/test/gridhard/control/similar_task/fix_rep/dqn_aux/input_decoder/sweep/",
    #  },
    # {"label": "ReLU+NAS",
    #  "control": "data/output/test/gridhard/control/similar_task/fix_rep/dqn_aux/nas_v2_delta/sweep/",
    #  },
    # {"label": "ReLU+Reward",
    #  "control": "data/output/test/gridhard/control/similar_task/fix_rep/dqn_aux/reward/sweep/",
    #  },
    # {"label": "ReLU+SF",
    #  "control": "data/output/test/gridhard/control/similar_task/fix_rep/dqn_aux/successor_as/sweep/",
    #  },
    # {"label": "LTA eta=0.2",
    #  "control": "data/output/test/gridhard/control/similar_task/fix_rep/dqn_lta/eta_study_0.2_sweep/",
    #  },
    # {"label": "LTA eta=0.4",
    #  "control": "data/output/test/gridhard/control/similar_task/fix_rep/dqn_lta/eta_study_0.4_sweep/",
    #  },
    # {"label": "LTA eta=0.6",
    #  "control": "data/output/test/gridhard/control/similar_task/fix_rep/dqn_lta/eta_study_0.6_sweep/",
    #  },
    # {"label": "LTA eta=0.8",
    #  "control": "data/output/test/gridhard/control/similar_task/fix_rep/dqn_lta/eta_study_0.8_sweep/",
    #  },
    # {"label": "LTA+Control1g",
    #  "control": "data/output/test/gridhard/control/similar_task/fix_rep/dqn_lta_aux/aux_control/sweep_1g/",
    #  },
    # {"label": "LTA+Control5g",
    #  "control": "data/output/test/gridhard/control/similar_task/fix_rep/dqn_lta_aux/aux_control/sweep_5g/",
    #  },
    # {"label": "LTA+XY",
    #  "control": "data/output/test/gridhard/control/similar_task/fix_rep/dqn_lta_aux/info/sweep/",
    #  },
    # {"label": "LTA+Decoder",
    #  "control": "data/output/test/gridhard/control/similar_task/fix_rep/dqn_lta_aux/input_decoder/sweep/",
    #  },
    # {"label": "LTA+NAS",
    #  "control": "data/output/test/gridhard/control/similar_task/fix_rep/dqn_lta_aux/nas_v2_delta/sweep/",
    #  },
    # {"label": "LTA+Reward",
    #  "control": "data/output/test/gridhard/control/similar_task/fix_rep/dqn_lta_aux/reward/sweep/",
    #  },
    # {"label": "LTA+SF",
    #  "control": "data/output/test/gridhard/control/similar_task/fix_rep/dqn_lta_aux/successor_as/sweep/",
    #  },
    # {"label": "Random",
    #  "control": "data/output/test/gridhard/control/similar_task/fix_rep/random/sweep/",
    #  },
    # {"label": "Input",
    #  "control": "data/output/test/gridhard/control/similar_task/fix_rep/input/sweep/",
    #  },
]

gh_diff_early_sweep = [
    # {"label": "ReLU",
    #  "control": "data/output/test_v3/gridhard/control/different_task/fix_rep/dqn/sweep/",
    #  },
    # {"label": "DQN+Control1g",
    #  "control": "data/output/test_v3/gridhard/control/different_task/fix_rep/dqn_aux/aux_control/sweep_1g/",
    #  },
    {"label": "DQN+Control5g",
     "control": "data/output/test/gridhard/control/early_stop/different_task/fix_rep/dqn_aux/aux_control/sweep_5g/",
     },
    # {"label": "DQN+XY",
    #  "control": "data/output/test_v3/gridhard/control/different_task/fix_rep/dqn_aux/info/sweep/",
    #  },
    # {"label": "DQN+Decoder",
    #  "control": "data/output/test_v3/gridhard/control/different_task/fix_rep/dqn_aux/input_decoder/sweep/",
    #  },
    # {"label": "DQN+NAS",
    #  "control": "data/output/test_v3/gridhard/control/different_task/fix_rep/dqn_aux/nas_v2_delta/sweep/",
    #  },
    # {"label": "DQN+Reward",
    #  "control": "data/output/test_v3/gridhard/control/different_task/fix_rep/dqn_aux/reward/sweep/",
    #  },
    # {"label": "DQN+SF",
    #  "control": "data/output/test_v3/gridhard/control/different_task/fix_rep/dqn_aux/successor_as/sweep/",
    #  },
    # {"label": "LTA eta=0.2",
    #  "control": "data/output/test_v3/gridhard/control/different_task/fix_rep/dqn_lta/eta_study_0.2_sweep/",
    #  },
    # {"label": "LTA eta=0.4",
    #  "control": "data/output/test_v3/gridhard/control/different_task/fix_rep/dqn_lta/eta_study_0.4_sweep/",
    #  },
    # {"label": "LTA eta=0.6",
    #  "control": "data/output/test_v3/gridhard/control/different_task/fix_rep/dqn_lta/eta_study_0.6_sweep/",
    #  },
    # {"label": "LTA eta=0.8",
    #  "control": "data/output/test_v3/gridhard/control/different_task/fix_rep/dqn_lta/eta_study_0.8_sweep/",
    #  },
    # {"label": "DQN+LTA+Control1g",
    #  "control": "data/output/test_v3/gridhard/control/different_task/fix_rep/dqn_lta_aux/aux_control/sweep_1g/",
    #  },
    # {"label": "DQN+LTA+Control5g",
    #  "control": "data/output/test_v3/gridhard/control/different_task/fix_rep/dqn_lta_aux/aux_control/sweep_5g/",
    #  },
    # {"label": "LTA+XY",
    #  "control": "data/output/test_v3/gridhard/control/different_task/fix_rep/dqn_lta_aux/info/sweep/",
    #  },
    # {"label": "LTA+Decoder",
    #  "control": "data/output/test_v3/gridhard/control/different_task/fix_rep/dqn_lta_aux/input_decoder/sweep/",
    #  },
    # {"label": "LTA+NAS",
    #  "control": "data/output/test_v3/gridhard/control/different_task/fix_rep/dqn_lta_aux/nas_v2_delta/sweep/",
    #  },
    # {"label": "LTA+Reward",
    #  "control": "data/output/test_v3/gridhard/control/different_task/fix_rep/dqn_lta_aux/reward/sweep/",
    #  },
    # {"label": "LTA+SF",
    #  "control": "data/output/test_v3/gridhard/control/different_task/fix_rep/dqn_lta_aux/successor_as/sweep/",
    #  },
    # {"label": "Random",
    #  "control": "data/output/test_v3/gridhard/control/different_task/fix_rep/random/sweep/",
    #  },
    # {"label": "Input",
    #  "control": "data/output/test_v3/gridhard/control/different_task/fix_rep/input/sweep/",
    #  },
]

gh_diff_tune_early_sweep = [
    # {"label": "ReLU",
    #  "control": "data/output/test/gridhard/control/different_task/fine_tune/dqn/sweep/",
    #  },
    # {"label": "DQN+Control1g",
    #  "control": "data/output/test/gridhard/control/different_task/fine_tune/dqn_aux/aux_control/sweep_1g/",
    #  },
    {"label": "DQN+Control5g",
     "control": "data/output/test/gridhard/control/early_stop/different_task/fine_tune/dqn_aux/aux_control/sweep_5g/",
     },
    # {"label": "DQN+XY",
    #  "control": "data/output/test/gridhard/control/different_task/fine_tune/dqn_aux/info/sweep/",
    #  },
    # {"label": "DQN+Decoder",
    #  "control": "data/output/test/gridhard/control/different_task/fine_tune/dqn_aux/input_decoder/sweep/",
    #  },
    # {"label": "DQN+NAS",
    #  "control": "data/output/test/gridhard/control/different_task/fine_tune/dqn_aux/nas_v2_delta/sweep/",
    #  },
    # {"label": "DQN+Reward",
    #  "control": "data/output/test/gridhard/control/different_task/fine_tune/dqn_aux/reward/sweep/",
    #  },
    # {"label": "DQN+SF",
    #  "control": "data/output/test/gridhard/control/different_task/fine_tune/dqn_aux/successor_as/sweep/",
    #  },
    # {"label": "LTA eta=0.2",
    #  "control": "data/output/test/gridhard/control/different_task/fine_tune/dqn_lta/eta_study_0.2_sweep/",
    #  },
    # {"label": "LTA eta=0.4",
    #  "control": "data/output/test/gridhard/control/different_task/fine_tune/dqn_lta/eta_study_0.4_sweep/",
    #  },
    # {"label": "LTA eta=0.6",
    #  "control": "data/output/test/gridhard/control/different_task/fine_tune/dqn_lta/eta_study_0.6_sweep/",
    #  },
    # {"label": "LTA eta=0.8",
    #  "control": "data/output/test/gridhard/control/different_task/fine_tune/dqn_lta/eta_study_0.8_sweep/",
    #  },
    # {"label": "DQN+LTA+Control1g",
    #  "control": "data/output/test/gridhard/control/different_task/fine_tune/dqn_lta_aux/aux_control/sweep_1g/",
    #  },
    # {"label": "DQN+LTA+Control5g",
    #  "control": "data/output/test/gridhard/control/different_task/fine_tune/dqn_lta_aux/aux_control/sweep_5g/",
    #  },
    # {"label": "LTA+XY",
    #  "control": "data/output/test/gridhard/control/different_task/fine_tune/dqn_lta_aux/info/sweep/",
    #  },
    # {"label": "LTA+Decoder",
    #  "control": "data/output/test/gridhard/control/different_task/fine_tune/dqn_lta_aux/input_decoder/sweep/",
    #  },
    # {"label": "LTA+NAS",
    #  "control": "data/output/test/gridhard/control/different_task/fine_tune/dqn_lta_aux/nas_v2_delta/sweep/",
    #  },
    # {"label": "LTA+Reward",
    #  "control": "data/output/test/gridhard/control/different_task/fine_tune/dqn_lta_aux/reward/sweep/",
    #  },
    # {"label": "LTA+SF",
    #  "control": "data/output/test/gridhard/control/different_task/fine_tune/dqn_lta_aux/successor_as/sweep/",
    #  },
    # {"label": "Random",
    #  "control": "data/output/test/gridhard/control/different_task/fine_tune/random/sweep/",
    #  },
]

gh_online = [
    {"label": "ReLU",
     "control": "data/output/test/gridhard/online_property/dqn/best/",
     },
    {"label": "DQN+Control1g",
     "control": "data/output/test/gridhard/online_property/dqn_aux/freq8/aux_control/best_1g/",
     },
    {"label": "DQN+Control5g",
     "control": "data/output/test/gridhard/online_property/dqn_aux/freq128/aux_control/best_5g/",
     },
    {"label": "DQN+XY",
     "control": "data/output/test/gridhard/online_property/dqn_aux/freq8/info/best/",
     },
    {"label": "DQN+Decoder",
     "control": "data/output/test/gridhard/online_property/dqn_aux/freq8/input_decoder/best/",
     },
    {"label": "DQN+NAS",
     "control": "data/output/test/gridhard/online_property/dqn_aux/freq8/nas_v2_delta/best/",
     },
    {"label": "DQN+Reward",
     "control": "data/output/test/gridhard/online_property/dqn_aux/freq8/reward/best/",
     },
    {"label": "DQN+SF",
     "control": "data/output/test/gridhard/online_property/dqn_aux/freq8/successor_as/best/",
     },
    {"label": "LTA eta=0.2",
     "control": "data/output/test/gridhard/online_property/dqn_lta/eta_study_0.2_best/",
     },
    {"label": "LTA eta=0.4",
     "control": "data/output/test/gridhard/online_property/dqn_lta/eta_study_0.4_best/",
     },
    {"label": "LTA eta=0.6",
     "control": "data/output/test/gridhard/online_property/dqn_lta/eta_study_0.6_best/",
     },
    {"label": "LTA eta=0.8",
     "control": "data/output/test/gridhard/online_property/dqn_lta/eta_study_0.8_best/",
     },
    {"label": "DQN+LTA+Control1g",
     "control": "data/output/test/gridhard/online_property/dqn_lta_aux/aux_control/best_1g/",
     },
    {"label": "DQN+LTA+Control5g",
     "control": "data/output/test/gridhard/online_property/dqn_lta_aux/aux_control/best_5g/",
     },
    {"label": "LTA+XY",
     "control": "data/output/test/gridhard/online_property/dqn_lta_aux/info/best/",
     },
    {"label": "LTA+Decoder",
     "control": "data/output/test/gridhard/online_property/dqn_lta_aux/input_decoder/best/",
     },
    {"label": "LTA+NAS",
     "control": "data/output/test/gridhard/online_property/dqn_lta_aux/nas_v2_delta/best/",
     },
    {"label": "LTA+Reward",
     "control": "data/output/test/gridhard/online_property/dqn_lta_aux/reward/best/",
     },
    {"label": "LTA+SF",
     "control": "data/output/test/gridhard/online_property/dqn_lta_aux/successor_as/best/",
     }
]

gh_same_early = [
    {"label": "ReLU", "property": "data/output/test/",
     "control": "data/output/test/gridhard/control/early_stop/same_task/fix_rep/dqn/best/",
     "online_measure": "data/output/test/gridhard/online_property/dqn/best/"
     },
    {"label": "DQN+Control1g",
     "control": "data/output/test/gridhard/control/early_stop/same_task/fix_rep/dqn_aux/aux_control/best_1g/",
     "online_measure": "data/output/test/gridhard/online_property/dqn_aux/freq8/aux_control/best_1g/"
     },
    {"label": "DQN+Control5g",
     "control": "data/output/test/gridhard/control/early_stop/same_task/fix_rep/dqn_aux/aux_control/best_5g/",
     "online_measure": "data/output/test/gridhard/online_property/dqn_aux/freq128/aux_control/best_5g/"
     },
    {"label": "DQN+XY",
     "control": "data/output/test/gridhard/control/early_stop/same_task/fix_rep/dqn_aux/info/best/",
     "online_measure": "data/output/test/gridhard/online_property/dqn_aux/freq8/info/best/"
     },
    {"label": "DQN+Decoder",
     "control": "data/output/test/gridhard/control/early_stop/same_task/fix_rep/dqn_aux/input_decoder/best/",
     "online_measure": "data/output/test/gridhard/online_property/dqn_aux/freq8/info/best/"
     },
    {"label": "DQN+NAS",
     "control": "data/output/test/gridhard/control/early_stop/same_task/fix_rep/dqn_aux/nas_v2_delta/best/",
     "online_measure": "data/output/test/gridhard/online_property/dqn_aux/freq8/nas_v2_delta/best/"
     },
    {"label": "DQN+Reward",
     "control": "data/output/test/gridhard/control/early_stop/same_task/fix_rep/dqn_aux/reward/best/",
     "online_measure": "data/output/test/gridhard/online_property/dqn_aux/freq8/reward/best/"
     },
    {"label": "DQN+SF",
     "control": "data/output/test/gridhard/control/early_stop/same_task/fix_rep/dqn_aux/successor_as/best/",
     "online_measure": "data/output/test/gridhard/online_property/dqn_aux/freq8/successor_as/best/"
     },
    {"label": "LTA eta=0.2",
     "control": "data/output/test/gridhard/control/early_stop/same_task/fix_rep/dqn_lta/eta_study_0.2_best/",
     "online_measure": "data/output/test/gridhard/online_property/dqn_lta/eta_study_0.2_best/"
     },
    {"label": "LTA eta=0.4",
     "control": "data/output/test/gridhard/control/early_stop/same_task/fix_rep/dqn_lta/eta_study_0.4_best/",
     "online_measure": "data/output/test/gridhard/online_property/dqn_lta/eta_study_0.4_best/"
     },
    {"label": "LTA eta=0.6",
     "control": "data/output/test/gridhard/control/early_stop/same_task/fix_rep/dqn_lta/eta_study_0.6_best/",
     "online_measure": "data/output/test/gridhard/online_property/dqn_lta/eta_study_0.6_best/"
     },
    {"label": "LTA eta=0.8",
     "control": "data/output/test/gridhard/control/early_stop/same_task/fix_rep/dqn_lta/eta_study_0.8_best/",
     "online_measure": "data/output/test/gridhard/online_property/dqn_lta/eta_study_0.8_best/"
     },
    {"label": "DQN+LTA+Control1g",
     "control": "data/output/test/gridhard/control/early_stop/same_task/fix_rep/dqn_lta_aux/aux_control/best_1g/",
     "online_measure": "data/output/test/gridhard/online_property/dqn_lta_aux/aux_control/best_1g/"
     },
    {"label": "DQN+LTA+Control5g",
     "control": "data/output/test/gridhard/control/early_stop/same_task/fix_rep/dqn_lta_aux/aux_control/best_5g/",
     "online_measure": "data/output/test/gridhard/online_property/dqn_lta_aux/aux_control/best_5g/"
     },
    {"label": "LTA+XY",
     "control": "data/output/test/gridhard/control/early_stop/same_task/fix_rep/dqn_lta_aux/info/best/",
     "online_measure": "data/output/test/gridhard/online_property/dqn_lta_aux/info/best/"
     },
    {"label": "LTA+Decoder",
     "control": "data/output/test/gridhard/control/early_stop/same_task/fix_rep/dqn_lta_aux/input_decoder/best/",
     "online_measure": "data/output/test/gridhard/online_property/dqn_lta_aux/input_decoder/best/"
     },
    {"label": "LTA+NAS",
     "control": "data/output/test/gridhard/control/early_stop/same_task/fix_rep/dqn_lta_aux/nas_v2_delta/best/",
     "online_measure": "data/output/test/gridhard/online_property/dqn_lta_aux/nas_v2_delta/best/"
     },
    {"label": "LTA+Reward",
     "control": "data/output/test/gridhard/control/early_stop/same_task/fix_rep/dqn_lta_aux/reward/best/",
     "online_measure": "data/output/test/gridhard/online_property/dqn_lta_aux/reward/best/"
     },
    {"label": "LTA+SF",
     "control": "data/output/test/gridhard/control/early_stop/same_task/fix_rep/dqn_lta_aux/successor_as/best/",
     "online_measure": "data/output/test/gridhard/online_property/dqn_lta_aux/successor_as/best/"
     },
    {"label": "Random",
     "control": "data/output/test/gridhard/control/baseline/same_task/fix_rep/random/best/",
     "online_measure": "data/output/test/gridhard/fixrep_property/random/best",
     },
    {"label": "Input",
     "control": "data/output/test/gridhard/control/baseline/same_task/fix_rep/input/best/",
     "online_measure": "data/output/test/gridhard/fixrep_property/input/best",
     },
]

gh_similar_early = [
    {"label": "ReLU", "property": "data/output/test/",
     "control": "data/output/test/gridhard/control/early_stop/similar_task/fix_rep/dqn/best/",
     "online_measure": "data/output/test/gridhard/online_property/dqn/best/"
     },
    {"label": "DQN+Control1g",
     "control": "data/output/test/gridhard/control/early_stop/similar_task/fix_rep/dqn_aux/aux_control/best_1g/",
     "online_measure": "data/output/test/gridhard/online_property/dqn_aux/freq8/aux_control/best_1g/"
     },
    {"label": "DQN+Control5g",
     "control": "data/output/test/gridhard/control/early_stop/similar_task/fix_rep/dqn_aux/aux_control/best_5g/",
     "online_measure": "data/output/test/gridhard/online_property/dqn_aux/freq128/aux_control/best_5g/"
     },
    {"label": "DQN+XY",
     "control": "data/output/test/gridhard/control/early_stop/similar_task/fix_rep/dqn_aux/info/best/",
     "online_measure": "data/output/test/gridhard/online_property/dqn_aux/freq8/info/best/"
     },
    {"label": "DQN+Decoder",
     "control": "data/output/test/gridhard/control/early_stop/similar_task/fix_rep/dqn_aux/input_decoder/best/",
     "online_measure": "data/output/test/gridhard/online_property/dqn_aux/freq8/info/best/"
     },
    {"label": "DQN+NAS",
     "control": "data/output/test/gridhard/control/early_stop/similar_task/fix_rep/dqn_aux/nas_v2_delta/best/",
     "online_measure": "data/output/test/gridhard/online_property/dqn_aux/freq8/nas_v2_delta/best/"
     },
    {"label": "DQN+Reward",
     "control": "data/output/test/gridhard/control/early_stop/similar_task/fix_rep/dqn_aux/reward/best/",
     "online_measure": "data/output/test/gridhard/online_property/dqn_aux/freq8/reward/best/"
     },
    {"label": "DQN+SF",
     "control": "data/output/test/gridhard/control/early_stop/similar_task/fix_rep/dqn_aux/successor_as/best/",
     "online_measure": "data/output/test/gridhard/online_property/dqn_aux/freq8/successor_as/best/"
     },
    {"label": "LTA eta=0.2",
     "control": "data/output/test/gridhard/control/early_stop/similar_task/fix_rep/dqn_lta/eta_study_0.2_best/",
     "online_measure": "data/output/test/gridhard/online_property/dqn_lta/eta_study_0.2_best/"
     },
    {"label": "LTA eta=0.4",
     "control": "data/output/test/gridhard/control/early_stop/similar_task/fix_rep/dqn_lta/eta_study_0.4_best/",
     "online_measure": "data/output/test/gridhard/online_property/dqn_lta/eta_study_0.4_best/"
     },
    {"label": "LTA eta=0.6",
     "control": "data/output/test/gridhard/control/early_stop/similar_task/fix_rep/dqn_lta/eta_study_0.6_best/",
     "online_measure": "data/output/test/gridhard/online_property/dqn_lta/eta_study_0.6_best/"
     },
    {"label": "LTA eta=0.8",
     "control": "data/output/test/gridhard/control/early_stop/similar_task/fix_rep/dqn_lta/eta_study_0.8_best/",
     "online_measure": "data/output/test/gridhard/online_property/dqn_lta/eta_study_0.8_best/"
     },
    {"label": "DQN+LTA+Control1g",
     "control": "data/output/test/gridhard/control/early_stop/similar_task/fix_rep/dqn_lta_aux/aux_control/best_1g/",
     "online_measure": "data/output/test/gridhard/online_property/dqn_lta_aux/aux_control/best_1g/"
     },
    {"label": "DQN+LTA+Control5g",
     "control": "data/output/test/gridhard/control/early_stop/similar_task/fix_rep/dqn_lta_aux/aux_control/best_5g/",
     "online_measure": "data/output/test/gridhard/online_property/dqn_lta_aux/aux_control/best_5g/"
     },
    {"label": "LTA+XY",
     "control": "data/output/test/gridhard/control/early_stop/similar_task/fix_rep/dqn_lta_aux/info/best/",
     "online_measure": "data/output/test/gridhard/online_property/dqn_lta_aux/info/best/"
     },
    {"label": "LTA+Decoder",
     "control": "data/output/test/gridhard/control/early_stop/similar_task/fix_rep/dqn_lta_aux/input_decoder/best/",
     "online_measure": "data/output/test/gridhard/online_property/dqn_lta_aux/input_decoder/best/"
     },
    {"label": "LTA+NAS",
     "control": "data/output/test/gridhard/control/early_stop/similar_task/fix_rep/dqn_lta_aux/nas_v2_delta/best/",
     "online_measure": "data/output/test/gridhard/online_property/dqn_lta_aux/nas_v2_delta/best/"
     },
    {"label": "LTA+Reward",
     "control": "data/output/test/gridhard/control/early_stop/similar_task/fix_rep/dqn_lta_aux/reward/best/",
     "online_measure": "data/output/test/gridhard/online_property/dqn_lta_aux/reward/best/"
     },
    {"label": "LTA+SF",
     "control": "data/output/test/gridhard/control/early_stop/similar_task/fix_rep/dqn_lta_aux/successor_as/best/",
     "online_measure": "data/output/test/gridhard/online_property/dqn_lta_aux/successor_as/best/"
     },
    {"label": "Random",
     "control": "data/output/test/gridhard/control/baseline/similar_task/fix_rep/random/best/",
     "online_measure": "data/output/test/gridhard/fixrep_property/random/best",
     },
    {"label": "Input",
     "control": "data/output/test/gridhard/control/baseline/similar_task/fix_rep/input/best/",
     "online_measure": "data/output/test/gridhard/fixrep_property/input/best",
     },
]

gh_diff_early = [
    {"label": "ReLU", "property": "data/output/test/",
     "control": "data/output/test/gridhard/control/early_stop/different_task/fix_rep/dqn/best/",
     "online_measure": "data/output/test/gridhard/online_property/dqn/best/"
     },
    {"label": "DQN+Control1g",
     "control": "data/output/test/gridhard/control/early_stop/different_task/fix_rep/dqn_aux/aux_control/best_1g/",
     "online_measure": "data/output/test/gridhard/online_property/dqn_aux/freq8/aux_control/best_1g/"
     },
    {"label": "DQN+Control5g",
     "control": "data/output/test/gridhard/control/early_stop/different_task/fix_rep/dqn_aux/aux_control/best_5g/",
     "online_measure": "data/output/test/gridhard/online_property/dqn_aux/freq128/aux_control/best_5g/"
     },
    {"label": "DQN+XY",
     "control": "data/output/test/gridhard/control/early_stop/different_task/fix_rep/dqn_aux/info/best/",
     "online_measure": "data/output/test/gridhard/online_property/dqn_aux/freq8/info/best/"
     },
    {"label": "DQN+Decoder",
     "control": "data/output/test/gridhard/control/early_stop/different_task/fix_rep/dqn_aux/input_decoder/best/",
     "online_measure": "data/output/test/gridhard/online_property/dqn_aux/freq8/info/best/"
     },
    {"label": "DQN+NAS",
     "control": "data/output/test/gridhard/control/early_stop/different_task/fix_rep/dqn_aux/nas_v2_delta/best/",
     "online_measure": "data/output/test/gridhard/online_property/dqn_aux/freq8/nas_v2_delta/best/"
     },
    {"label": "DQN+Reward",
     "control": "data/output/test/gridhard/control/early_stop/different_task/fix_rep/dqn_aux/reward/best/",
     "online_measure": "data/output/test/gridhard/online_property/dqn_aux/freq8/reward/best/"
     },
    {"label": "DQN+SF",
     "control": "data/output/test/gridhard/control/early_stop/different_task/fix_rep/dqn_aux/successor_as/best/",
     "online_measure": "data/output/test/gridhard/online_property/dqn_aux/freq8/successor_as/best/"
     },
    {"label": "LTA eta=0.2",
     "control": "data/output/test/gridhard/control/early_stop/different_task/fix_rep/dqn_lta/eta_study_0.2_best/",
     "online_measure": "data/output/test/gridhard/online_property/dqn_lta/eta_study_0.2_best/"
     },
    {"label": "LTA eta=0.4",
     "control": "data/output/test/gridhard/control/early_stop/different_task/fix_rep/dqn_lta/eta_study_0.4_best/",
     "online_measure": "data/output/test/gridhard/online_property/dqn_lta/eta_study_0.4_best/"
     },
    {"label": "LTA eta=0.6",
     "control": "data/output/test/gridhard/control/early_stop/different_task/fix_rep/dqn_lta/eta_study_0.6_best/",
     "online_measure": "data/output/test/gridhard/online_property/dqn_lta/eta_study_0.6_best/"
     },
    {"label": "LTA eta=0.8",
     "control": "data/output/test/gridhard/control/early_stop/different_task/fix_rep/dqn_lta/eta_study_0.8_best/",
     "online_measure": "data/output/test/gridhard/online_property/dqn_lta/eta_study_0.8_best/"
     },
    {"label": "DQN+LTA+Control1g",
     "control": "data/output/test/gridhard/control/early_stop/different_task/fix_rep/dqn_lta_aux/aux_control/best_1g/",
     "online_measure": "data/output/test/gridhard/online_property/dqn_lta_aux/aux_control/best_1g/"
     },
    {"label": "DQN+LTA+Control5g",
     "control": "data/output/test/gridhard/control/early_stop/different_task/fix_rep/dqn_lta_aux/aux_control/best_5g/",
     "online_measure": "data/output/test/gridhard/online_property/dqn_lta_aux/aux_control/best_5g/"
     },
    {"label": "LTA+XY",
     "control": "data/output/test/gridhard/control/early_stop/different_task/fix_rep/dqn_lta_aux/info/best/",
     "online_measure": "data/output/test/gridhard/online_property/dqn_lta_aux/info/best/"
     },
    {"label": "LTA+Decoder",
     "control": "data/output/test/gridhard/control/early_stop/different_task/fix_rep/dqn_lta_aux/input_decoder/best/",
     "online_measure": "data/output/test/gridhard/online_property/dqn_lta_aux/input_decoder/best/"
     },
    {"label": "LTA+NAS",
     "control": "data/output/test/gridhard/control/early_stop/different_task/fix_rep/dqn_lta_aux/nas_v2_delta/best/",
     "online_measure": "data/output/test/gridhard/online_property/dqn_lta_aux/nas_v2_delta/best/"
     },
    {"label": "LTA+Reward",
     "control": "data/output/test/gridhard/control/early_stop/different_task/fix_rep/dqn_lta_aux/reward/best/",
     "online_measure": "data/output/test/gridhard/online_property/dqn_lta_aux/reward/best/"
     },
    {"label": "LTA+SF",
     "control": "data/output/test/gridhard/control/early_stop/different_task/fix_rep/dqn_lta_aux/successor_as/best/",
     "online_measure": "data/output/test/gridhard/online_property/dqn_lta_aux/successor_as/best/"
     },
    {"label": "Random",
     "control": "data/output/test/gridhard/control/baseline/different_task/fix_rep/random/best/",
     "online_measure": "data/output/test/gridhard/fixrep_property/random/best",
     },
    {"label": "Input",
     "control": "data/output/test/gridhard/control/baseline/different_task/fix_rep/input/best/",
     "online_measure": "data/output/test/gridhard/fixrep_property/input/best",
     },
]

gh_diff_tune_early = [
    {"label": "ReLU", "property": "data/output/test/",
     "control": "data/output/test/gridhard/control/early_stop/different_task/fine_tune/dqn/best/",
     "online_measure": "data/output/test/gridhard/online_property/dqn/best/"
     },
    {"label": "DQN+Control1g",
     "control": "data/output/test/gridhard/control/early_stop/different_task/fine_tune/dqn_aux/aux_control/best_1g/",
     "online_measure": "data/output/test/gridhard/online_property/dqn_aux/freq8/aux_control/best_1g/"
     },
    {"label": "DQN+Control5g",
     "control": "data/output/test/gridhard/control/early_stop/different_task/fine_tune/dqn_aux/aux_control/best_5g/",
     "online_measure": "data/output/test/gridhard/online_property/dqn_aux/freq128/aux_control/best_5g/"
     },
    {"label": "DQN+XY",
     "control": "data/output/test/gridhard/control/early_stop/different_task/fine_tune/dqn_aux/info/best/",
     "online_measure": "data/output/test/gridhard/online_property/dqn_aux/freq8/info/best/"
     },
    {"label": "DQN+Decoder",
     "control": "data/output/test/gridhard/control/early_stop/different_task/fine_tune/dqn_aux/input_decoder/best/",
     "online_measure": "data/output/test/gridhard/online_property/dqn_aux/freq8/info/best/"
     },
    {"label": "DQN+NAS",
     "control": "data/output/test/gridhard/control/early_stop/different_task/fine_tune/dqn_aux/nas_v2_delta/best/",
     "online_measure": "data/output/test/gridhard/online_property/dqn_aux/freq8/nas_v2_delta/best/"
     },
    {"label": "DQN+Reward",
     "control": "data/output/test/gridhard/control/early_stop/different_task/fine_tune/dqn_aux/reward/best/",
     "online_measure": "data/output/test/gridhard/online_property/dqn_aux/freq8/reward/best/"
     },
    {"label": "DQN+SF",
     "control": "data/output/test/gridhard/control/early_stop/different_task/fine_tune/dqn_aux/successor_as/best/",
     "online_measure": "data/output/test/gridhard/online_property/dqn_aux/freq8/successor_as/best/"
     },
    {"label": "LTA eta=0.2",
     "control": "data/output/test/gridhard/control/early_stop/different_task/fine_tune/dqn_lta/eta_study_0.2_best/",
     "online_measure": "data/output/test/gridhard/online_property/dqn_lta/eta_study_0.2_best/"
     },
    {"label": "LTA eta=0.4",
     "control": "data/output/test/gridhard/control/early_stop/different_task/fine_tune/dqn_lta/eta_study_0.4_best/",
     "online_measure": "data/output/test/gridhard/online_property/dqn_lta/eta_study_0.4_best/"
     },
    {"label": "LTA eta=0.6",
     "control": "data/output/test/gridhard/control/early_stop/different_task/fine_tune/dqn_lta/eta_study_0.6_best/",
     "online_measure": "data/output/test/gridhard/online_property/dqn_lta/eta_study_0.6_best/"
     },
    {"label": "LTA eta=0.8",
     "control": "data/output/test/gridhard/control/early_stop/different_task/fine_tune/dqn_lta/eta_study_0.8_best/",
     "online_measure": "data/output/test/gridhard/online_property/dqn_lta/eta_study_0.8_best/"
     },
    {"label": "DQN+LTA+Control1g",
     "control": "data/output/test/gridhard/control/early_stop/different_task/fine_tune/dqn_lta_aux/aux_control/best_1g/",
     "online_measure": "data/output/test/gridhard/online_property/dqn_lta_aux/aux_control/best_1g/"
     },
    {"label": "DQN+LTA+Control5g",
     "control": "data/output/test/gridhard/control/early_stop/different_task/fine_tune/dqn_lta_aux/aux_control/best_5g/",
     "online_measure": "data/output/test/gridhard/online_property/dqn_lta_aux/aux_control/best_5g/"
     },
    {"label": "LTA+XY",
     "control": "data/output/test/gridhard/control/early_stop/different_task/fine_tune/dqn_lta_aux/info/best/",
     "online_measure": "data/output/test/gridhard/online_property/dqn_lta_aux/info/best/"
     },
    {"label": "LTA+Decoder",
     "control": "data/output/test/gridhard/control/early_stop/different_task/fine_tune/dqn_lta_aux/input_decoder/best/",
     "online_measure": "data/output/test/gridhard/online_property/dqn_lta_aux/input_decoder/best/"
     },
    {"label": "LTA+NAS",
     "control": "data/output/test/gridhard/control/early_stop/different_task/fine_tune/dqn_lta_aux/nas_v2_delta/best/",
     "online_measure": "data/output/test/gridhard/online_property/dqn_lta_aux/nas_v2_delta/best/"
     },
    {"label": "LTA+Reward",
     "control": "data/output/test/gridhard/control/early_stop/different_task/fine_tune/dqn_lta_aux/reward/best/",
     "online_measure": "data/output/test/gridhard/online_property/dqn_lta_aux/reward/best/"
     },
    {"label": "LTA+SF",
     "control": "data/output/test/gridhard/control/early_stop/different_task/fine_tune/dqn_lta_aux/successor_as/best/",
     "online_measure": "data/output/test/gridhard/online_property/dqn_lta_aux/successor_as/best/"
     },
    {"label": "Random",
     "control": "data/output/test/gridhard/control/baseline/different_task/fine_tune/random/best/",
     "online_measure": "data/output/test/gridhard/fixrep_property/random/best",
     },
]

gh_same_last = [
    {"label": "ReLU", "property": "data/output/test/",
     "control": "data/output/test/gridhard/control/last/same_task/fix_rep/dqn/best/",
     "online_measure": "data/output/test/gridhard/online_property/dqn/best/"
     },
    {"label": "DQN+Control1g",
     "control": "data/output/test/gridhard/control/last/same_task/fix_rep/dqn_aux/aux_control/best_1g/",
     "online_measure": "data/output/test/gridhard/online_property/dqn_aux/freq8/aux_control/best_1g/"
     },
    {"label": "DQN+Control5g",
     "control": "data/output/test/gridhard/control/last/same_task/fix_rep/dqn_aux/aux_control/best_5g/",
     "online_measure": "data/output/test/gridhard/online_property/dqn_aux/freq128/aux_control/best_5g/"
     },
    {"label": "DQN+XY",
     "control": "data/output/test/gridhard/control/last/same_task/fix_rep/dqn_aux/info/best/",
     "online_measure": "data/output/test/gridhard/online_property/dqn_aux/freq8/info/best/"
     },
    {"label": "DQN+Decoder",
     "control": "data/output/test/gridhard/control/last/same_task/fix_rep/dqn_aux/input_decoder/best/",
     "online_measure": "data/output/test/gridhard/online_property/dqn_aux/freq8/info/best/"
     },
    {"label": "DQN+NAS",
     "control": "data/output/test/gridhard/control/last/same_task/fix_rep/dqn_aux/nas_v2_delta/best/",
     "online_measure": "data/output/test/gridhard/online_property/dqn_aux/freq8/nas_v2_delta/best/"
     },
    {"label": "DQN+Reward",
     "control": "data/output/test/gridhard/control/last/same_task/fix_rep/dqn_aux/reward/best/",
     "online_measure": "data/output/test/gridhard/online_property/dqn_aux/freq8/reward/best/"
     },
    {"label": "DQN+SF",
     "control": "data/output/test/gridhard/control/last/same_task/fix_rep/dqn_aux/successor_as/best/",
     "online_measure": "data/output/test/gridhard/online_property/dqn_aux/freq8/successor_as/best/"
     },
    {"label": "LTA eta=0.2",
     "control": "data/output/test/gridhard/control/last/same_task/fix_rep/dqn_lta/eta_study_0.2_best/",
     "online_measure": "data/output/test/gridhard/online_property/dqn_lta/eta_study_0.2_best/"
     },
    {"label": "LTA eta=0.4",
     "control": "data/output/test/gridhard/control/last/same_task/fix_rep/dqn_lta/eta_study_0.4_best/",
     "online_measure": "data/output/test/gridhard/online_property/dqn_lta/eta_study_0.4_best/"
     },
    {"label": "LTA eta=0.6",
     "control": "data/output/test/gridhard/control/last/same_task/fix_rep/dqn_lta/eta_study_0.6_best/",
     "online_measure": "data/output/test/gridhard/online_property/dqn_lta/eta_study_0.6_best/"
     },
    {"label": "LTA eta=0.8",
     "control": "data/output/test/gridhard/control/last/same_task/fix_rep/dqn_lta/eta_study_0.8_best/",
     "online_measure": "data/output/test/gridhard/online_property/dqn_lta/eta_study_0.8_best/"
     },
    {"label": "DQN+LTA+Control1g",
     "control": "data/output/test/gridhard/control/last/same_task/fix_rep/dqn_lta_aux/aux_control/best_1g/",
     "online_measure": "data/output/test/gridhard/online_property/dqn_lta_aux/aux_control/best_1g/"
     },
    {"label": "DQN+LTA+Control5g",
     "control": "data/output/test/gridhard/control/last/same_task/fix_rep/dqn_lta_aux/aux_control/best_5g/",
     "online_measure": "data/output/test/gridhard/online_property/dqn_lta_aux/aux_control/best_5g/"
     },
    {"label": "LTA+XY",
     "control": "data/output/test/gridhard/control/last/same_task/fix_rep/dqn_lta_aux/info/best/",
     "online_measure": "data/output/test/gridhard/online_property/dqn_lta_aux/info/best/"
     },
    {"label": "LTA+Decoder",
     "control": "data/output/test/gridhard/control/last/same_task/fix_rep/dqn_lta_aux/input_decoder/best/",
     "online_measure": "data/output/test/gridhard/online_property/dqn_lta_aux/input_decoder/best/"
     },
    {"label": "LTA+NAS",
     "control": "data/output/test/gridhard/control/last/same_task/fix_rep/dqn_lta_aux/nas_v2_delta/best/",
     "online_measure": "data/output/test/gridhard/online_property/dqn_lta_aux/nas_v2_delta/best/"
     },
    {"label": "LTA+Reward",
     "control": "data/output/test/gridhard/control/last/same_task/fix_rep/dqn_lta_aux/reward/best/",
     "online_measure": "data/output/test/gridhard/online_property/dqn_lta_aux/reward/best/"
     },
    {"label": "LTA+SF",
     "control": "data/output/test/gridhard/control/last/same_task/fix_rep/dqn_lta_aux/successor_as/best/",
     "online_measure": "data/output/test/gridhard/online_property/dqn_lta_aux/successor_as/best/"
     },
    {"label": "Random",
     "control": "data/output/test/gridhard/control/baseline/same_task/fix_rep/random/best/",
     "online_measure": "data/output/test/gridhard/fixrep_property/random/best",
     },
    {"label": "Input",
     "control": "data/output/test/gridhard/control/baseline/same_task/fix_rep/input/best/",
     "online_measure": "data/output/test/gridhard/fixrep_property/input/best",
     },
]

gh_similar_last = [
    {"label": "ReLU", "property": "data/output/test/",
     "control": "data/output/test/gridhard/control/last/similar_task/fix_rep/dqn/best/",
     "online_measure": "data/output/test/gridhard/online_property/dqn/best/"
     },
    {"label": "DQN+Control1g",
     "control": "data/output/test/gridhard/control/last/similar_task/fix_rep/dqn_aux/aux_control/best_1g/",
     "online_measure": "data/output/test/gridhard/online_property/dqn_aux/freq8/aux_control/best_1g/"
     },
    {"label": "DQN+Control5g",
     "control": "data/output/test/gridhard/control/last/similar_task/fix_rep/dqn_aux/aux_control/best_5g/",
     "online_measure": "data/output/test/gridhard/online_property/dqn_aux/freq128/aux_control/best_5g/"
     },
    {"label": "DQN+XY",
     "control": "data/output/test/gridhard/control/last/similar_task/fix_rep/dqn_aux/info/best/",
     "online_measure": "data/output/test/gridhard/online_property/dqn_aux/freq8/info/best/"
     },
    {"label": "DQN+Decoder",
     "control": "data/output/test/gridhard/control/last/similar_task/fix_rep/dqn_aux/input_decoder/best/",
     "online_measure": "data/output/test/gridhard/online_property/dqn_aux/freq8/info/best/"
     },
    {"label": "DQN+NAS",
     "control": "data/output/test/gridhard/control/last/similar_task/fix_rep/dqn_aux/nas_v2_delta/best/",
     "online_measure": "data/output/test/gridhard/online_property/dqn_aux/freq8/nas_v2_delta/best/"
     },
    {"label": "DQN+Reward",
     "control": "data/output/test/gridhard/control/last/similar_task/fix_rep/dqn_aux/reward/best/",
     "online_measure": "data/output/test/gridhard/online_property/dqn_aux/freq8/reward/best/"
     },
    {"label": "DQN+SF",
     "control": "data/output/test/gridhard/control/last/similar_task/fix_rep/dqn_aux/successor_as/best/",
     "online_measure": "data/output/test/gridhard/online_property/dqn_aux/freq8/successor_as/best/"
     },
    {"label": "LTA eta=0.2",
     "control": "data/output/test/gridhard/control/last/similar_task/fix_rep/dqn_lta/eta_study_0.2_best/",
     "online_measure": "data/output/test/gridhard/online_property/dqn_lta/eta_study_0.2_best/"
     },
    {"label": "LTA eta=0.4",
     "control": "data/output/test/gridhard/control/last/similar_task/fix_rep/dqn_lta/eta_study_0.4_best/",
     "online_measure": "data/output/test/gridhard/online_property/dqn_lta/eta_study_0.4_best/"
     },
    {"label": "LTA eta=0.6",
     "control": "data/output/test/gridhard/control/last/similar_task/fix_rep/dqn_lta/eta_study_0.6_best/",
     "online_measure": "data/output/test/gridhard/online_property/dqn_lta/eta_study_0.6_best/"
     },
    {"label": "LTA eta=0.8",
     "control": "data/output/test/gridhard/control/last/similar_task/fix_rep/dqn_lta/eta_study_0.8_best/",
     "online_measure": "data/output/test/gridhard/online_property/dqn_lta/eta_study_0.8_best/"
     },
    {"label": "DQN+LTA+Control1g",
     "control": "data/output/test/gridhard/control/last/similar_task/fix_rep/dqn_lta_aux/aux_control/best_1g/",
     "online_measure": "data/output/test/gridhard/online_property/dqn_lta_aux/aux_control/best_1g/"
     },
    {"label": "DQN+LTA+Control5g",
     "control": "data/output/test/gridhard/control/last/similar_task/fix_rep/dqn_lta_aux/aux_control/best_5g/",
     "online_measure": "data/output/test/gridhard/online_property/dqn_lta_aux/aux_control/best_5g/"
     },
    {"label": "LTA+XY",
     "control": "data/output/test/gridhard/control/last/similar_task/fix_rep/dqn_lta_aux/info/best/",
     "online_measure": "data/output/test/gridhard/online_property/dqn_lta_aux/info/best/"
     },
    {"label": "LTA+Decoder",
     "control": "data/output/test/gridhard/control/last/similar_task/fix_rep/dqn_lta_aux/input_decoder/best/",
     "online_measure": "data/output/test/gridhard/online_property/dqn_lta_aux/input_decoder/best/"
     },
    {"label": "LTA+NAS",
     "control": "data/output/test/gridhard/control/last/similar_task/fix_rep/dqn_lta_aux/nas_v2_delta/best/",
     "online_measure": "data/output/test/gridhard/online_property/dqn_lta_aux/nas_v2_delta/best/"
     },
    {"label": "LTA+Reward",
     "control": "data/output/test/gridhard/control/last/similar_task/fix_rep/dqn_lta_aux/reward/best/",
     "online_measure": "data/output/test/gridhard/online_property/dqn_lta_aux/reward/best/"
     },
    {"label": "LTA+SF",
     "control": "data/output/test/gridhard/control/last/similar_task/fix_rep/dqn_lta_aux/successor_as/best/",
     "online_measure": "data/output/test/gridhard/online_property/dqn_lta_aux/successor_as/best/"
     },
    {"label": "Random",
     "control": "data/output/test/gridhard/control/baseline/similar_task/fix_rep/random/best/",
     "online_measure": "data/output/test/gridhard/fixrep_property/random/best",
     },
    {"label": "Input",
     "control": "data/output/test/gridhard/control/baseline/similar_task/fix_rep/input/best/",
     "online_measure": "data/output/test/gridhard/fixrep_property/input/best",
     },
]

gh_diff_last = [
    {"label": "ReLU", "property": "data/output/test/",
     "control": "data/output/test/gridhard/control/last/different_task/fix_rep/dqn/best/",
     "online_measure": "data/output/test/gridhard/online_property/dqn/best/"
     },
    {"label": "DQN+Control1g",
     "control": "data/output/test/gridhard/control/last/different_task/fix_rep/dqn_aux/aux_control/best_1g/",
     "online_measure": "data/output/test/gridhard/online_property/dqn_aux/freq8/aux_control/best_1g/"
     },
    {"label": "DQN+Control5g",
     "control": "data/output/test/gridhard/control/last/different_task/fix_rep/dqn_aux/aux_control/best_5g/",
     "online_measure": "data/output/test/gridhard/online_property/dqn_aux/freq128/aux_control/best_5g/"
     },
    {"label": "DQN+XY",
     "control": "data/output/test/gridhard/control/last/different_task/fix_rep/dqn_aux/info/best/",
     "online_measure": "data/output/test/gridhard/online_property/dqn_aux/freq8/info/best/"
     },
    {"label": "DQN+Decoder",
     "control": "data/output/test/gridhard/control/last/different_task/fix_rep/dqn_aux/input_decoder/best/",
     "online_measure": "data/output/test/gridhard/online_property/dqn_aux/freq8/info/best/"
     },
    {"label": "DQN+NAS",
     "control": "data/output/test/gridhard/control/last/different_task/fix_rep/dqn_aux/nas_v2_delta/best/",
     "online_measure": "data/output/test/gridhard/online_property/dqn_aux/freq8/nas_v2_delta/best/"
     },
    {"label": "DQN+Reward",
     "control": "data/output/test/gridhard/control/last/different_task/fix_rep/dqn_aux/reward/best/",
     "online_measure": "data/output/test/gridhard/online_property/dqn_aux/freq8/reward/best/"
     },
    {"label": "DQN+SF",
     "control": "data/output/test/gridhard/control/last/different_task/fix_rep/dqn_aux/successor_as/best/",
     "online_measure": "data/output/test/gridhard/online_property/dqn_aux/freq8/successor_as/best/"
     },
    {"label": "LTA eta=0.2",
     "control": "data/output/test/gridhard/control/last/different_task/fix_rep/dqn_lta/eta_study_0.2_best/",
     "online_measure": "data/output/test/gridhard/online_property/dqn_lta/eta_study_0.2_best/"
     },
    {"label": "LTA eta=0.4",
     "control": "data/output/test/gridhard/control/last/different_task/fix_rep/dqn_lta/eta_study_0.4_best/",
     "online_measure": "data/output/test/gridhard/online_property/dqn_lta/eta_study_0.4_best/"
     },
    {"label": "LTA eta=0.6",
     "control": "data/output/test/gridhard/control/last/different_task/fix_rep/dqn_lta/eta_study_0.6_best/",
     "online_measure": "data/output/test/gridhard/online_property/dqn_lta/eta_study_0.6_best/"
     },
    {"label": "LTA eta=0.8",
     "control": "data/output/test/gridhard/control/last/different_task/fix_rep/dqn_lta/eta_study_0.8_best/",
     "online_measure": "data/output/test/gridhard/online_property/dqn_lta/eta_study_0.8_best/"
     },
    {"label": "DQN+LTA+Control1g",
     "control": "data/output/test/gridhard/control/last/different_task/fix_rep/dqn_lta_aux/aux_control/best_1g/",
     "online_measure": "data/output/test/gridhard/online_property/dqn_lta_aux/aux_control/best_1g/"
     },
    {"label": "DQN+LTA+Control5g",
     "control": "data/output/test/gridhard/control/last/different_task/fix_rep/dqn_lta_aux/aux_control/best_5g/",
     "online_measure": "data/output/test/gridhard/online_property/dqn_lta_aux/aux_control/best_5g/"
     },
    {"label": "LTA+XY",
     "control": "data/output/test/gridhard/control/last/different_task/fix_rep/dqn_lta_aux/info/best/",
     "online_measure": "data/output/test/gridhard/online_property/dqn_lta_aux/info/best/"
     },
    {"label": "LTA+Decoder",
     "control": "data/output/test/gridhard/control/last/different_task/fix_rep/dqn_lta_aux/input_decoder/best/",
     "online_measure": "data/output/test/gridhard/online_property/dqn_lta_aux/input_decoder/best/"
     },
    {"label": "LTA+NAS",
     "control": "data/output/test/gridhard/control/last/different_task/fix_rep/dqn_lta_aux/nas_v2_delta/best/",
     "online_measure": "data/output/test/gridhard/online_property/dqn_lta_aux/nas_v2_delta/best/"
     },
    {"label": "LTA+Reward",
     "control": "data/output/test/gridhard/control/last/different_task/fix_rep/dqn_lta_aux/reward/best/",
     "online_measure": "data/output/test/gridhard/online_property/dqn_lta_aux/reward/best/"
     },
    {"label": "LTA+SF",
     "control": "data/output/test/gridhard/control/last/different_task/fix_rep/dqn_lta_aux/successor_as/best/",
     "online_measure": "data/output/test/gridhard/online_property/dqn_lta_aux/successor_as/best/"
     },
    {"label": "Random",
     "control": "data/output/test/gridhard/control/baseline/different_task/fix_rep/random/best/",
     "online_measure": "data/output/test/gridhard/fixrep_property/random/best",
     },
    {"label": "Input",
     "control": "data/output/test/gridhard/control/baseline/different_task/fix_rep/input/best/",
     "online_measure": "data/output/test/gridhard/fixrep_property/input/best",
     },
]

gh_diff_tune_last = [
    {"label": "ReLU", "property": "data/output/test/",
     "control": "data/output/test/gridhard/control/last/different_task/fine_tune/dqn/best/",
     "online_measure": "data/output/test/gridhard/online_property/dqn/best/"
     },
    {"label": "DQN+Control1g",
     "control": "data/output/test/gridhard/control/last/different_task/fine_tune/dqn_aux/aux_control/best_1g/",
     "online_measure": "data/output/test/gridhard/online_property/dqn_aux/freq8/aux_control/best_1g/"
     },
    {"label": "DQN+Control5g",
     "control": "data/output/test/gridhard/control/last/different_task/fine_tune/dqn_aux/aux_control/best_5g/",
     "online_measure": "data/output/test/gridhard/online_property/dqn_aux/freq128/aux_control/best_5g/"
     },
    {"label": "DQN+XY",
     "control": "data/output/test/gridhard/control/last/different_task/fine_tune/dqn_aux/info/best/",
     "online_measure": "data/output/test/gridhard/online_property/dqn_aux/freq8/info/best/"
     },
    {"label": "DQN+Decoder",
     "control": "data/output/test/gridhard/control/last/different_task/fine_tune/dqn_aux/input_decoder/best/",
     "online_measure": "data/output/test/gridhard/online_property/dqn_aux/freq8/info/best/"
     },
    {"label": "DQN+NAS",
     "control": "data/output/test/gridhard/control/last/different_task/fine_tune/dqn_aux/nas_v2_delta/best/",
     "online_measure": "data/output/test/gridhard/online_property/dqn_aux/freq8/nas_v2_delta/best/"
     },
    {"label": "DQN+Reward",
     "control": "data/output/test/gridhard/control/last/different_task/fine_tune/dqn_aux/reward/best/",
     "online_measure": "data/output/test/gridhard/online_property/dqn_aux/freq8/reward/best/"
     },
    {"label": "DQN+SF",
     "control": "data/output/test/gridhard/control/last/different_task/fine_tune/dqn_aux/successor_as/best/",
     "online_measure": "data/output/test/gridhard/online_property/dqn_aux/freq8/successor_as/best/"
     },
    {"label": "LTA eta=0.2",
     "control": "data/output/test/gridhard/control/last/different_task/fine_tune/dqn_lta/eta_study_0.2_best/",
     "online_measure": "data/output/test/gridhard/online_property/dqn_lta/eta_study_0.2_best/"
     },
    {"label": "LTA eta=0.4",
     "control": "data/output/test/gridhard/control/last/different_task/fine_tune/dqn_lta/eta_study_0.4_best/",
     "online_measure": "data/output/test/gridhard/online_property/dqn_lta/eta_study_0.4_best/"
     },
    {"label": "LTA eta=0.6",
     "control": "data/output/test/gridhard/control/last/different_task/fine_tune/dqn_lta/eta_study_0.6_best/",
     "online_measure": "data/output/test/gridhard/online_property/dqn_lta/eta_study_0.6_best/"
     },
    {"label": "LTA eta=0.8",
     "control": "data/output/test/gridhard/control/last/different_task/fine_tune/dqn_lta/eta_study_0.8_best/",
     "online_measure": "data/output/test/gridhard/online_property/dqn_lta/eta_study_0.8_best/"
     },
    {"label": "DQN+LTA+Control1g",
     "control": "data/output/test/gridhard/control/last/different_task/fine_tune/dqn_lta_aux/aux_control/best_1g/",
     "online_measure": "data/output/test/gridhard/online_property/dqn_lta_aux/aux_control/best_1g/"
     },
    {"label": "DQN+LTA+Control5g",
     "control": "data/output/test/gridhard/control/last/different_task/fine_tune/dqn_lta_aux/aux_control/best_5g/",
     "online_measure": "data/output/test/gridhard/online_property/dqn_lta_aux/aux_control/best_5g/"
     },
    {"label": "LTA+XY",
     "control": "data/output/test/gridhard/control/last/different_task/fine_tune/dqn_lta_aux/info/best/",
     "online_measure": "data/output/test/gridhard/online_property/dqn_lta_aux/info/best/"
     },
    {"label": "LTA+Decoder",
     "control": "data/output/test/gridhard/control/last/different_task/fine_tune/dqn_lta_aux/input_decoder/best/",
     "online_measure": "data/output/test/gridhard/online_property/dqn_lta_aux/input_decoder/best/"
     },
    {"label": "LTA+NAS",
     "control": "data/output/test/gridhard/control/last/different_task/fine_tune/dqn_lta_aux/nas_v2_delta/best/",
     "online_measure": "data/output/test/gridhard/online_property/dqn_lta_aux/nas_v2_delta/best/"
     },
    {"label": "LTA+Reward",
     "control": "data/output/test/gridhard/control/last/different_task/fine_tune/dqn_lta_aux/reward/best/",
     "online_measure": "data/output/test/gridhard/online_property/dqn_lta_aux/reward/best/"
     },
    {"label": "LTA+SF",
     "control": "data/output/test/gridhard/control/last/different_task/fine_tune/dqn_lta_aux/successor_as/best/",
     "online_measure": "data/output/test/gridhard/online_property/dqn_lta_aux/successor_as/best/"
     },
    {"label": "Random",
     "control": "data/output/test/gridhard/control/baseline/different_task/fine_tune/random/best/",
     "online_measure": "data/output/test/gridhard/fixrep_property/random/best",
     },
]

# ------------------------- Picky Eater Environment --------------------

dqn_lta_learn_sweep = [
    {"label": "DQN+LTA",
     "control": "data/output/test/picky_eater/online_property/dqn_lta/sweep",
     },
    ]

dqn_lta_1_learn_sweep = [
    {"label": "DQN+LTA+1",
     "control": "data/output/test/picky_eater/online_property/dqn_lta_1/sweep",
     },
    ]


dqn_learn_sweep = [
    {"label": "ReLU",
     "control": "data/output/test/picky_eater/online_property/dqn/sweep",
     },
    ]
