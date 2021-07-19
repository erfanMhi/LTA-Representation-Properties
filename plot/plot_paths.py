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
    "ReLU+Color": c_default[6],
    "ReLU+Control+XY+Color": c_default[8],

    "ReLU+Decoder": c_default[7],
    "ReLU+NAS": c_default[8],
    "ReLU+Reward": c_default[9],
    "ReLU+SF": c_default[10],

    "FTA": c_default[0],
    "FTA(no target)": c_default[0],

    "FTA eta=2": c_default[1],
    
    "FTA eta=0.2": c_default[1],
    "FTA eta=0.4": c_default[0],
    "FTA eta=0.6": c_default[11],
    "FTA eta=0.8": c_default[12],

    "FTA+Control": c_default[2],
    "FTA+Control1g": c_default[2],
    "FTA+Control5g": c_default[3],
    "FTA+XY": c_default[4],
    "FTA+XY+color": c_default[5],
    "FTA+XY+count": c_default[6],
    "FTA+Color": c_default[6],
    "FTA+Control+XY+Color": c_default[8],

    "FTA+Decoder": c_default[7],
    "FTA+NAS": c_default[8],
    "FTA+Reward": c_default[9],
    "FTA+SF": c_default[10],

    "Random": c_default[12],
    "Input": c_default[11],
    "Scratch": "black",
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
    "ReLU+Color": s_default[1],
    "ReLU+Control+XY+Color": s_default[1],

    "FTA": s_default[0],
    "FTA(no target)": s_default[0],

    "FTA eta=2": s_default[0],
    
    "FTA eta=0.2": s_default[0],
    "FTA eta=0.4": s_default[0],
    "FTA eta=0.6": s_default[0],
    "FTA eta=0.8": s_default[0],

    "FTA+Control": s_default[0],
    "FTA+Control1g": s_default[0],
    "FTA+Control5g": s_default[0],
    "FTA+XY": s_default[0],
    "FTA+XY+color": s_default[0],
    "FTA+XY+count": s_default[0],
    "FTA+Decoder": s_default[0],
    "FTA+NAS": s_default[0],
    "FTA+Reward": s_default[0],
    "FTA+SF": s_default[0],
    "FTA+Color": s_default[0],
    "FTA+Control+XY+Color": s_default[0],

    "Random": s_default[0],
    "Input": s_default[0],
    "Scratch": s_default[0],
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

    "FTA": m_default[0],
    "FTA(no target)": m_default[0],

    "FTA eta=2": m_default[0],
    
    "FTA eta=0.2": m_default[0],
    "FTA eta=0.4": m_default[0],
    "FTA eta=0.6": m_default[0],
    "FTA eta=0.8": m_default[0],

    "FTA+Control": m_default[0],
    "FTA+Control1g": m_default[0],
    "FTA+Control5g": m_default[0],
    "FTA+XY": m_default[0],
    "FTA+XY+color": m_default[0],
    "FTA+XY+count": m_default[0],
    "FTA+Decoder": m_default[0],
    "FTA+NAS": m_default[0],
    "FTA+Reward": m_default[0],
    "FTA+SF": m_default[0],

    "Random": m_default[0],
    "Input": m_default[0],
}

mc_learn_sweep = [
    {"label": "ReLU",
     "control": "data/output/test/mountaincar/representations/dqn/sweep/",
     },
    {"label": "FTA",
     "control": "data/output/test/mountaincar/representations/dqn_lta/sweep/",
     },
    {"label": "FTA(no target)",
     "control": "data/output/test/mountaincar/representations/dqn_lta/no_target/",
     },
]

gh_online_sweep = [
    {"label": "ReLU",
     "control": "data/output/test_v7/gridhard/linear_vf/representation/dqn/sweep/",
     },
    # {"label": "ReLU+Control1g",
    #  "control": "data/output/test_v7/gridhard/linear_vf/representation/dqn_aux/aux_control/sweep_1g/",
    #  },
    # {"label": "ReLU+Control5g",
    #  "control": "data/output/test_v7/gridhard/linear_vf/representation/dqn_aux/aux_control/sweep_5g/",
    #  },
    {"label": "ReLU+Control1g (0.9)",
     "control": "data/output/test_v7/gridhard/linear_vf/representation/dqn_aux/aux_control/sweep_1g_gamma0.9/",
     },
    {"label": "ReLU+Control5g (0.9)",
     "control": "data/output/test_v7/gridhard/linear_vf/representation/dqn_aux/aux_control/sweep_5g_gamma0.9/",
     },
    {"label": "ReLU+XY",
     "control": "data/output/test_v7/gridhard/linear_vf/representation/dqn_aux/info/sweep/",
     },
    {"label": "ReLU+Decoder",
     "control": "data/output/test_v7/gridhard/linear_vf/representation/dqn_aux/input_decoder/sweep/",
     },
    {"label": "ReLU+NAS",
     "control": "data/output/test_v7/gridhard/linear_vf/representation/dqn_aux/nas_v2_delta/sweep/",
     },
    {"label": "ReLU+Reward",
     "control": "data/output/test_v7/gridhard/linear_vf/representation/dqn_aux/reward/sweep/",
     },
    {"label": "ReLU+SF",
     "control": "data/output/test_v7/gridhard/linear_vf/representation/dqn_aux/successor_as/sweep/",
     },
    {"label": "FTA eta=0.2",
     "control": "data/output/test_v7/gridhard/linear_vf/representation/dqn_lta/eta_study_0.2_sweep/",
     },
    {"label": "FTA eta=0.4",
     "control": "data/output/test_v7/gridhard/linear_vf/representation/dqn_lta/eta_study_0.4_sweep/",
     },
    {"label": "FTA eta=0.6",
     "control": "data/output/test_v7/gridhard/linear_vf/representation/dqn_lta/eta_study_0.6_sweep/",
     },
    {"label": "FTA eta=0.8",
     "control": "data/output/test_v7/gridhard/linear_vf/representation/dqn_lta/eta_study_0.8_sweep/",
     },
    # {"label": "FTA+Control1g",
    #  "control": "data/output/test_v7/gridhard/linear_vf/representation/dqn_lta_aux/aux_control/sweep_1g/",
    #  },
    # {"label": "FTA+Control5g",
    #  "control": "data/output/test_v7/gridhard/linear_vf/representation/dqn_lta_aux/aux_control/sweep_5g/",
    #  },
    {"label": "FTA+Control1g (0.9)",
     "control": "data/output/test_v7/gridhard/linear_vf/representation/dqn_lta_aux/aux_control/sweep_1g_gamma0.9/",
     },
    {"label": "FTA+Control5g (0.9)",
     "control": "data/output/test_v7/gridhard/linear_vf/representation/dqn_lta_aux/aux_control/sweep_5g_gamma0.9/",
     },
    {"label": "FTA+XY",
     "control": "data/output/test_v7/gridhard/linear_vf/representation/dqn_lta_aux/info/sweep/",
     },
    {"label": "FTA+Decoder",
     "control": "data/output/test_v7/gridhard/linear_vf/representation/dqn_lta_aux/input_decoder/sweep/",
     },
    {"label": "FTA+NAS",
     "control": "data/output/test_v7/gridhard/linear_vf/representation/dqn_lta_aux/nas_v2_delta/sweep/",
     },
    {"label": "FTA+Reward",
     "control": "data/output/test_v7/gridhard/linear_vf/representation/dqn_lta_aux/reward/sweep/",
     },
    {"label": "FTA+SF",
     "control": "data/output/test_v7/gridhard/linear_vf/representation/dqn_lta_aux/successor_as/sweep/",
     }
]

gh_same_early_sweep = [
    {"label": "ReLU",
     "control": "data/output/result/gridhard/linear_vf/control/early_stop/same_task/fix_rep/dqn/sweep/",
     },
    {"label": "ReLU+Control1g (0.9)",
     "control": "data/output/result/gridhard/linear_vf/control/early_stop/same_task/fix_rep/dqn_aux/aux_control/sweep_1g_gamma0.9/",
     },
    {"label": "ReLU+Control5g (0.9)",
     "control": "data/output/result/gridhard/linear_vf/control/early_stop/same_task/fix_rep/dqn_aux/aux_control/sweep_5g_gamma0.9/",
     },
    {"label": "ReLU+XY",
     "control": "data/output/result/gridhard/linear_vf/control/early_stop/same_task/fix_rep/dqn_aux/info/sweep/",
     },
    {"label": "ReLU+Decoder",
     "control": "data/output/result/gridhard/linear_vf/control/early_stop/same_task/fix_rep/dqn_aux/input_decoder/sweep/",
     },
    {"label": "ReLU+NAS",
     "control": "data/output/result/gridhard/linear_vf/control/early_stop/same_task/fix_rep/dqn_aux/nas_v2_delta/sweep/",
     },
    {"label": "ReLU+Reward",
     "control": "data/output/result/gridhard/linear_vf/control/early_stop/same_task/fix_rep/dqn_aux/reward/sweep/",
     },
    {"label": "ReLU+SF",
     "control": "data/output/result/gridhard/linear_vf/control/early_stop/same_task/fix_rep/dqn_aux/successor_as/sweep/",
     },
    {"label": "FTA eta=0.2",
     "control": "data/output/result/gridhard/linear_vf/control/early_stop/same_task/fix_rep/dqn_lta/eta_study_0.2_sweep/",
     },
    {"label": "FTA eta=0.4",
     "control": "data/output/result/gridhard/linear_vf/control/early_stop/same_task/fix_rep/dqn_lta/eta_study_0.4_sweep/",
     },
    {"label": "FTA eta=0.6",
     "control": "data/output/result/gridhard/linear_vf/control/early_stop/same_task/fix_rep/dqn_lta/eta_study_0.6_sweep/",
     },
    {"label": "FTA eta=0.8",
     "control": "data/output/result/gridhard/linear_vf/control/early_stop/same_task/fix_rep/dqn_lta/eta_study_0.8_sweep/",
     },
    {"label": "FTA+Control1g (0.9)",
     "control": "data/output/result/gridhard/linear_vf/control/early_stop/same_task/fix_rep/dqn_lta_aux/aux_control/sweep_1g_gamma0.9/",
     },
    {"label": "FTA+Control5g (0.9)",
     "control": "data/output/result/gridhard/linear_vf/control/early_stop/same_task/fix_rep/dqn_lta_aux/aux_control/sweep_5g_gamma0.9/",
     },
    {"label": "FTA+XY",
     "control": "data/output/result/gridhard/linear_vf/control/early_stop/same_task/fix_rep/dqn_lta_aux/info/sweep/",
     },
    {"label": "FTA+Decoder",
     "control": "data/output/result/gridhard/linear_vf/control/early_stop/same_task/fix_rep/dqn_lta_aux/input_decoder/sweep/",
     },
    {"label": "FTA+NAS",
     "control": "data/output/result/gridhard/linear_vf/control/early_stop/same_task/fix_rep/dqn_lta_aux/nas_v2_delta/sweep/",
     },
    {"label": "FTA+Reward",
     "control": "data/output/result/gridhard/linear_vf/control/early_stop/same_task/fix_rep/dqn_lta_aux/reward/sweep/",
     },
    {"label": "FTA+SF",
     "control": "data/output/result/gridhard/linear_vf/control/early_stop/same_task/fix_rep/dqn_lta_aux/successor_as/sweep/",
     },
    {"label": "Random",
     "control": "data/output/result/gridhard/linear_vf/control/early_stop/same_task/fix_rep/random/sweep/",
     },
    {"label": "Input",
     "control": "data/output/result/gridhard/linear_vf/control/early_stop/same_task/fix_rep/input/sweep/",
     },
]

gh_similar_early_sweep = [
    {"label": "ReLU",
     "control": "data/output/result/gridhard/linear_vf/control/early_stop/similar_task/fix_rep/dqn/sweep/",
     },
    {"label": "ReLU+Control1g (0.9)",
     "control": "data/output/result/gridhard/linear_vf/control/early_stop/similar_task/fix_rep/dqn_aux/aux_control/sweep_1g_gamma0.9/",
     },
    {"label": "ReLU+Control5g (0.9)",
     "control": "data/output/result/gridhard/linear_vf/control/early_stop/similar_task/fix_rep/dqn_aux/aux_control/sweep_5g_gamma0.9/",
     },
    {"label": "ReLU+XY",
     "control": "data/output/result/gridhard/linear_vf/control/early_stop/similar_task/fix_rep/dqn_aux/info/sweep/",
     },
    {"label": "ReLU+Decoder",
     "control": "data/output/result/gridhard/linear_vf/control/early_stop/similar_task/fix_rep/dqn_aux/input_decoder/sweep/",
     },
    {"label": "ReLU+NAS",
     "control": "data/output/result/gridhard/linear_vf/control/early_stop/similar_task/fix_rep/dqn_aux/nas_v2_delta/sweep/",
     },
    {"label": "ReLU+Reward",
     "control": "data/output/result/gridhard/linear_vf/control/early_stop/similar_task/fix_rep/dqn_aux/reward/sweep/",
     },
    {"label": "ReLU+SF",
     "control": "data/output/result/gridhard/linear_vf/control/early_stop/similar_task/fix_rep/dqn_aux/successor_as/sweep/",
     },
    {"label": "FTA eta=0.2",
     "control": "data/output/result/gridhard/linear_vf/control/early_stop/similar_task/fix_rep/dqn_lta/eta_study_0.2_sweep/",
     },
    {"label": "FTA eta=0.4",
     "control": "data/output/result/gridhard/linear_vf/control/early_stop/similar_task/fix_rep/dqn_lta/eta_study_0.4_sweep/",
     },
    {"label": "FTA eta=0.6",
     "control": "data/output/result/gridhard/linear_vf/control/early_stop/similar_task/fix_rep/dqn_lta/eta_study_0.6_sweep/",
     },
    {"label": "FTA eta=0.8",
     "control": "data/output/result/gridhard/linear_vf/control/early_stop/similar_task/fix_rep/dqn_lta/eta_study_0.8_sweep/",
     },
    {"label": "FTA+Control1g (0.9)",
     "control": "data/output/result/gridhard/linear_vf/control/early_stop/similar_task/fix_rep/dqn_lta_aux/aux_control/sweep_1g_gamma0.9/",
     },
    {"label": "FTA+Control5g (0.9)",
     "control": "data/output/result/gridhard/linear_vf/control/early_stop/similar_task/fix_rep/dqn_lta_aux/aux_control/sweep_5g_gamma0.9/",
     },
    {"label": "FTA+XY",
     "control": "data/output/result/gridhard/linear_vf/control/early_stop/similar_task/fix_rep/dqn_lta_aux/info/sweep/",
     },
    {"label": "FTA+Decoder",
     "control": "data/output/result/gridhard/linear_vf/control/early_stop/similar_task/fix_rep/dqn_lta_aux/input_decoder/sweep/",
     },
    {"label": "FTA+NAS",
     "control": "data/output/result/gridhard/linear_vf/control/early_stop/similar_task/fix_rep/dqn_lta_aux/nas_v2_delta/sweep/",
     },
    {"label": "FTA+Reward",
     "control": "data/output/result/gridhard/linear_vf/control/early_stop/similar_task/fix_rep/dqn_lta_aux/reward/sweep/",
     },
    {"label": "FTA+SF",
     "control": "data/output/result/gridhard/linear_vf/control/early_stop/similar_task/fix_rep/dqn_lta_aux/successor_as/sweep/",
     },
    {"label": "Random",
     "control": "data/output/result/gridhard/linear_vf/control/early_stop/similar_task/fix_rep/random/sweep/",
     },
    {"label": "Input",
     "control": "data/output/result/gridhard/linear_vf/control/early_stop/similar_task/fix_rep/input/sweep/",
     },
]

gh_diff_early_sweep = [
    {"label": "ReLU",
     "control": "data/output/test_v7/gridhard/linear_vf/control/early_stop/different_task/fix_rep/dqn/sweep/",
     },
    {"label": "ReLU+Control1g",
     "control": "data/output/test_v7/gridhard/linear_vf/control/early_stop/different_task/fix_rep/dqn_aux/aux_control/sweep_1g/",
     },
    {"label": "ReLU+Control5g",
     "control": "data/output/test_v7/gridhard/linear_vf/control/early_stop/different_task/fix_rep/dqn_aux/aux_control/sweep_5g/",
     },
    {"label": "ReLU+Control1g (0.9)",
     "control": "data/output/test_v7/gridhard/linear_vf/control/early_stop/different_task/fix_rep/dqn_aux/aux_control/sweep_1g_gamma0.9/",
     },
    {"label": "ReLU+Control5g (0.9)",
     "control": "data/output/test_v7/gridhard/linear_vf/control/early_stop/different_task/fix_rep/dqn_aux/aux_control/sweep_5g_gamma0.9/",
     },
    {"label": "ReLU+XY",
     "control": "data/output/test_v7/gridhard/linear_vf/control/early_stop/different_task/fix_rep/dqn_aux/info/sweep/",
     },
    {"label": "ReLU+Decoder",
     "control": "data/output/test_v7/gridhard/linear_vf/control/early_stop/different_task/fix_rep/dqn_aux/input_decoder/sweep/",
     },
    {"label": "ReLU+NAS",
     "control": "data/output/test_v7/gridhard/linear_vf/control/early_stop/different_task/fix_rep/dqn_aux/nas_v2_delta/sweep/",
     },
    {"label": "ReLU+Reward",
     "control": "data/output/test_v7/gridhard/linear_vf/control/early_stop/different_task/fix_rep/dqn_aux/reward/sweep/",
     },
    {"label": "ReLU+SF",
     "control": "data/output/test_v7/gridhard/linear_vf/control/early_stop/different_task/fix_rep/dqn_aux/successor_as/sweep/",
     },
    {"label": "FTA eta=0.2",
     "control": "data/output/test_v7/gridhard/linear_vf/control/early_stop/different_task/fix_rep/dqn_lta/eta_study_0.2_sweep/",
     },
    {"label": "FTA eta=0.4",
     "control": "data/output/test_v7/gridhard/linear_vf/control/early_stop/different_task/fix_rep/dqn_lta/eta_study_0.4_sweep/",
     },
    {"label": "FTA eta=0.6",
     "control": "data/output/test_v7/gridhard/linear_vf/control/early_stop/different_task/fix_rep/dqn_lta/eta_study_0.6_sweep/",
     },
    {"label": "FTA eta=0.8",
     "control": "data/output/test_v7/gridhard/linear_vf/control/early_stop/different_task/fix_rep/dqn_lta/eta_study_0.8_sweep/",
     },
    {"label": "FTA+Control1g",
     "control": "data/output/test_v7/gridhard/linear_vf/control/early_stop/different_task/fix_rep/dqn_lta_aux/aux_control/sweep_1g/",
     },
    {"label": "FTA+Control5g",
     "control": "data/output/test_v7/gridhard/linear_vf/control/early_stop/different_task/fix_rep/dqn_lta_aux/aux_control/sweep_5g/",
     },
    {"label": "FTA+Control1g (0.9)",
     "control": "data/output/test_v7/gridhard/linear_vf/control/early_stop/different_task/fix_rep/dqn_lta_aux/aux_control/sweep_1g_gamma0.9/",
     },
    {"label": "FTA+Control5g (0.9)",
     "control": "data/output/test_v7/gridhard/linear_vf/control/early_stop/different_task/fix_rep/dqn_lta_aux/aux_control/sweep_5g_gamma0.9/",
     },
    {"label": "FTA+XY",
     "control": "data/output/test_v7/gridhard/linear_vf/control/early_stop/different_task/fix_rep/dqn_lta_aux/info/sweep/",
     },
    {"label": "FTA+Decoder",
     "control": "data/output/test_v7/gridhard/linear_vf/control/early_stop/different_task/fix_rep/dqn_lta_aux/input_decoder/sweep/",
     },
    {"label": "FTA+NAS",
     "control": "data/output/test_v7/gridhard/linear_vf/control/early_stop/different_task/fix_rep/dqn_lta_aux/nas_v2_delta/sweep/",
     },
    {"label": "FTA+Reward",
     "control": "data/output/test_v7/gridhard/linear_vf/control/early_stop/different_task/fix_rep/dqn_lta_aux/reward/sweep/",
     },
    {"label": "FTA+SF",
     "control": "data/output/test_v7/gridhard/linear_vf/control/early_stop/different_task/fix_rep/dqn_lta_aux/successor_as/sweep/",
     },
    {"label": "Random",
     "control": "data/output/result/gridhard/linear_vf/control/early_stop/different_task/fix_rep/random/sweep/",
     },
    {"label": "Input",
     "control": "data/output/result/gridhard/linear_vf/control/early_stop/different_task/fix_rep/input/sweep/",
     },
    {"label": "Scratch",
     "control": "data/output/result/gridhard/linear_vf/control/early_stop/different_task/scratch/dqn/sweep/",
     },

]

gh_diff_tune_early_sweep = [
    {"label": "ReLU",
     "control": "data/output/test/gridhard/control/different_task/fine_tune/dqn/sweep/",
     },
    {"label": "ReLU+Control1g",
     "control": "data/output/test/gridhard/control/different_task/fine_tune/dqn_aux/aux_control/sweep_1g/",
     },
    {"label": "ReLU+Control5g",
     "control": "data/output/test/gridhard/control/early_stop/different_task/fine_tune/dqn_aux/aux_control/sweep_5g/",
     },
    {"label": "ReLU+XY",
     "control": "data/output/test/gridhard/control/different_task/fine_tune/dqn_aux/info/sweep/",
     },
    {"label": "ReLU+Decoder",
     "control": "data/output/test/gridhard/control/different_task/fine_tune/dqn_aux/input_decoder/sweep/",
     },
    {"label": "ReLU+NAS",
     "control": "data/output/test/gridhard/control/different_task/fine_tune/dqn_aux/nas_v2_delta/sweep/",
     },
    {"label": "ReLU+Reward",
     "control": "data/output/test/gridhard/control/different_task/fine_tune/dqn_aux/reward/sweep/",
     },
    {"label": "ReLU+SF",
     "control": "data/output/test/gridhard/control/different_task/fine_tune/dqn_aux/successor_as/sweep/",
     },
    {"label": "FTA eta=0.2",
     "control": "data/output/test/gridhard/control/different_task/fine_tune/dqn_lta/eta_study_0.2_sweep/",
     },
    {"label": "FTA eta=0.4",
     "control": "data/output/test/gridhard/control/different_task/fine_tune/dqn_lta/eta_study_0.4_sweep/",
     },
    {"label": "FTA eta=0.6",
     "control": "data/output/test/gridhard/control/different_task/fine_tune/dqn_lta/eta_study_0.6_sweep/",
     },
    {"label": "FTA eta=0.8",
     "control": "data/output/test/gridhard/control/different_task/fine_tune/dqn_lta/eta_study_0.8_sweep/",
     },
    {"label": "FTA+Control1g",
     "control": "data/output/test/gridhard/control/different_task/fine_tune/dqn_lta_aux/aux_control/sweep_1g/",
     },
    {"label": "FTA+Control5g",
     "control": "data/output/test/gridhard/control/different_task/fine_tune/dqn_lta_aux/aux_control/sweep_5g/",
     },
    {"label": "FTA+XY",
     "control": "data/output/test/gridhard/control/different_task/fine_tune/dqn_lta_aux/info/sweep/",
     },
    {"label": "FTA+Decoder",
     "control": "data/output/test/gridhard/control/different_task/fine_tune/dqn_lta_aux/input_decoder/sweep/",
     },
    {"label": "FTA+NAS",
     "control": "data/output/test/gridhard/control/different_task/fine_tune/dqn_lta_aux/nas_v2_delta/sweep/",
     },
    {"label": "FTA+Reward",
     "control": "data/output/test/gridhard/control/different_task/fine_tune/dqn_lta_aux/reward/sweep/",
     },
    {"label": "FTA+SF",
     "control": "data/output/test/gridhard/control/different_task/fine_tune/dqn_lta_aux/successor_as/sweep/",
     },
    {"label": "Random",
     "control": "data/output/test/gridhard/control/different_task/fine_tune/random/sweep/",
     },
]

gh_online = [
    {"label": "ReLU",
     "control": "data/output/result/gridhard/linear_vf/online_property/dqn/best/",
     },
    {"label": "ReLU+Control1g",
     "control": "data/output/result/gridhard/linear_vf/online_property/dqn_aux/aux_control/best_1g_gamma0.9/",
     },
    {"label": "ReLU+Control5g",
     "control": "data/output/result/gridhard/linear_vf/online_property/dqn_aux/aux_control/best_5g_gamma0.9/",
     },
    {"label": "ReLU+XY",
     "control": "data/output/result/gridhard/linear_vf/online_property/dqn_aux/info/best/",
     },
    {"label": "ReLU+Decoder",
     "control": "data/output/result/gridhard/linear_vf/online_property/dqn_aux/input_decoder/best/",
     },
    {"label": "ReLU+NAS",
     "control": "data/output/result/gridhard/linear_vf/online_property/dqn_aux/nas_v2_delta/best/",
     },
    {"label": "ReLU+Reward",
     "control": "data/output/result/gridhard/linear_vf/online_property/dqn_aux/reward/best/",
     },
    {"label": "ReLU+SF",
     "control": "data/output/result/gridhard/linear_vf/online_property/dqn_aux/successor_as/best/",
     },
    {"label": "FTA eta=0.2",
     "control": "data/output/result/gridhard/linear_vf/online_property/dqn_lta/eta_study_0.2_best/",
     },
    {"label": "FTA eta=0.4",
     "control": "data/output/result/gridhard/linear_vf/online_property/dqn_lta/eta_study_0.4_best/",
     },
    {"label": "FTA eta=0.6",
     "control": "data/output/result/gridhard/linear_vf/online_property/dqn_lta/eta_study_0.6_best/",
     },
    {"label": "FTA eta=0.8",
     "control": "data/output/result/gridhard/linear_vf/online_property/dqn_lta/eta_study_0.8_best/",
     },
    {"label": "FTA+Control1g",
     "control": "data/output/result/gridhard/linear_vf/online_property/dqn_lta_aux/aux_control/best_1g_gamma0.9/",
     },
    {"label": "FTA+Control5g",
     "control": "data/output/result/gridhard/linear_vf/online_property/dqn_lta_aux/aux_control/best_5g_gamma0.9/",
     },
    {"label": "FTA+XY",
     "control": "data/output/result/gridhard/linear_vf/online_property/dqn_lta_aux/info/best/",
     },
    {"label": "FTA+Decoder",
     "control": "data/output/result/gridhard/linear_vf/online_property/dqn_lta_aux/input_decoder/best/",
     },
    {"label": "FTA+NAS",
     "control": "data/output/result/gridhard/linear_vf/online_property/dqn_lta_aux/nas_v2_delta/best/",
     },
    {"label": "FTA+Reward",
     "control": "data/output/result/gridhard/linear_vf/online_property/dqn_lta_aux/reward/best/",
     },
    {"label": "FTA+SF",
     "control": "data/output/result/gridhard/linear_vf/online_property/dqn_lta_aux/successor_as/best/",
     }
]

gh_same_early = [
    {"label": "ReLU", "property": "data/output/result/",
     "control": "data/output/result/gridhard/linear_vf/control/early_stop/same_task/fix_rep/dqn/best/",
     "online_measure": "data/output/result/gridhard/linear_vf/online_property/dqn/best/"
     },
    {"label": "ReLU+Control1g",
     "control": "data/output/result/gridhard/linear_vf/control/early_stop/same_task/fix_rep/dqn_aux/aux_control/best_1g_gamma0.9/",
     "online_measure": "data/output/result/gridhard/linear_vf/online_property/dqn_aux/aux_control/best_1g_gamma0.9/"
     },
    {"label": "ReLU+Control5g",
     "control": "data/output/result/gridhard/linear_vf/control/early_stop/same_task/fix_rep/dqn_aux/aux_control/best_5g_gamma0.9/",
     "online_measure": "data/output/result/gridhard/linear_vf/online_property/dqn_aux/aux_control/best_5g_gamma0.9/"
     },
    {"label": "ReLU+XY",
     "control": "data/output/result/gridhard/linear_vf/control/early_stop/same_task/fix_rep/dqn_aux/info/best/",
     "online_measure": "data/output/result/gridhard/linear_vf/online_property/dqn_aux/info/best/"
     },
    {"label": "ReLU+Decoder",
     "control": "data/output/result/gridhard/linear_vf/control/early_stop/same_task/fix_rep/dqn_aux/input_decoder/best/",
     "online_measure": "data/output/result/gridhard/linear_vf/online_property/dqn_aux/info/best/"
     },
    {"label": "ReLU+NAS",
     "control": "data/output/result/gridhard/linear_vf/control/early_stop/same_task/fix_rep/dqn_aux/nas_v2_delta/best/",
     "online_measure": "data/output/result/gridhard/linear_vf/online_property/dqn_aux/nas_v2_delta/best/"
     },
    {"label": "ReLU+Reward",
     "control": "data/output/result/gridhard/linear_vf/control/early_stop/same_task/fix_rep/dqn_aux/reward/best/",
     "online_measure": "data/output/result/gridhard/linear_vf/online_property/dqn_aux/reward/best/"
     },
    {"label": "ReLU+SF",
     "control": "data/output/result/gridhard/linear_vf/control/early_stop/same_task/fix_rep/dqn_aux/successor_as/best/",
     "online_measure": "data/output/result/gridhard/linear_vf/online_property/dqn_aux/successor_as/best/"
     },
    {"label": "FTA eta=0.2",
     "control": "data/output/result/gridhard/linear_vf/control/early_stop/same_task/fix_rep/dqn_lta/eta_study_0.2_best/",
     "online_measure": "data/output/result/gridhard/linear_vf/online_property/dqn_lta/eta_study_0.2_best/"
     },
    {"label": "FTA eta=0.4",
     "control": "data/output/result/gridhard/linear_vf/control/early_stop/same_task/fix_rep/dqn_lta/eta_study_0.4_best/",
     "online_measure": "data/output/result/gridhard/linear_vf/online_property/dqn_lta/eta_study_0.4_best/"
     },
    {"label": "FTA eta=0.6",
     "control": "data/output/result/gridhard/linear_vf/control/early_stop/same_task/fix_rep/dqn_lta/eta_study_0.6_best/",
     "online_measure": "data/output/result/gridhard/linear_vf/online_property/dqn_lta/eta_study_0.6_best/"
     },
    {"label": "FTA eta=0.8",
     "control": "data/output/result/gridhard/linear_vf/control/early_stop/same_task/fix_rep/dqn_lta/eta_study_0.8_best/",
     "online_measure": "data/output/result/gridhard/linear_vf/online_property/dqn_lta/eta_study_0.8_best/"
     },
    {"label": "FTA+Control1g",
     "control": "data/output/result/gridhard/linear_vf/control/early_stop/same_task/fix_rep/dqn_lta_aux/aux_control/best_1g_gamma0.9/",
     "online_measure": "data/output/result/gridhard/linear_vf/online_property/dqn_lta_aux/aux_control/best_1g_gamma0.9/"
     },
    {"label": "FTA+Control5g",
     "control": "data/output/result/gridhard/linear_vf/control/early_stop/same_task/fix_rep/dqn_lta_aux/aux_control/best_5g_gamma0.9/",
     "online_measure": "data/output/result/gridhard/linear_vf/online_property/dqn_lta_aux/aux_control/best_5g_gamma0.9/"
     },
    {"label": "FTA+XY",
     "control": "data/output/result/gridhard/linear_vf/control/early_stop/same_task/fix_rep/dqn_lta_aux/info/best/",
     "online_measure": "data/output/result/gridhard/linear_vf/online_property/dqn_lta_aux/info/best/"
     },
    {"label": "FTA+Decoder",
     "control": "data/output/result/gridhard/linear_vf/control/early_stop/same_task/fix_rep/dqn_lta_aux/input_decoder/best/",
     "online_measure": "data/output/result/gridhard/linear_vf/online_property/dqn_lta_aux/input_decoder/best/"
     },
    {"label": "FTA+NAS",
     "control": "data/output/result/gridhard/linear_vf/control/early_stop/same_task/fix_rep/dqn_lta_aux/nas_v2_delta/best/",
     "online_measure": "data/output/result/gridhard/linear_vf/online_property/dqn_lta_aux/nas_v2_delta/best/"
     },
    {"label": "FTA+Reward",
     "control": "data/output/result/gridhard/linear_vf/control/early_stop/same_task/fix_rep/dqn_lta_aux/reward/best/",
     "online_measure": "data/output/result/gridhard/linear_vf/online_property/dqn_lta_aux/reward/best/"
     },
    {"label": "FTA+SF",
     "control": "data/output/result/gridhard/linear_vf/control/early_stop/same_task/fix_rep/dqn_lta_aux/successor_as/best/",
     "online_measure": "data/output/result/gridhard/linear_vf/online_property/dqn_lta_aux/successor_as/best/"
     },
    {"label": "Random",
     "control": "data/output/result/gridhard/linear_vf/control/early_stop/same_task/fix_rep/random/best/",
     "online_measure": "data/output/result/gridhard/linear_vf/fixrep_property/random/best",
     },
    {"label": "Input",
     "control": "data/output/result/gridhard/linear_vf/control/early_stop/same_task/fix_rep/input/best/",
     "online_measure": "data/output/result/gridhard/linear_vf/fixrep_property/input/best",
     },
]

gh_similar_early = [
    {"label": "ReLU", "property": "data/output/result/",
     "control": "data/output/result/gridhard/linear_vf/control/early_stop/similar_task/fix_rep/dqn/best/",
     "online_measure": "data/output/result/gridhard/linear_vf/online_property/dqn/best/"
     },
    {"label": "ReLU+Control1g",
     "control": "data/output/result/gridhard/linear_vf/control/early_stop/similar_task/fix_rep/dqn_aux/aux_control/best_1g_gamma0.9/",
     "online_measure": "data/output/result/gridhard/linear_vf/online_property/dqn_aux/aux_control/best_1g_gamma0.9/"
     },
    {"label": "ReLU+Control5g",
     "control": "data/output/result/gridhard/linear_vf/control/early_stop/similar_task/fix_rep/dqn_aux/aux_control/best_5g_gamma0.9/",
     "online_measure": "data/output/result/gridhard/linear_vf/online_property/dqn_aux/aux_control/best_5g_gamma0.9/"
     },
    {"label": "ReLU+XY",
     "control": "data/output/result/gridhard/linear_vf/control/early_stop/similar_task/fix_rep/dqn_aux/info/best/",
     "online_measure": "data/output/result/gridhard/linear_vf/online_property/dqn_aux/info/best/"
     },
    {"label": "ReLU+Decoder",
     "control": "data/output/result/gridhard/linear_vf/control/early_stop/similar_task/fix_rep/dqn_aux/input_decoder/best/",
     "online_measure": "data/output/result/gridhard/linear_vf/online_property/dqn_aux/info/best/"
     },
    {"label": "ReLU+NAS",
     "control": "data/output/result/gridhard/linear_vf/control/early_stop/similar_task/fix_rep/dqn_aux/nas_v2_delta/best/",
     "online_measure": "data/output/result/gridhard/linear_vf/online_property/dqn_aux/nas_v2_delta/best/"
     },
    {"label": "ReLU+Reward",
     "control": "data/output/result/gridhard/linear_vf/control/early_stop/similar_task/fix_rep/dqn_aux/reward/best/",
     "online_measure": "data/output/result/gridhard/linear_vf/online_property/dqn_aux/reward/best/"
     },
    {"label": "ReLU+SF",
     "control": "data/output/result/gridhard/linear_vf/control/early_stop/similar_task/fix_rep/dqn_aux/successor_as/best/",
     "online_measure": "data/output/result/gridhard/linear_vf/online_property/dqn_aux/successor_as/best/"
     },
    {"label": "FTA eta=0.2",
     "control": "data/output/result/gridhard/linear_vf/control/early_stop/similar_task/fix_rep/dqn_lta/eta_study_0.2_best/",
     "online_measure": "data/output/result/gridhard/linear_vf/online_property/dqn_lta/eta_study_0.2_best/"
     },
    {"label": "FTA eta=0.4",
     "control": "data/output/result/gridhard/linear_vf/control/early_stop/similar_task/fix_rep/dqn_lta/eta_study_0.4_best/",
     "online_measure": "data/output/result/gridhard/linear_vf/online_property/dqn_lta/eta_study_0.4_best/"
     },
    {"label": "FTA eta=0.6",
     "control": "data/output/result/gridhard/linear_vf/control/early_stop/similar_task/fix_rep/dqn_lta/eta_study_0.6_best/",
     "online_measure": "data/output/result/gridhard/linear_vf/online_property/dqn_lta/eta_study_0.6_best/"
     },
    {"label": "FTA eta=0.8",
     "control": "data/output/result/gridhard/linear_vf/control/early_stop/similar_task/fix_rep/dqn_lta/eta_study_0.8_best/",
     "online_measure": "data/output/result/gridhard/linear_vf/online_property/dqn_lta/eta_study_0.8_best/"
     },
    {"label": "FTA+Control1g",
     "control": "data/output/result/gridhard/linear_vf/control/early_stop/similar_task/fix_rep/dqn_lta_aux/aux_control/best_1g_gamma0.9/",
     "online_measure": "data/output/result/gridhard/linear_vf/online_property/dqn_lta_aux/aux_control/best_1g_gamma0.9/"
     },
    {"label": "FTA+Control5g",
     "control": "data/output/result/gridhard/linear_vf/control/early_stop/similar_task/fix_rep/dqn_lta_aux/aux_control/best_5g_gamma0.9/",
     "online_measure": "data/output/result/gridhard/linear_vf/online_property/dqn_lta_aux/aux_control/best_5g_gamma0.9/"
     },
    {"label": "FTA+XY",
     "control": "data/output/result/gridhard/linear_vf/control/early_stop/similar_task/fix_rep/dqn_lta_aux/info/best/",
     "online_measure": "data/output/result/gridhard/linear_vf/online_property/dqn_lta_aux/info/best/"
     },
    {"label": "FTA+Decoder",
     "control": "data/output/result/gridhard/linear_vf/control/early_stop/similar_task/fix_rep/dqn_lta_aux/input_decoder/best/",
     "online_measure": "data/output/result/gridhard/linear_vf/online_property/dqn_lta_aux/input_decoder/best/"
     },
    {"label": "FTA+NAS",
     "control": "data/output/result/gridhard/linear_vf/control/early_stop/similar_task/fix_rep/dqn_lta_aux/nas_v2_delta/best/",
     "online_measure": "data/output/result/gridhard/linear_vf/online_property/dqn_lta_aux/nas_v2_delta/best/"
     },
    {"label": "FTA+Reward",
     "control": "data/output/result/gridhard/linear_vf/control/early_stop/similar_task/fix_rep/dqn_lta_aux/reward/best/",
     "online_measure": "data/output/result/gridhard/linear_vf/online_property/dqn_lta_aux/reward/best/"
     },
    {"label": "FTA+SF",
     "control": "data/output/result/gridhard/linear_vf/control/early_stop/similar_task/fix_rep/dqn_lta_aux/successor_as/best/",
     "online_measure": "data/output/result/gridhard/linear_vf/online_property/dqn_lta_aux/successor_as/best/"
     },
    {"label": "Random",
     "control": "data/output/result/gridhard/linear_vf/control/early_stop/similar_task/fix_rep/random/best/",
     "online_measure": "data/output/result/gridhard/linear_vf/fixrep_property/random/best",
     },
    {"label": "Input",
     "control": "data/output/result/gridhard/linear_vf/control/early_stop/similar_task/fix_rep/input/best/",
     "online_measure": "data/output/result/gridhard/linear_vf/fixrep_property/input/best",
     },
]

gh_diff_early = [
    {"label": "ReLU", "property": "data/output/result/",
     "control": "data/output/result/gridhard/linear_vf/control/early_stop/different_task/fix_rep/dqn/best/",
     "online_measure": "data/output/result/gridhard/linear_vf/online_property/dqn/best/"
     },
    {"label": "ReLU+Control1g",
     "control": "data/output/result/gridhard/linear_vf/control/early_stop/different_task/fix_rep/dqn_aux/aux_control/best_1g_gamma0.9/",
     "online_measure": "data/output/result/gridhard/linear_vf/online_property/dqn_aux/aux_control/best_1g_gamma0.9/"
     },
    {"label": "ReLU+Control5g",
     "control": "data/output/result/gridhard/linear_vf/control/early_stop/different_task/fix_rep/dqn_aux/aux_control/best_5g_gamma0.9/",
     "online_measure": "data/output/result/gridhard/linear_vf/online_property/dqn_aux/aux_control/best_5g_gamma0.9/"
     },
    {"label": "ReLU+XY",
     "control": "data/output/result/gridhard/linear_vf/control/early_stop/different_task/fix_rep/dqn_aux/info/best/",
     "online_measure": "data/output/result/gridhard/linear_vf/online_property/dqn_aux/info/best/"
     },
    {"label": "ReLU+Decoder",
     "control": "data/output/result/gridhard/linear_vf/control/early_stop/different_task/fix_rep/dqn_aux/input_decoder/best/",
     "online_measure": "data/output/result/gridhard/linear_vf/online_property/dqn_aux/info/best/"
     },
    {"label": "ReLU+NAS",
     "control": "data/output/result/gridhard/linear_vf/control/early_stop/different_task/fix_rep/dqn_aux/nas_v2_delta/best/",
     "online_measure": "data/output/result/gridhard/linear_vf/online_property/dqn_aux/nas_v2_delta/best/"
     },
    {"label": "ReLU+Reward",
     "control": "data/output/result/gridhard/linear_vf/control/early_stop/different_task/fix_rep/dqn_aux/reward/best/",
     "online_measure": "data/output/result/gridhard/linear_vf/online_property/dqn_aux/reward/best/"
     },
    {"label": "ReLU+SF",
     "control": "data/output/result/gridhard/linear_vf/control/early_stop/different_task/fix_rep/dqn_aux/successor_as/best/",
     "online_measure": "data/output/result/gridhard/linear_vf/online_property/dqn_aux/successor_as/best/"
     },
    {"label": "FTA eta=0.2",
     "control": "data/output/result/gridhard/linear_vf/control/early_stop/different_task/fix_rep/dqn_lta/eta_study_0.2_best/",
     "online_measure": "data/output/result/gridhard/linear_vf/online_property/dqn_lta/eta_study_0.2_best/"
     },
    {"label": "FTA eta=0.4",
     "control": "data/output/result/gridhard/linear_vf/control/early_stop/different_task/fix_rep/dqn_lta/eta_study_0.4_best/",
     "online_measure": "data/output/result/gridhard/linear_vf/online_property/dqn_lta/eta_study_0.4_best/"
     },
    {"label": "FTA eta=0.6",
     "control": "data/output/result/gridhard/linear_vf/control/early_stop/different_task/fix_rep/dqn_lta/eta_study_0.6_best/",
     "online_measure": "data/output/result/gridhard/linear_vf/online_property/dqn_lta/eta_study_0.6_best/"
     },
    {"label": "FTA eta=0.8",
     "control": "data/output/result/gridhard/linear_vf/control/early_stop/different_task/fix_rep/dqn_lta/eta_study_0.8_best/",
     "online_measure": "data/output/result/gridhard/linear_vf/online_property/dqn_lta/eta_study_0.8_best/"
     },
    {"label": "FTA+Control1g",
     "control": "data/output/result/gridhard/linear_vf/control/early_stop/different_task/fix_rep/dqn_lta_aux/aux_control/best_1g_gamma0.9/",
     "online_measure": "data/output/result/gridhard/linear_vf/online_property/dqn_lta_aux/aux_control/best_1g_gamma0.9/"
     },
    {"label": "FTA+Control5g",
     "control": "data/output/result/gridhard/linear_vf/control/early_stop/different_task/fix_rep/dqn_lta_aux/aux_control/best_5g_gamma0.9/",
     "online_measure": "data/output/result/gridhard/linear_vf/online_property/dqn_lta_aux/aux_control/best_5g_gamma0.9/"
     },
    {"label": "FTA+XY",
     "control": "data/output/result/gridhard/linear_vf/control/early_stop/different_task/fix_rep/dqn_lta_aux/info/best/",
     "online_measure": "data/output/result/gridhard/linear_vf/online_property/dqn_lta_aux/info/best/"
     },
    {"label": "FTA+Decoder",
     "control": "data/output/result/gridhard/linear_vf/control/early_stop/different_task/fix_rep/dqn_lta_aux/input_decoder/best/",
     "online_measure": "data/output/result/gridhard/linear_vf/online_property/dqn_lta_aux/input_decoder/best/"
     },
    {"label": "FTA+NAS",
     "control": "data/output/result/gridhard/linear_vf/control/early_stop/different_task/fix_rep/dqn_lta_aux/nas_v2_delta/best/",
     "online_measure": "data/output/result/gridhard/linear_vf/online_property/dqn_lta_aux/nas_v2_delta/best/"
     },
    {"label": "FTA+Reward",
     "control": "data/output/result/gridhard/linear_vf/control/early_stop/different_task/fix_rep/dqn_lta_aux/reward/best/",
     "online_measure": "data/output/result/gridhard/linear_vf/online_property/dqn_lta_aux/reward/best/"
     },
    {"label": "FTA+SF",
     "control": "data/output/result/gridhard/linear_vf/control/early_stop/different_task/fix_rep/dqn_lta_aux/successor_as/best/",
     "online_measure": "data/output/result/gridhard/linear_vf/online_property/dqn_lta_aux/successor_as/best/"
     },
    {"label": "Random",
     "control": "data/output/result/gridhard/linear_vf/control/early_stop/different_task/fix_rep/random/best/",
     "online_measure": "data/output/result/gridhard/linear_vf/fixrep_property/random/best",
     },
    {"label": "Input",
     "control": "data/output/result/gridhard/linear_vf/control/early_stop/different_task/fix_rep/input/best/",
     "online_measure": "data/output/result/gridhard/linear_vf/fixrep_property/input/best",
     },
    {"label": "Scratch",
     "control": "data/output/result/gridhard/linear_vf/control/early_stop/different_task/scratch/dqn/best/",
     },
]

# gh_diff_tune_early = [
#     {"label": "ReLU", "property": "data/output/result/",
#      "control": "data/output/result/gridhard/linear_vf/control/early_stop/different_task/fine_tune/dqn/best/",
#      "online_measure": "data/output/result/gridhard/linear_vf/online_property/dqn/best/"
#      },
#     {"label": "ReLU+Control1g",
#      "control": "data/output/result/gridhard/linear_vf/control/early_stop/different_task/fine_tune/dqn_aux/aux_control/best_1g_gamma0.9/",
#      "online_measure": "data/output/result/gridhard/linear_vf/online_property/dqn_aux/aux_control/best_1g_gamma0.9/"
#      },
#     {"label": "ReLU+Control5g",
#      "control": "data/output/result/gridhard/linear_vf/control/early_stop/different_task/fine_tune/dqn_aux/aux_control/best_5g_gamma0.9/",
#      "online_measure": "data/output/result/gridhard/linear_vf/online_property/dqn_aux/aux_control/best_5g_gamma0.9/"
#      },
#     {"label": "ReLU+XY",
#      "control": "data/output/result/gridhard/linear_vf/control/early_stop/different_task/fine_tune/dqn_aux/info/best/",
#      "online_measure": "data/output/result/gridhard/linear_vf/online_property/dqn_aux/info/best/"
#      },
#     {"label": "ReLU+Decoder",
#      "control": "data/output/result/gridhard/linear_vf/control/early_stop/different_task/fine_tune/dqn_aux/input_decoder/best/",
#      "online_measure": "data/output/result/gridhard/linear_vf/online_property/dqn_aux/info/best/"
#      },
#     {"label": "ReLU+NAS",
#      "control": "data/output/result/gridhard/linear_vf/control/early_stop/different_task/fine_tune/dqn_aux/nas_v2_delta/best/",
#      "online_measure": "data/output/result/gridhard/linear_vf/online_property/dqn_aux/nas_v2_delta/best/"
#      },
#     {"label": "ReLU+Reward",
#      "control": "data/output/result/gridhard/linear_vf/control/early_stop/different_task/fine_tune/dqn_aux/reward/best/",
#      "online_measure": "data/output/result/gridhard/linear_vf/online_property/dqn_aux/reward/best/"
#      },
#     {"label": "ReLU+SF",
#      "control": "data/output/result/gridhard/linear_vf/control/early_stop/different_task/fine_tune/dqn_aux/successor_as/best/",
#      "online_measure": "data/output/result/gridhard/linear_vf/online_property/dqn_aux/successor_as/best/"
#      },
#     {"label": "FTA eta=0.2",
#      "control": "data/output/result/gridhard/linear_vf/control/early_stop/different_task/fine_tune/dqn_lta/eta_study_0.2_best/",
#      "online_measure": "data/output/result/gridhard/linear_vf/online_property/dqn_lta/eta_study_0.2_best/"
#      },
#     {"label": "FTA eta=0.4",
#      "control": "data/output/result/gridhard/linear_vf/control/early_stop/different_task/fine_tune/dqn_lta/eta_study_0.4_best/",
#      "online_measure": "data/output/result/gridhard/linear_vf/online_property/dqn_lta/eta_study_0.4_best/"
#      },
#     {"label": "FTA eta=0.6",
#      "control": "data/output/result/gridhard/linear_vf/control/early_stop/different_task/fine_tune/dqn_lta/eta_study_0.6_best/",
#      "online_measure": "data/output/result/gridhard/linear_vf/online_property/dqn_lta/eta_study_0.6_best/"
#      },
#     {"label": "FTA eta=0.8",
#      "control": "data/output/result/gridhard/linear_vf/control/early_stop/different_task/fine_tune/dqn_lta/eta_study_0.8_best/",
#      "online_measure": "data/output/result/gridhard/linear_vf/online_property/dqn_lta/eta_study_0.8_best/"
#      },
#     {"label": "FTA+Control1g",
#      "control": "data/output/result/gridhard/linear_vf/control/early_stop/different_task/fine_tune/dqn_lta_aux/aux_control/best_1g_gamma0.9/",
#      "online_measure": "data/output/result/gridhard/linear_vf/online_property/dqn_lta_aux/aux_control/best_1g_gamma0.9/"
#      },
#     {"label": "FTA+Control5g",
#      "control": "data/output/result/gridhard/linear_vf/control/early_stop/different_task/fine_tune/dqn_lta_aux/aux_control/best_5g_gamma0.9/",
#      "online_measure": "data/output/result/gridhard/linear_vf/online_property/dqn_lta_aux/aux_control/best_5g_gamma0.9/"
#      },
#     {"label": "FTA+XY",
#      "control": "data/output/result/gridhard/linear_vf/control/early_stop/different_task/fine_tune/dqn_lta_aux/info/best/",
#      "online_measure": "data/output/result/gridhard/linear_vf/online_property/dqn_lta_aux/info/best/"
#      },
#     {"label": "FTA+Decoder",
#      "control": "data/output/result/gridhard/linear_vf/control/early_stop/different_task/fine_tune/dqn_lta_aux/input_decoder/best/",
#      "online_measure": "data/output/result/gridhard/linear_vf/online_property/dqn_lta_aux/input_decoder/best/"
#      },
#     {"label": "FTA+NAS",
#      "control": "data/output/result/gridhard/linear_vf/control/early_stop/different_task/fine_tune/dqn_lta_aux/nas_v2_delta/best/",
#      "online_measure": "data/output/result/gridhard/linear_vf/online_property/dqn_lta_aux/nas_v2_delta/best/"
#      },
#     {"label": "FTA+Reward",
#      "control": "data/output/result/gridhard/linear_vf/control/early_stop/different_task/fine_tune/dqn_lta_aux/reward/best/",
#      "online_measure": "data/output/result/gridhard/linear_vf/online_property/dqn_lta_aux/reward/best/"
#      },
#     {"label": "FTA+SF",
#      "control": "data/output/result/gridhard/linear_vf/control/early_stop/different_task/fine_tune/dqn_lta_aux/successor_as/best/",
#      "online_measure": "data/output/result/gridhard/linear_vf/online_property/dqn_lta_aux/successor_as/best/"
#      },
#     {"label": "Random",
#      "control": "data/output/result/gridhard/linear_vf/control/baseline/different_task/fine_tune/random/best/",
#      "online_measure": "data/output/result/gridhard/linear_vf/fixrep_property/random/best",
#      },
# ]

gh_same_last = [
    {"label": "ReLU", "property": "data/output/test/",
     "control": "data/output/test/gridhard/control/last/same_task/fix_rep/dqn/best/",
     "online_measure": "data/output/test/gridhard/online_property/dqn/best/"
     },
    {"label": "ReLU+Control1g",
     "control": "data/output/test/gridhard/control/last/same_task/fix_rep/dqn_aux/aux_control/best_1g/",
     "online_measure": "data/output/test/gridhard/online_property/dqn_aux/freq8/aux_control/best_1g/"
     },
    {"label": "ReLU+Control5g",
     "control": "data/output/test/gridhard/control/last/same_task/fix_rep/dqn_aux/aux_control/best_5g/",
     "online_measure": "data/output/test/gridhard/online_property/dqn_aux/freq128/aux_control/best_5g/"
     },
    {"label": "ReLU+XY",
     "control": "data/output/test/gridhard/control/last/same_task/fix_rep/dqn_aux/info/best/",
     "online_measure": "data/output/test/gridhard/online_property/dqn_aux/freq8/info/best/"
     },
    {"label": "ReLU+Decoder",
     "control": "data/output/test/gridhard/control/last/same_task/fix_rep/dqn_aux/input_decoder/best/",
     "online_measure": "data/output/test/gridhard/online_property/dqn_aux/freq8/info/best/"
     },
    {"label": "ReLU+NAS",
     "control": "data/output/test/gridhard/control/last/same_task/fix_rep/dqn_aux/nas_v2_delta/best/",
     "online_measure": "data/output/test/gridhard/online_property/dqn_aux/freq8/nas_v2_delta/best/"
     },
    {"label": "ReLU+Reward",
     "control": "data/output/test/gridhard/control/last/same_task/fix_rep/dqn_aux/reward/best/",
     "online_measure": "data/output/test/gridhard/online_property/dqn_aux/freq8/reward/best/"
     },
    {"label": "ReLU+SF",
     "control": "data/output/test/gridhard/control/last/same_task/fix_rep/dqn_aux/successor_as/best/",
     "online_measure": "data/output/test/gridhard/online_property/dqn_aux/freq8/successor_as/best/"
     },
    {"label": "FTA eta=0.2",
     "control": "data/output/test/gridhard/control/last/same_task/fix_rep/dqn_lta/eta_study_0.2_best/",
     "online_measure": "data/output/test/gridhard/online_property/dqn_lta/eta_study_0.2_best/"
     },
    {"label": "FTA eta=0.4",
     "control": "data/output/test/gridhard/control/last/same_task/fix_rep/dqn_lta/eta_study_0.4_best/",
     "online_measure": "data/output/test/gridhard/online_property/dqn_lta/eta_study_0.4_best/"
     },
    {"label": "FTA eta=0.6",
     "control": "data/output/test/gridhard/control/last/same_task/fix_rep/dqn_lta/eta_study_0.6_best/",
     "online_measure": "data/output/test/gridhard/online_property/dqn_lta/eta_study_0.6_best/"
     },
    {"label": "FTA eta=0.8",
     "control": "data/output/test/gridhard/control/last/same_task/fix_rep/dqn_lta/eta_study_0.8_best/",
     "online_measure": "data/output/test/gridhard/online_property/dqn_lta/eta_study_0.8_best/"
     },
    {"label": "FTA+Control1g",
     "control": "data/output/test/gridhard/control/last/same_task/fix_rep/dqn_lta_aux/aux_control/best_1g/",
     "online_measure": "data/output/test/gridhard/online_property/dqn_lta_aux/aux_control/best_1g/"
     },
    {"label": "FTA+Control5g",
     "control": "data/output/test/gridhard/control/last/same_task/fix_rep/dqn_lta_aux/aux_control/best_5g/",
     "online_measure": "data/output/test/gridhard/online_property/dqn_lta_aux/aux_control/best_5g/"
     },
    {"label": "FTA+XY",
     "control": "data/output/test/gridhard/control/last/same_task/fix_rep/dqn_lta_aux/info/best/",
     "online_measure": "data/output/test/gridhard/online_property/dqn_lta_aux/info/best/"
     },
    {"label": "FTA+Decoder",
     "control": "data/output/test/gridhard/control/last/same_task/fix_rep/dqn_lta_aux/input_decoder/best/",
     "online_measure": "data/output/test/gridhard/online_property/dqn_lta_aux/input_decoder/best/"
     },
    {"label": "FTA+NAS",
     "control": "data/output/test/gridhard/control/last/same_task/fix_rep/dqn_lta_aux/nas_v2_delta/best/",
     "online_measure": "data/output/test/gridhard/online_property/dqn_lta_aux/nas_v2_delta/best/"
     },
    {"label": "FTA+Reward",
     "control": "data/output/test/gridhard/control/last/same_task/fix_rep/dqn_lta_aux/reward/best/",
     "online_measure": "data/output/test/gridhard/online_property/dqn_lta_aux/reward/best/"
     },
    {"label": "FTA+SF",
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
    {"label": "ReLU+Control1g",
     "control": "data/output/test/gridhard/control/last/similar_task/fix_rep/dqn_aux/aux_control/best_1g/",
     "online_measure": "data/output/test/gridhard/online_property/dqn_aux/freq8/aux_control/best_1g/"
     },
    {"label": "ReLU+Control5g",
     "control": "data/output/test/gridhard/control/last/similar_task/fix_rep/dqn_aux/aux_control/best_5g/",
     "online_measure": "data/output/test/gridhard/online_property/dqn_aux/freq128/aux_control/best_5g/"
     },
    {"label": "ReLU+XY",
     "control": "data/output/test/gridhard/control/last/similar_task/fix_rep/dqn_aux/info/best/",
     "online_measure": "data/output/test/gridhard/online_property/dqn_aux/freq8/info/best/"
     },
    {"label": "ReLU+Decoder",
     "control": "data/output/test/gridhard/control/last/similar_task/fix_rep/dqn_aux/input_decoder/best/",
     "online_measure": "data/output/test/gridhard/online_property/dqn_aux/freq8/info/best/"
     },
    {"label": "ReLU+NAS",
     "control": "data/output/test/gridhard/control/last/similar_task/fix_rep/dqn_aux/nas_v2_delta/best/",
     "online_measure": "data/output/test/gridhard/online_property/dqn_aux/freq8/nas_v2_delta/best/"
     },
    {"label": "ReLU+Reward",
     "control": "data/output/test/gridhard/control/last/similar_task/fix_rep/dqn_aux/reward/best/",
     "online_measure": "data/output/test/gridhard/online_property/dqn_aux/freq8/reward/best/"
     },
    {"label": "ReLU+SF",
     "control": "data/output/test/gridhard/control/last/similar_task/fix_rep/dqn_aux/successor_as/best/",
     "online_measure": "data/output/test/gridhard/online_property/dqn_aux/freq8/successor_as/best/"
     },
    {"label": "FTA eta=0.2",
     "control": "data/output/test/gridhard/control/last/similar_task/fix_rep/dqn_lta/eta_study_0.2_best/",
     "online_measure": "data/output/test/gridhard/online_property/dqn_lta/eta_study_0.2_best/"
     },
    {"label": "FTA eta=0.4",
     "control": "data/output/test/gridhard/control/last/similar_task/fix_rep/dqn_lta/eta_study_0.4_best/",
     "online_measure": "data/output/test/gridhard/online_property/dqn_lta/eta_study_0.4_best/"
     },
    {"label": "FTA eta=0.6",
     "control": "data/output/test/gridhard/control/last/similar_task/fix_rep/dqn_lta/eta_study_0.6_best/",
     "online_measure": "data/output/test/gridhard/online_property/dqn_lta/eta_study_0.6_best/"
     },
    {"label": "FTA eta=0.8",
     "control": "data/output/test/gridhard/control/last/similar_task/fix_rep/dqn_lta/eta_study_0.8_best/",
     "online_measure": "data/output/test/gridhard/online_property/dqn_lta/eta_study_0.8_best/"
     },
    {"label": "FTA+Control1g",
     "control": "data/output/test/gridhard/control/last/similar_task/fix_rep/dqn_lta_aux/aux_control/best_1g/",
     "online_measure": "data/output/test/gridhard/online_property/dqn_lta_aux/aux_control/best_1g/"
     },
    {"label": "FTA+Control5g",
     "control": "data/output/test/gridhard/control/last/similar_task/fix_rep/dqn_lta_aux/aux_control/best_5g/",
     "online_measure": "data/output/test/gridhard/online_property/dqn_lta_aux/aux_control/best_5g/"
     },
    {"label": "FTA+XY",
     "control": "data/output/test/gridhard/control/last/similar_task/fix_rep/dqn_lta_aux/info/best/",
     "online_measure": "data/output/test/gridhard/online_property/dqn_lta_aux/info/best/"
     },
    {"label": "FTA+Decoder",
     "control": "data/output/test/gridhard/control/last/similar_task/fix_rep/dqn_lta_aux/input_decoder/best/",
     "online_measure": "data/output/test/gridhard/online_property/dqn_lta_aux/input_decoder/best/"
     },
    {"label": "FTA+NAS",
     "control": "data/output/test/gridhard/control/last/similar_task/fix_rep/dqn_lta_aux/nas_v2_delta/best/",
     "online_measure": "data/output/test/gridhard/online_property/dqn_lta_aux/nas_v2_delta/best/"
     },
    {"label": "FTA+Reward",
     "control": "data/output/test/gridhard/control/last/similar_task/fix_rep/dqn_lta_aux/reward/best/",
     "online_measure": "data/output/test/gridhard/online_property/dqn_lta_aux/reward/best/"
     },
    {"label": "FTA+SF",
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
    {"label": "ReLU+Control1g",
     "control": "data/output/test/gridhard/control/last/different_task/fix_rep/dqn_aux/aux_control/best_1g/",
     "online_measure": "data/output/test/gridhard/online_property/dqn_aux/freq8/aux_control/best_1g/"
     },
    {"label": "ReLU+Control5g",
     "control": "data/output/test/gridhard/control/last/different_task/fix_rep/dqn_aux/aux_control/best_5g/",
     "online_measure": "data/output/test/gridhard/online_property/dqn_aux/freq128/aux_control/best_5g/"
     },
    {"label": "ReLU+XY",
     "control": "data/output/test/gridhard/control/last/different_task/fix_rep/dqn_aux/info/best/",
     "online_measure": "data/output/test/gridhard/online_property/dqn_aux/freq8/info/best/"
     },
    {"label": "ReLU+Decoder",
     "control": "data/output/test/gridhard/control/last/different_task/fix_rep/dqn_aux/input_decoder/best/",
     "online_measure": "data/output/test/gridhard/online_property/dqn_aux/freq8/info/best/"
     },
    {"label": "ReLU+NAS",
     "control": "data/output/test/gridhard/control/last/different_task/fix_rep/dqn_aux/nas_v2_delta/best/",
     "online_measure": "data/output/test/gridhard/online_property/dqn_aux/freq8/nas_v2_delta/best/"
     },
    {"label": "ReLU+Reward",
     "control": "data/output/test/gridhard/control/last/different_task/fix_rep/dqn_aux/reward/best/",
     "online_measure": "data/output/test/gridhard/online_property/dqn_aux/freq8/reward/best/"
     },
    {"label": "ReLU+SF",
     "control": "data/output/test/gridhard/control/last/different_task/fix_rep/dqn_aux/successor_as/best/",
     "online_measure": "data/output/test/gridhard/online_property/dqn_aux/freq8/successor_as/best/"
     },
    {"label": "FTA eta=0.2",
     "control": "data/output/test/gridhard/control/last/different_task/fix_rep/dqn_lta/eta_study_0.2_best/",
     "online_measure": "data/output/test/gridhard/online_property/dqn_lta/eta_study_0.2_best/"
     },
    {"label": "FTA eta=0.4",
     "control": "data/output/test/gridhard/control/last/different_task/fix_rep/dqn_lta/eta_study_0.4_best/",
     "online_measure": "data/output/test/gridhard/online_property/dqn_lta/eta_study_0.4_best/"
     },
    {"label": "FTA eta=0.6",
     "control": "data/output/test/gridhard/control/last/different_task/fix_rep/dqn_lta/eta_study_0.6_best/",
     "online_measure": "data/output/test/gridhard/online_property/dqn_lta/eta_study_0.6_best/"
     },
    {"label": "FTA eta=0.8",
     "control": "data/output/test/gridhard/control/last/different_task/fix_rep/dqn_lta/eta_study_0.8_best/",
     "online_measure": "data/output/test/gridhard/online_property/dqn_lta/eta_study_0.8_best/"
     },
    {"label": "FTA+Control1g",
     "control": "data/output/test/gridhard/control/last/different_task/fix_rep/dqn_lta_aux/aux_control/best_1g/",
     "online_measure": "data/output/test/gridhard/online_property/dqn_lta_aux/aux_control/best_1g/"
     },
    {"label": "FTA+Control5g",
     "control": "data/output/test/gridhard/control/last/different_task/fix_rep/dqn_lta_aux/aux_control/best_5g/",
     "online_measure": "data/output/test/gridhard/online_property/dqn_lta_aux/aux_control/best_5g/"
     },
    {"label": "FTA+XY",
     "control": "data/output/test/gridhard/control/last/different_task/fix_rep/dqn_lta_aux/info/best/",
     "online_measure": "data/output/test/gridhard/online_property/dqn_lta_aux/info/best/"
     },
    {"label": "FTA+Decoder",
     "control": "data/output/test/gridhard/control/last/different_task/fix_rep/dqn_lta_aux/input_decoder/best/",
     "online_measure": "data/output/test/gridhard/online_property/dqn_lta_aux/input_decoder/best/"
     },
    {"label": "FTA+NAS",
     "control": "data/output/test/gridhard/control/last/different_task/fix_rep/dqn_lta_aux/nas_v2_delta/best/",
     "online_measure": "data/output/test/gridhard/online_property/dqn_lta_aux/nas_v2_delta/best/"
     },
    {"label": "FTA+Reward",
     "control": "data/output/test/gridhard/control/last/different_task/fix_rep/dqn_lta_aux/reward/best/",
     "online_measure": "data/output/test/gridhard/online_property/dqn_lta_aux/reward/best/"
     },
    {"label": "FTA+SF",
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
    {"label": "ReLU+Control1g",
     "control": "data/output/test/gridhard/control/last/different_task/fine_tune/dqn_aux/aux_control/best_1g/",
     "online_measure": "data/output/test/gridhard/online_property/dqn_aux/freq8/aux_control/best_1g/"
     },
    {"label": "ReLU+Control5g",
     "control": "data/output/test/gridhard/control/last/different_task/fine_tune/dqn_aux/aux_control/best_5g/",
     "online_measure": "data/output/test/gridhard/online_property/dqn_aux/freq128/aux_control/best_5g/"
     },
    {"label": "ReLU+XY",
     "control": "data/output/test/gridhard/control/last/different_task/fine_tune/dqn_aux/info/best/",
     "online_measure": "data/output/test/gridhard/online_property/dqn_aux/freq8/info/best/"
     },
    {"label": "ReLU+Decoder",
     "control": "data/output/test/gridhard/control/last/different_task/fine_tune/dqn_aux/input_decoder/best/",
     "online_measure": "data/output/test/gridhard/online_property/dqn_aux/freq8/info/best/"
     },
    {"label": "ReLU+NAS",
     "control": "data/output/test/gridhard/control/last/different_task/fine_tune/dqn_aux/nas_v2_delta/best/",
     "online_measure": "data/output/test/gridhard/online_property/dqn_aux/freq8/nas_v2_delta/best/"
     },
    {"label": "ReLU+Reward",
     "control": "data/output/test/gridhard/control/last/different_task/fine_tune/dqn_aux/reward/best/",
     "online_measure": "data/output/test/gridhard/online_property/dqn_aux/freq8/reward/best/"
     },
    {"label": "ReLU+SF",
     "control": "data/output/test/gridhard/control/last/different_task/fine_tune/dqn_aux/successor_as/best/",
     "online_measure": "data/output/test/gridhard/online_property/dqn_aux/freq8/successor_as/best/"
     },
    {"label": "FTA eta=0.2",
     "control": "data/output/test/gridhard/control/last/different_task/fine_tune/dqn_lta/eta_study_0.2_best/",
     "online_measure": "data/output/test/gridhard/online_property/dqn_lta/eta_study_0.2_best/"
     },
    {"label": "FTA eta=0.4",
     "control": "data/output/test/gridhard/control/last/different_task/fine_tune/dqn_lta/eta_study_0.4_best/",
     "online_measure": "data/output/test/gridhard/online_property/dqn_lta/eta_study_0.4_best/"
     },
    {"label": "FTA eta=0.6",
     "control": "data/output/test/gridhard/control/last/different_task/fine_tune/dqn_lta/eta_study_0.6_best/",
     "online_measure": "data/output/test/gridhard/online_property/dqn_lta/eta_study_0.6_best/"
     },
    {"label": "FTA eta=0.8",
     "control": "data/output/test/gridhard/control/last/different_task/fine_tune/dqn_lta/eta_study_0.8_best/",
     "online_measure": "data/output/test/gridhard/online_property/dqn_lta/eta_study_0.8_best/"
     },
    {"label": "FTA+Control1g",
     "control": "data/output/test/gridhard/control/last/different_task/fine_tune/dqn_lta_aux/aux_control/best_1g/",
     "online_measure": "data/output/test/gridhard/online_property/dqn_lta_aux/aux_control/best_1g/"
     },
    {"label": "FTA+Control5g",
     "control": "data/output/test/gridhard/control/last/different_task/fine_tune/dqn_lta_aux/aux_control/best_5g/",
     "online_measure": "data/output/test/gridhard/online_property/dqn_lta_aux/aux_control/best_5g/"
     },
    {"label": "FTA+XY",
     "control": "data/output/test/gridhard/control/last/different_task/fine_tune/dqn_lta_aux/info/best/",
     "online_measure": "data/output/test/gridhard/online_property/dqn_lta_aux/info/best/"
     },
    {"label": "FTA+Decoder",
     "control": "data/output/test/gridhard/control/last/different_task/fine_tune/dqn_lta_aux/input_decoder/best/",
     "online_measure": "data/output/test/gridhard/online_property/dqn_lta_aux/input_decoder/best/"
     },
    {"label": "FTA+NAS",
     "control": "data/output/test/gridhard/control/last/different_task/fine_tune/dqn_lta_aux/nas_v2_delta/best/",
     "online_measure": "data/output/test/gridhard/online_property/dqn_lta_aux/nas_v2_delta/best/"
     },
    {"label": "FTA+Reward",
     "control": "data/output/test/gridhard/control/last/different_task/fine_tune/dqn_lta_aux/reward/best/",
     "online_measure": "data/output/test/gridhard/online_property/dqn_lta_aux/reward/best/"
     },
    {"label": "FTA+SF",
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
    {"label": "FTA",
     "control": "data/output/test/picky_eater/online_property/dqn_lta/sweep",
     },
    ]

dqn_lta_1_learn_sweep = [
    {"label": "FTA+1",
     "control": "data/output/test/picky_eater/online_property/dqn_lta_1/sweep",
     },
    ]


dqn_learn_sweep = [
    {"label": "ReLU",
     "control": "data/output/test/picky_eater/online_property/dqn/sweep",
     },
    ]

crgb_online_dqn = [
    {"label": "DQN",
     "control": "data/output/test/picky_eater/online_property/dqn/best/",
     },
   {"label": "DQN+Reward",
     "control": "data/output/test/picky_eater/online_property/dqn_aux/reward/initial/",
     },
    {"label": "DQN+AuxControl", "control": "data/output/test/picky_eater/online_property/dqn_aux/aux_control/initial/",
     },    
    {"label": "DQN+XY",
     "control": "data/output/test/picky_eater/online_property/dqn_aux/info/initial_xy/",
     },
    {"label": "DQN+Decoder",
     "control": "data/output/test/picky_eater/online_property/dqn_aux/input_decoder/initial/",
     },
    {"label": "DQN+NAS",
     "control": "data/output/test/picky_eater/online_property/dqn_aux/nas_v2_delta/initial/",
     },
    {"label": "DQN+SF",
     "control": "data/output/test/picky_eater/online_property/dqn_aux/successor_as/initial/",
     },
]
crgb_online_dqn_lta = [
    {"label": "DQN+LTA",
     "control": "data/output/test/picky_eater/online_property/dqn_lta/best/",
     },
 
    {"label": "DQN+LTA+AuxControl",
     "control": "data/output/test/picky_eater/online_property/dqn_lta_aux/aux_control/initial/",
     },    
    {"label": "DQN+LTA+XY",
     "control": "data/output/test/picky_eater/online_property/dqn_lta_aux/info/initial_xy/",
     },
    {"label": "DQN+LTA+Decoder",
     "control": "data/output/test/picky_eater/online_property/dqn_lta_aux/input_decoder/initial/",
     },
    {"label": "DQN+LTA+NAS",
     "control": "data/output/test/picky_eater/online_property/dqn_lta_aux/nas_v2_delta/initial/",
     },
    {"label": "DQN+LTA+SF",
     "control": "data/output/test/picky_eater/online_property/dqn_lta_aux/successor_as/initial/",
     },
    {"label": "DQN+LTA+Reward",
     "control": "data/output/test/picky_eater/online_property/dqn_lta_aux/reward/initial/",
    },
    ]

crgb_online = [
    {"label": "FTA eta=2",
     "control": "data/output/test/picky_eater/online_property/dqn_lta/best/",
     },
    {"label": "FTA+Control",
     "control": "data/output/test/picky_eater/online_property/dqn_lta_aux/aux_control/best/",
     },    
    {"label": "FTA+XY",
     "control": "data/output/test/picky_eater/online_property/dqn_lta_aux/info/best_xy/",
     },
    {"label": "FTA+Decoder",
     "control": "data/output/test/picky_eater/online_property/dqn_lta_aux/input_decoder/best/",
     },
    {"label": "FTA+NAS",
     "control": "data/output/test/picky_eater/online_property/dqn_lta_aux/nas_v2_delta/best/",
     },
    {"label": "FTA+SF",
     "control": "data/output/test/picky_eater/online_property/dqn_lta_aux/successor_as/best/",
     },
    {"label": "FTA+Reward",
     "control": "data/output/test/picky_eater/online_property/dqn_lta_aux/reward/best/",
    },
    {"label": "ReLU",
     "control": "data/output/test/picky_eater/online_property/dqn/best/",
     },
   {"label": "ReLU+Reward",
     "control": "data/output/test/picky_eater/online_property/dqn_aux/reward/best/",
     },
    {"label": "ReLU+Control", "control": "data/output/test/picky_eater/online_property/dqn_aux/aux_control/best/",
     },    
    {"label": "ReLU+XY",
     "control": "data/output/test/picky_eater/online_property/dqn_aux/info/best_xy/",
     },
     {"label": "ReLU+Decoder",
     "control": "data/output/test/picky_eater/online_property/dqn_aux/input_decoder/best/",
     },
    {"label": "ReLU+NAS",
     "control": "data/output/test/picky_eater/online_property/dqn_aux/nas_v2_delta/best/",
     },
    {"label": "ReLU+SF",
     "control": "data/output/test/picky_eater/online_property/dqn_aux/successor_as/best/",
     },
        ]

crgb_online_st_fr = [
    {"label": "ReLU",
     "control": "data/output/test/picky_eater/control/same_task/fix_rep/dqn/best_early/",
     },
    {"label": "LTA",
     "control": "data/output/test/picky_eater/control/same_task/fix_rep/dqn_lta/best_early/",
     },
]

crgb_online_dt_fr = [
    {"label": "ReLU",
     "control": "data/output/test/picky_eater/control/different_task/fix_rep/dqn/best_early/",
     },
    {"label": "LTA",
     "control": "data/output/test/picky_eater/control/different_task/fix_rep/dqn_lta/best_early/",
     },
]

crgb_online_dt_ft = [
    {"label": "ReLU",
     "control": "data/output/test/picky_eater/control/different_task/fine_tune/dqn/best_early/",
     },
    {"label": "LTA",
     "control": "data/output/test/picky_eater/control/different_task/fine_tune/dqn_lta/best_early/",
     },
]

crgb = [
    {"label": "DQN",
     "control": "data/output/test/picky_eater/control/different_task/fine_tune/dqn/best_early/",
     },
    {"label": "DQN+LTA",
     "control": "data/output/test/picky_eater/control/different_task/fine_tune/dqn_lta/best_early/",
     },
    {"label": "DQN",
     "control": "data/output/test/picky_eater/control/different_task/fix_rep/dqn/best_early/",
     },
    {"label": "DQN+LTA",
     "control": "data/output/test/picky_eater/control/different_task/fix_rep/dqn_lta/best_early/",
     },    
    {"label": "DQN",
     "control": "data/output/test/picky_eater/control/same_task/fix_rep/dqn/best_early/",
     },
    {"label": "DQN+LTA",
     "control": "data/output/test/picky_eater/control/same_task/fix_rep/dqn_lta/best_early/",
     },
        ]


crgb_same_last = [
    {"label": "ReLU",
     "control": "data/output/test/picky_eater/control/same_task/fix_rep/dqn/best_final/",
     "online_measure": "data/output/test/picky_eater/online_property/dqn/best/"
     },
#     {"label": "ReLU+Control",
     # "control": "data/output/test/gridhard/control/last/same_task/fix_rep/dqn_aux/aux_control/best_1g/",
     # "online_measure": "data/output/test/gridhard/online_property/dqn_aux/freq8/aux_control/best_1g/"
     # },
    # {"label": "ReLU+XY",
     # "control": "data/output/test/gridhard/control/last/same_task/fix_rep/dqn_aux/info/best/",
     # "online_measure": "data/output/test/gridhard/online_property/dqn_aux/freq8/info/best/"
     # },
    # {"label": "ReLU+Decoder",
     # "control": "data/output/test/gridhard/control/last/same_task/fix_rep/dqn_aux/input_decoder/best/",
     # "online_measure": "data/output/test/gridhard/online_property/dqn_aux/freq8/info/best/"
     # },
    # {"label": "ReLU+NAS",
     # "control": "data/output/test/gridhard/control/last/same_task/fix_rep/dqn_aux/nas_v2_delta/best/",
     # "online_measure": "data/output/test/gridhard/online_property/dqn_aux/freq8/nas_v2_delta/best/"
     # },
    # {"label": "ReLU+Reward",
     # "control": "data/output/test/gridhard/control/last/same_task/fix_rep/dqn_aux/reward/best/",
     # "online_measure": "data/output/test/gridhard/online_property/dqn_aux/freq8/reward/best/"
     # },
    # {"label": "ReLU+SF",
     # "control": "data/output/test/gridhard/control/last/same_task/fix_rep/dqn_aux/successor_as/best/",
     # "online_measure": "data/output/test/gridhard/online_property/dqn_aux/freq8/successor_as/best/"
     # },
    {"label": "LTA",
     "control": "data/output/test/picky_eater/control/same_task/fix_rep/dqn_lta/best_final/",
     "online_measure": "data/output/test/picky_eater/online_property/dqn_lta/best/"
     },
#     {"label": "LTA+Control",
     # "control": "data/output/test/gridhard/control/last/same_task/fix_rep/dqn_lta_aux/aux_control/best_5g/",
     # "online_measure": "data/output/test/gridhard/online_property/dqn_lta_aux/aux_control/best_5g/"
     # },
    # {"label": "LTA+XY",
     # "control": "data/output/test/gridhard/control/last/same_task/fix_rep/dqn_lta_aux/info/best/",
     # "online_measure": "data/output/test/gridhard/online_property/dqn_lta_aux/info/best/"
     # },
    # {"label": "LTA+Decoder",
     # "control": "data/output/test/gridhard/control/last/same_task/fix_rep/dqn_lta_aux/input_decoder/best/",
     # "online_measure": "data/output/test/gridhard/online_property/dqn_lta_aux/input_decoder/best/"
     # },
    # {"label": "LTA+NAS",
     # "control": "data/output/test/gridhard/control/last/same_task/fix_rep/dqn_lta_aux/nas_v2_delta/best/",
     # "online_measure": "data/output/test/gridhard/online_property/dqn_lta_aux/nas_v2_delta/best/"
     # },
    # {"label": "LTA+Reward",
     # "control": "data/output/test/gridhard/control/last/same_task/fix_rep/dqn_lta_aux/reward/best/",
     # "online_measure": "data/output/test/gridhard/online_property/dqn_lta_aux/reward/best/"
     # },
    # {"label": "LTA+SF",
     # "control": "data/output/test/gridhard/control/last/same_task/fix_rep/dqn_lta_aux/successor_as/best/",
     # "online_measure": "data/output/test/gridhard/online_property/dqn_lta_aux/successor_as/best/"
     # },
    # {"label": "Random",
     # "control": "data/output/test/gridhard/control/baseline/same_task/fix_rep/random/best/",
     # "online_measure": "data/output/test/gridhard/fixrep_property/random/best",
     # },
    # {"label": "Input",
     # "control": "data/output/test/gridhard/control/baseline/same_task/fix_rep/input/best/",
     # "online_measure": "data/output/test/gridhard/fixrep_property/input/best",
     # },
]

crgb_diff_last = [
    {"label": "ReLU",
    "control": "data/output/test/picky_eater/control/different_task/fix_rep/dqn/best_final/",
     "online_measure": "data/output/test/picky_eater/online_property/dqn/best/"
     },
#     {"label": "ReLU+Control1g",
     # "control": "data/output/test/gridhard/control/last/different_task/fix_rep/dqn_aux/aux_control/best_1g/",
     # "online_measure": "data/output/test/gridhard/online_property/dqn_aux/freq8/aux_control/best_1g/"
     # },
    # {"label": "ReLU+Control5g",
     # "control": "data/output/test/gridhard/control/last/different_task/fix_rep/dqn_aux/aux_control/best_5g/",
     # "online_measure": "data/output/test/gridhard/online_property/dqn_aux/freq128/aux_control/best_5g/"
     # },
    # {"label": "ReLU+XY",
     # "control": "data/output/test/gridhard/control/last/different_task/fix_rep/dqn_aux/info/best/",
     # "online_measure": "data/output/test/gridhard/online_property/dqn_aux/freq8/info/best/"
     # },
    # {"label": "ReLU+Decoder",
     # "control": "data/output/test/gridhard/control/last/different_task/fix_rep/dqn_aux/input_decoder/best/",
     # "online_measure": "data/output/test/gridhard/online_property/dqn_aux/freq8/info/best/"
     # },
    # {"label": "ReLU+NAS",
     # "control": "data/output/test/gridhard/control/last/different_task/fix_rep/dqn_aux/nas_v2_delta/best/",
     # "online_measure": "data/output/test/gridhard/online_property/dqn_aux/freq8/nas_v2_delta/best/"
     # },
    # {"label": "ReLU+Reward",
     # "control": "data/output/test/gridhard/control/last/different_task/fix_rep/dqn_aux/reward/best/",
     # "online_measure": "data/output/test/gridhard/online_property/dqn_aux/freq8/reward/best/"
     # },
    # {"label": "ReLU+SF",
     # "control": "data/output/test/gridhard/control/last/different_task/fix_rep/dqn_aux/successor_as/best/",
     # "online_measure": "data/output/test/gridhard/online_property/dqn_aux/freq8/successor_as/best/"
#      },
    {"label": "LTA",
     "control": "data/output/test/picky_eater/control/different_task/fix_rep/dqn_lta/best_final/",
     "online_measure": "data/output/test/picky_eater/online_property/dqn_lta/best/"
     },
#     {"label": "LTA+Control1g",
     # "control": "data/output/test/gridhard/control/last/different_task/fix_rep/dqn_lta_aux/aux_control/best_1g/",
     # "online_measure": "data/output/test/gridhard/online_property/dqn_lta_aux/aux_control/best_1g/"
     # },
    # {"label": "LTA+Control5g",
     # "control": "data/output/test/gridhard/control/last/different_task/fix_rep/dqn_lta_aux/aux_control/best_5g/",
     # "online_measure": "data/output/test/gridhard/online_property/dqn_lta_aux/aux_control/best_5g/"
     # },
    # {"label": "LTA+XY",
     # "control": "data/output/test/gridhard/control/last/different_task/fix_rep/dqn_lta_aux/info/best/",
     # "online_measure": "data/output/test/gridhard/online_property/dqn_lta_aux/info/best/"
     # },
    # {"label": "LTA+Decoder",
     # "control": "data/output/test/gridhard/control/last/different_task/fix_rep/dqn_lta_aux/input_decoder/best/",
     # "online_measure": "data/output/test/gridhard/online_property/dqn_lta_aux/input_decoder/best/"
     # },
    # {"label": "LTA+NAS",
     # "control": "data/output/test/gridhard/control/last/different_task/fix_rep/dqn_lta_aux/nas_v2_delta/best/",
     # "online_measure": "data/output/test/gridhard/online_property/dqn_lta_aux/nas_v2_delta/best/"
     # },
    # {"label": "LTA+Reward",
     # "control": "data/output/test/gridhard/control/last/different_task/fix_rep/dqn_lta_aux/reward/best/",
     # "online_measure": "data/output/test/gridhard/online_property/dqn_lta_aux/reward/best/"
     # },
    # {"label": "LTA+SF",
     # "control": "data/output/test/gridhard/control/last/different_task/fix_rep/dqn_lta_aux/successor_as/best/",
     # "online_measure": "data/output/test/gridhard/online_property/dqn_lta_aux/successor_as/best/"
     # },
    # {"label": "Random",
     # "control": "data/output/test/gridhard/control/baseline/different_task/fix_rep/random/best/",
     # "online_measure": "data/output/test/gridhard/fixrep_property/random/best",
     # },
    # {"label": "Input",
     # "control": "data/output/test/gridhard/control/baseline/different_task/fix_rep/input/best/",
     # "online_measure": "data/output/test/gridhard/fixrep_property/input/best",
#      },
]

crgb_diff_tune_last = [
    {"label": "ReLU", "property": "data/output/test/",
     "control": "data/output/test/picky_eater/control/different_task/fine_tune/dqn/best_final",
     "online_measure": "data/output/test/picky_eater/online_property/dqn/best/"
     },
#     {"label": "ReLU+Control",
     # "control": "data/output/test/gridhard/control/last/different_task/fine_tune/dqn_aux/aux_control/best_5g/",
     # "online_measure": "data/output/test/gridhard/online_property/dqn_aux/freq128/aux_control/best_5g/"
     # },
    # {"label": "ReLU+XY",
     # "control": "data/output/test/gridhard/control/last/different_task/fine_tune/dqn_aux/info/best/",
     # "online_measure": "data/output/test/gridhard/online_property/dqn_aux/freq8/info/best/"
     # },
    # {"label": "ReLU+Decoder",
     # "control": "data/output/test/gridhard/control/last/different_task/fine_tune/dqn_aux/input_decoder/best/",
     # "online_measure": "data/output/test/gridhard/online_property/dqn_aux/freq8/info/best/"
     # },
    # {"label": "ReLU+NAS",
     # "control": "data/output/test/gridhard/control/last/different_task/fine_tune/dqn_aux/nas_v2_delta/best/",
     # "online_measure": "data/output/test/gridhard/online_property/dqn_aux/freq8/nas_v2_delta/best/"
     # },
    # {"label": "ReLU+Reward",
     # "control": "data/output/test/gridhard/control/last/different_task/fine_tune/dqn_aux/reward/best/",
     # "online_measure": "data/output/test/gridhard/online_property/dqn_aux/freq8/reward/best/"
     # },
    # {"label": "ReLU+SF",
     # "control": "data/output/test/gridhard/control/last/different_task/fine_tune/dqn_aux/successor_as/best/",
     # "online_measure": "data/output/test/gridhard/online_property/dqn_aux/freq8/successor_as/best/"
#      },
    {"label": "LTA",
     "control": "data/output/test/picky_eater/control/different_task/fine_tune/dqn_lta/best_final",
     "online_measure": "data/output/test/picky_eater/online_property/dqn_lta/best/"
     },
#     {"label": "LTA+Control",
     # "control": "data/output/test/gridhard/control/last/different_task/fine_tune/dqn_lta_aux/aux_control/best_1g/",
     # "online_measure": "data/output/test/gridhard/online_property/dqn_lta_aux/aux_control/best_1g/"
     # },
    # {"label": "LTA+XY",
     # "control": "data/output/test/gridhard/control/last/different_task/fine_tune/dqn_lta_aux/info/best/",
     # "online_measure": "data/output/test/gridhard/online_property/dqn_lta_aux/info/best/"
     # },
    # {"label": "LTA+Decoder",
     # "control": "data/output/test/gridhard/control/last/different_task/fine_tune/dqn_lta_aux/input_decoder/best/",
     # "online_measure": "data/output/test/gridhard/online_property/dqn_lta_aux/input_decoder/best/"
     # },
    # {"label": "LTA+NAS",
     # "control": "data/output/test/gridhard/control/last/different_task/fine_tune/dqn_lta_aux/nas_v2_delta/best/",
     # "online_measure": "data/output/test/gridhard/online_property/dqn_lta_aux/nas_v2_delta/best/"
     # },
    # {"label": "LTA+Reward",
     # "control": "data/output/test/gridhard/control/last/different_task/fine_tune/dqn_lta_aux/reward/best/",
     # "online_measure": "data/output/test/gridhard/online_property/dqn_lta_aux/reward/best/"
     # },
    # {"label": "LTA+SF",
     # "control": "data/output/test/gridhard/control/last/different_task/fine_tune/dqn_lta_aux/successor_as/best/",
     # "online_measure": "data/output/test/gridhard/online_property/dqn_lta_aux/successor_as/best/"
     # },
    # {"label": "Random",
     # "control": "data/output/test/gridhard/control/baseline/different_task/fine_tune/random/best/",
     # "online_measure": "data/output/test/gridhard/fixrep_property/random/best",
#      },
]


crgb_similar_last = [
    {"label": "ReLU", "property": "data/output/test/",
     "control": "data/output/test/gridhard/control/similar_task/fix_rep/dqn/best_final/",
     "online_measure": "data/output/test/gridhard/online_property/dqn/best/"
     },
#     {"label": "ReLU+Control1g",
     # "control": "data/output/test/gridhard/control/last/similar_task/fix_rep/dqn_aux/aux_control/best_1g/",
     # "online_measure": "data/output/test/gridhard/online_property/dqn_aux/freq8/aux_control/best_1g/"
     # },
    # {"label": "ReLU+Control5g",
     # "control": "data/output/test/gridhard/control/last/similar_task/fix_rep/dqn_aux/aux_control/best_5g/",
     # "online_measure": "data/output/test/gridhard/online_property/dqn_aux/freq128/aux_control/best_5g/"
     # },
    # {"label": "ReLU+XY",
     # "control": "data/output/test/gridhard/control/last/similar_task/fix_rep/dqn_aux/info/best/",
     # "online_measure": "data/output/test/gridhard/online_property/dqn_aux/freq8/info/best/"
     # },
    # {"label": "ReLU+Decoder",
     # "control": "data/output/test/gridhard/control/last/similar_task/fix_rep/dqn_aux/input_decoder/best/",
     # "online_measure": "data/output/test/gridhard/online_property/dqn_aux/freq8/info/best/"
     # },
    # {"label": "ReLU+NAS",
     # "control": "data/output/test/gridhard/control/last/similar_task/fix_rep/dqn_aux/nas_v2_delta/best/",
     # "online_measure": "data/output/test/gridhard/online_property/dqn_aux/freq8/nas_v2_delta/best/"
     # },
    # {"label": "ReLU+Reward",
     # "control": "data/output/test/gridhard/control/last/similar_task/fix_rep/dqn_aux/reward/best/",
     # "online_measure": "data/output/test/gridhard/online_property/dqn_aux/freq8/reward/best/"
     # },
    # {"label": "ReLU+SF",
     # "control": "data/output/test/gridhard/control/last/similar_task/fix_rep/dqn_aux/successor_as/best/",
     # "online_measure": "data/output/test/gridhard/online_property/dqn_aux/freq8/successor_as/best/"
     # },
    {"label": "LTA",
      "control": "data/output/test/picky_eater/control/similar_task/fix_rep/dqn_lta/best_final/",
      "online_measure": "data/output/test/picky_eater/online_property/dqn_lta/best/"
      },
    # {"label": "LTA+Control1g",
     # "control": "data/output/test/gridhard/control/last/similar_task/fix_rep/dqn_lta_aux/aux_control/best_1g/",
     # "online_measure": "data/output/test/gridhard/online_property/dqn_lta_aux/aux_control/best_1g/"
     # },
    # {"label": "LTA+Control5g",
     # "control": "data/output/test/gridhard/control/last/similar_task/fix_rep/dqn_lta_aux/aux_control/best_5g/",
     # "online_measure": "data/output/test/gridhard/online_property/dqn_lta_aux/aux_control/best_5g/"
     # },
    # {"label": "LTA+XY",
     # "control": "data/output/test/gridhard/control/last/similar_task/fix_rep/dqn_lta_aux/info/best/",
     # "online_measure": "data/output/test/gridhard/online_property/dqn_lta_aux/info/best/"
     # },
    # {"label": "LTA+Decoder",
     # "control": "data/output/test/gridhard/control/last/similar_task/fix_rep/dqn_lta_aux/input_decoder/best/",
     # "online_measure": "data/output/test/gridhard/online_property/dqn_lta_aux/input_decoder/best/"
     # },
    # {"label": "LTA+NAS",
     # "control": "data/output/test/gridhard/control/last/similar_task/fix_rep/dqn_lta_aux/nas_v2_delta/best/",
     # "online_measure": "data/output/test/gridhard/online_property/dqn_lta_aux/nas_v2_delta/best/"
     # },
    # {"label": "LTA+Reward",
     # "control": "data/output/test/gridhard/control/last/similar_task/fix_rep/dqn_lta_aux/reward/best/",
     # "online_measure": "data/output/test/gridhard/online_property/dqn_lta_aux/reward/best/"
     # },
    # {"label": "LTA+SF",
     # "control": "data/output/test/gridhard/control/last/similar_task/fix_rep/dqn_lta_aux/successor_as/best/",
     # "online_measure": "data/output/test/gridhard/online_property/dqn_lta_aux/successor_as/best/"
     # },
    # {"label": "Random",
     # "control": "data/output/test/gridhard/control/baseline/similar_task/fix_rep/random/best/",
     # "online_measure": "data/output/test/gridhard/fixrep_property/random/best",
     # },
    # {"label": "Input",
     # "control": "data/output/test/gridhard/control/baseline/similar_task/fix_rep/input/best/",
     # "online_measure": "data/output/test/gridhard/fixrep_property/input/best",
     # },
]


crgb_same_early_sweep = [
    {"label": "ReLU", "property": "data/output/test/",
     "control": "data/output/test/picky_eater/control/early_stop/same_task/fix_rep/dqn/sweep/",
     "online_measure": "data/output/test/picky_eater/online_property/dqn/sweep/",
     "best": '1'
     },
    {"label": "ReLU+Control",
     "control": "data/output/test/picky_eater/control/early_stop/same_task/fix_rep/dqn_aux/aux_control/sweep/",
     "online_measure": "data/output/test/picky_eater/online_property/dqn_aux/aux_control/sweep/",
     "best": '0'
     },
    {"label": "ReLU+XY",
     "control": "data/output/test/picky_eater/control/early_stop/same_task/fix_rep/dqn_aux/info/sweep_xy/",
     "online_measure": "data/output/test/picky_eater/online_property/dqn_aux/info/sweep_xy/",
     "best": '0'
     },
    {"label": "ReLU+Decoder",
     "control": "data/output/test/picky_eater/control/early_stop/same_task/fix_rep/dqn_aux/input_decoder/sweep/",
     "online_measure": "data/output/test/picky_eater/online_property/dqn_aux/input_decoder/sweep/",
     "best": '1'
     },
    {"label": "ReLU+NAS",
     "control": "data/output/test/picky_eater/control/early_stop/same_task/fix_rep/dqn_aux/nas_v2_delta/sweep/",
     "online_measure": "data/output/test/picky_eater/online_property/dqn_aux/nas_v2_delta/sweep/",
     "best": '0'
     },
    {"label": "ReLU+Reward",
     "control": "data/output/test/picky_eater/control/early_stop/same_task/fix_rep/dqn_aux/reward/sweep/",
     "online_measure": "data/output/test/picky_eater/online_property/dqn_aux/reward/sweep/",
     "best": '0'
     },
    {"label": "ReLU+SF",
     "control": "data/output/test/picky_eater/control/early_stop/same_task/fix_rep/dqn_aux/successor_as/sweep/",
     "online_measure": "data/output/test/picky_eater/online_property/dqn_aux/successor_as/sweep/",
     "best": '0'
     },
    {"label": "FTA eta=2",
     "control": "data/output/test/picky_eater/control/early_stop/same_task/fix_rep/dqn_lta/sweep/",
     "online_measure": "data/output/test/picky_eater/online_property/dqn_lta/sweep/",
     "best": '1'
     },
    {"label": "FTA+Control",
     "control": "data/output/test/picky_eater/control/early_stop/same_task/fix_rep/dqn_lta_aux/aux_control/sweep/",
     "online_measure": "data/output/test/picky_eater/online_property/dqn_lta_aux/aux_control/sweep/",
     "best": '0'

     },
    {"label": "FTA+XY",
     "control": "data/output/test/picky_eater/control/early_stop/same_task/fix_rep/dqn_lta_aux/info/sweep_xy/",
     "online_measure": "data/output/test/picky_eater/online_property/dqn_lta_aux/info/sweep_xy/",
     "best": '0'
     },
    {"label": "FTA+Decoder",
     "control": "data/output/test/picky_eater/control/early_stop/same_task/fix_rep/dqn_lta_aux/input_decoder/sweep/",
     "online_measure": "data/output/test/picky_eater/online_property/dqn_lta_aux/input_decoder/sweep/",
     "best": '0'
     },
    {"label": "FTA+NAS",
     "control": "data/output/test/picky_eater/control/early_stop/same_task/fix_rep/dqn_lta_aux/nas_v2_delta/sweep/",
     "online_measure": "data/output/test/picky_eater/online_property/dqn_lta_aux/nas_v2_delta/sweep/",
     "best": '0'
     },
    {"label": "FTA+Reward",
     "control": "data/output/test/picky_eater/control/early_stop/same_task/fix_rep/dqn_lta_aux/reward/sweep/",
     "online_measure": "data/output/test/picky_eater/online_property/dqn_lta_aux/reward/sweep/",
     "best": '0'
     },
    {"label": "FTA+SF",
     "control": "data/output/test/picky_eater/control/early_stop/same_task/fix_rep/dqn_lta_aux/successor_as/sweep/",
     "online_measure": "data/output/test/picky_eater/online_property/dqn_lta_aux/successor_as/sweep/",
     "best": '0'
     },
    # {"label": "Random",
     # "control": "data/output/test/picky_eater/control/baseline/same_task/fix_rep/random/sweep/",
     # "online_measure": "data/output/test/picky_eater/fixrep_property/random/sweep",
    #  },
]


crgb_same_early = [
    {"label": "ReLU", "property": "data/output/test/",
     "control": "data/output/test/picky_eater/control/early_stop/same_task/fix_rep/dqn/best/",
     "online_measure": "data/output/test/picky_eater/online_property/dqn/best/",
     },
    {"label": "ReLU+Control",
     "control": "data/output/test/picky_eater/control/early_stop/same_task/fix_rep/dqn_aux/aux_control/best/",
     "online_measure": "data/output/test/picky_eater/online_property/dqn_aux/aux_control/best/",
     },
    {"label": "ReLU+XY",
     "control": "data/output/test/picky_eater/control/early_stop/same_task/fix_rep/dqn_aux/info/best_xy/",
     "online_measure": "data/output/test/picky_eater/online_property/dqn_aux/info/best_xy/",
     },
    {"label": "ReLU+Decoder",
     "control": "data/output/test/picky_eater/control/early_stop/same_task/fix_rep/dqn_aux/input_decoder/best/",
     "online_measure": "data/output/test/picky_eater/online_property/dqn_aux/input_decoder/best/",
     },
    {"label": "ReLU+NAS",
     "control": "data/output/test/picky_eater/control/early_stop/same_task/fix_rep/dqn_aux/nas_v2_delta/best/",
     "online_measure": "data/output/test/picky_eater/online_property/dqn_aux/nas_v2_delta/best/"
     },
    {"label": "ReLU+Reward",
     "control": "data/output/test/picky_eater/control/early_stop/same_task/fix_rep/dqn_aux/reward/best/",
     "online_measure": "data/output/test/picky_eater/online_property/dqn_aux/reward/best/"
     },
    {"label": "ReLU+SF",
     "control": "data/output/test/picky_eater/control/early_stop/same_task/fix_rep/dqn_aux/successor_as/best/",
     "online_measure": "data/output/test/picky_eater/online_property/dqn_aux/successor_as/best/"
     },
    {"label": "FTA eta=2",
     "control": "data/output/test/picky_eater/control/early_stop/same_task/fix_rep/dqn_lta/best/",
     "online_measure": "data/output/test/picky_eater/online_property/dqn_lta/best/"
     },
    {"label": "FTA+Control",
     "control": "data/output/test/picky_eater/control/early_stop/same_task/fix_rep/dqn_lta_aux/aux_control/best/",
     "online_measure": "data/output/test/picky_eater/online_property/dqn_lta_aux/aux_control/best/"
     },
    {"label": "FTA+XY",
     "control": "data/output/test/picky_eater/control/early_stop/same_task/fix_rep/dqn_lta_aux/info/best_xy/",
     "online_measure": "data/output/test/picky_eater/online_property/dqn_lta_aux/info/best_xy/"
     },
    {"label": "FTA+Decoder",
     "control": "data/output/test/picky_eater/control/early_stop/same_task/fix_rep/dqn_lta_aux/input_decoder/best/",
     "online_measure": "data/output/test/picky_eater/online_property/dqn_lta_aux/input_decoder/best/"
     },
    {"label": "FTA+NAS",
     "control": "data/output/test/picky_eater/control/early_stop/same_task/fix_rep/dqn_lta_aux/nas_v2_delta/best/",
     "online_measure": "data/output/test/picky_eater/online_property/dqn_lta_aux/nas_v2_delta/best/"
     },
    {"label": "FTA+Reward",
     "control": "data/output/test/picky_eater/control/early_stop/same_task/fix_rep/dqn_lta_aux/reward/best/",
     "online_measure": "data/output/test/picky_eater/online_property/dqn_lta_aux/reward/best/"
     },
    {"label": "FTA+SF",
     "control": "data/output/test/picky_eater/control/early_stop/same_task/fix_rep/dqn_lta_aux/successor_as/best/",
     "online_measure": "data/output/test/picky_eater/online_property/dqn_lta_aux/successor_as/best/"
     },
    # {"label": "Random",
     # "control": "data/output/test/picky_eater/control/baseline/same_task/fix_rep/random/best/",
     # "online_measure": "data/output/test/picky_eater/fixrep_property/random/best",
    #  },
]

crgb_diff_early = [
    {"label": "ReLU",
    "control": "data/output/test/picky_eater/control/different_task/fix_rep/dqn/best_early/",
     "online_measure": "data/output/test/picky_eater/online_property/dqn/best/"
     },
#     {"label": "ReLU+Control1g",
     # "control": "data/output/test/gridhard/control/last/different_task/fix_rep/dqn_aux/aux_control/best_1g/",
     # "online_measure": "data/output/test/gridhard/online_property/dqn_aux/freq8/aux_control/best_1g/"
     # },
    # {"label": "ReLU+Control5g",
     # "control": "data/output/test/gridhard/control/last/different_task/fix_rep/dqn_aux/aux_control/best_5g/",
     # "online_measure": "data/output/test/gridhard/online_property/dqn_aux/freq128/aux_control/best_5g/"
     # },
    # {"label": "ReLU+XY",
     # "control": "data/output/test/gridhard/control/last/different_task/fix_rep/dqn_aux/info/best/",
     # "online_measure": "data/output/test/gridhard/online_property/dqn_aux/freq8/info/best/"
     # },
    # {"label": "ReLU+Decoder",
     # "control": "data/output/test/gridhard/control/last/different_task/fix_rep/dqn_aux/input_decoder/best/",
     # "online_measure": "data/output/test/gridhard/online_property/dqn_aux/freq8/info/best/"
     # },
    # {"label": "ReLU+NAS",
     # "control": "data/output/test/gridhard/control/last/different_task/fix_rep/dqn_aux/nas_v2_delta/best/",
     # "online_measure": "data/output/test/gridhard/online_property/dqn_aux/freq8/nas_v2_delta/best/"
     # },
    # {"label": "ReLU+Reward",
     # "control": "data/output/test/gridhard/control/last/different_task/fix_rep/dqn_aux/reward/best/",
     # "online_measure": "data/output/test/gridhard/online_property/dqn_aux/freq8/reward/best/"
     # },
    # {"label": "ReLU+SF",
     # "control": "data/output/test/gridhard/control/last/different_task/fix_rep/dqn_aux/successor_as/best/",
     # "online_measure": "data/output/test/gridhard/online_property/dqn_aux/freq8/successor_as/best/"
#      },
    {"label": "LTA",
     "control": "data/output/test/picky_eater/control/different_task/fix_rep/dqn_lta/best_early/",
     "online_measure": "data/output/test/picky_eater/online_property/dqn_lta/best/"
     },
#     {"label": "LTA+Control1g",
     # "control": "data/output/test/gridhard/control/last/different_task/fix_rep/dqn_lta_aux/aux_control/best_1g/",
     # "online_measure": "data/output/test/gridhard/online_property/dqn_lta_aux/aux_control/best_1g/"
     # },
    # {"label": "LTA+Control5g",
     # "control": "data/output/test/gridhard/control/last/different_task/fix_rep/dqn_lta_aux/aux_control/best_5g/",
     # "online_measure": "data/output/test/gridhard/online_property/dqn_lta_aux/aux_control/best_5g/"
     # },
    # {"label": "LTA+XY",
     # "control": "data/output/test/gridhard/control/last/different_task/fix_rep/dqn_lta_aux/info/best/",
     # "online_measure": "data/output/test/gridhard/online_property/dqn_lta_aux/info/best/"
     # },
    # {"label": "LTA+Decoder",
     # "control": "data/output/test/gridhard/control/last/different_task/fix_rep/dqn_lta_aux/input_decoder/best/",
     # "online_measure": "data/output/test/gridhard/online_property/dqn_lta_aux/input_decoder/best/"
     # },
    # {"label": "LTA+NAS",
     # "control": "data/output/test/gridhard/control/last/different_task/fix_rep/dqn_lta_aux/nas_v2_delta/best/",
     # "online_measure": "data/output/test/gridhard/online_property/dqn_lta_aux/nas_v2_delta/best/"
     # },
    # {"label": "LTA+Reward",
     # "control": "data/output/test/gridhard/control/last/different_task/fix_rep/dqn_lta_aux/reward/best/",
     # "online_measure": "data/output/test/gridhard/online_property/dqn_lta_aux/reward/best/"
     # },
    # {"label": "LTA+SF",
     # "control": "data/output/test/gridhard/control/last/different_task/fix_rep/dqn_lta_aux/successor_as/best/",
     # "online_measure": "data/output/test/gridhard/online_property/dqn_lta_aux/successor_as/best/"
     # },
    # {"label": "Random",
     # "control": "data/output/test/gridhard/control/baseline/different_task/fix_rep/random/best/",
     # "online_measure": "data/output/test/gridhard/fixrep_property/random/best",
     # },
    # {"label": "Input",
     # "control": "data/output/test/gridhard/control/baseline/different_task/fix_rep/input/best/",
     # "online_measure": "data/output/test/gridhard/fixrep_property/input/best",
#      },
]

crgb_diff_tune_early_sweep = [
    {"label": "ReLU", "property": "data/output/test/",
     "control": "data/output/test/picky_eater/control/early_stop/different_task/fine_tune/dqn/sweep/",
     "online_measure": "data/output/test/picky_eater/online_property/dqn/sweep/",
     "best": '2',
     },
    {"label": "ReLU+Control",
     "control": "data/output/test/picky_eater/control/early_stop/different_task/fine_tune/dqn_aux/aux_control/sweep/",
     "online_measure": "data/output/test/picky_eater/online_property/dqn_aux/aux_control/sweep/",
     "best": '2',
     },
    {"label": "ReLU+XY",
     "control": "data/output/test/picky_eater/control/early_stop/different_task/fine_tune/dqn_aux/info/sweep_xy/",
     "online_measure": "data/output/test/picky_eater/online_property/dqn_aux/info/sweep_xy/",
     "best": '2',
     },
    {"label": "ReLU+Decoder",
     "control": "data/output/test/picky_eater/control/early_stop/different_task/fine_tune/dqn_aux/input_decoder/sweep/",
     "online_measure": "data/output/test/picky_eater/online_property/dqn_aux/input_decoder/sweep/",
     "best": '1',
     },
    {"label": "ReLU+NAS",
     "control": "data/output/test/picky_eater/control/early_stop/different_task/fine_tune/dqn_aux/nas_v2_delta/sweep/",
     "online_measure": "data/output/test/picky_eater/online_property/dqn_aux/nas_v2_delta/sweep/",
     "best": '1',
     },
    {"label": "ReLU+Reward",
     "control": "data/output/test/picky_eater/control/early_stop/different_task/fine_tune/dqn_aux/reward/sweep/",
     "online_measure": "data/output/test/picky_eater/online_property/dqn_aux/reward/sweep/",
     "best": '2',
     },
    {"label": "ReLU+SF",
     "control": "data/output/test/picky_eater/control/early_stop/different_task/fine_tune/dqn_aux/successor_as/sweep/",
     "online_measure": "data/output/test/picky_eater/online_property/dqn_aux/successor_as/sweep/",
     "best": '2',
     },
    {"label": "FTA eta=2",
     "control": "data/output/test/picky_eater/control/early_stop/different_task/fine_tune/dqn_lta/sweep/",
     "online_measure": "data/output/test/picky_eater/online_property/dqn_lta/sweep/",
    "best": '2',
     },
    {"label": "FTA+Control",
     "control": "data/output/test/picky_eater/control/early_stop/different_task/fine_tune/dqn_lta_aux/aux_control/sweep/",
     "online_measure": "data/output/test/picky_eater/online_property/dqn_lta_aux/aux_control/sweep/",
     "best": '3',
     },
    {"label": "FTA+XY",
     "control": "data/output/test/picky_eater/control/early_stop/different_task/fine_tune/dqn_lta_aux/info/sweep_xy/",
     "online_measure": "data/output/test/picky_eater/online_property/dqn_lta_aux/info/sweep_xy/",
     "best": '1',
     },
    {"label": "FTA+Decoder",
     "control": "data/output/test/picky_eater/control/early_stop/different_task/fine_tune/dqn_lta_aux/input_decoder/sweep/",
     "online_measure": "data/output/test/picky_eater/online_property/dqn_lta_aux/input_decoder/sweep/",
     "best": '2',
     },
    {"label": "FTA+NAS",
     "control": "data/output/test/picky_eater/control/early_stop/different_task/fine_tune/dqn_lta_aux/nas_v2_delta/sweep/",
     "online_measure": "data/output/test/picky_eater/online_property/dqn_lta_aux/nas_v2_delta/sweep/",
     "best": '2',
     },
    {"label": "FTA+Reward",
     "control": "data/output/test/picky_eater/control/early_stop/different_task/fine_tune/dqn_lta_aux/reward/sweep/",
     "online_measure": "data/output/test/picky_eater/online_property/dqn_lta_aux/reward/sweep/",
     "best": '2',
     },
    {"label": "FTA+SF",
     "control": "data/output/test/picky_eater/control/early_stop/different_task/fine_tune/dqn_lta_aux/successor_as/sweep/",
     "online_measure": "data/output/test/picky_eater/online_property/dqn_lta_aux/successor_as/sweep/",
     "best": '2',
     },
    # {"label": "Random",
     # "control": "data/output/test/picky_eater/control/baseline/different_task/fine_tune/random/sweep/",
     # "online_measure": "data/output/test/picky_eater/fixrep_property/random/sweep",
    #  },
]

crgb_diff_fix_early_sweep = [
    {"label": "ReLU", "property": "data/output/test/",
     "control": "data/output/test/picky_eater/control/early_stop/different_task/fix_rep/dqn/sweep/",
     "online_measure": "data/output/test/picky_eater/online_property/dqn/sweep/",
     'best': '1'
     },
    {"label": "ReLU+Control",
     "control": "data/output/test/picky_eater/control/early_stop/different_task/fix_rep/dqn_aux/aux_control/sweep/",
     "online_measure": "data/output/test/picky_eater/online_property/dqn_aux/aux_control/sweep/",
     'best': '0'
     },
    {"label": "ReLU+XY",
     "control": "data/output/test/picky_eater/control/early_stop/different_task/fix_rep/dqn_aux/info/sweep_xy/",
     "online_measure": "data/output/test/picky_eater/online_property/dqn_aux/info/sweep_xy/",
     'best': '0'
     },
    {"label": "ReLU+Decoder",
     "control": "data/output/test/picky_eater/control/early_stop/different_task/fix_rep/dqn_aux/input_decoder/sweep/",
     "online_measure": "data/output/test/picky_eater/online_property/dqn_aux/input_decoder/sweep/",
     'best': '2'
     },
    {"label": "ReLU+NAS",
     "control": "data/output/test/picky_eater/control/early_stop/different_task/fix_rep/dqn_aux/nas_v2_delta/sweep/",
     "online_measure": "data/output/test/picky_eater/online_property/dqn_aux/nas_v2_delta/sweep/",
     'best': '1'
     },
    {"label": "ReLU+Reward",
     "control": "data/output/test/picky_eater/control/early_stop/different_task/fix_rep/dqn_aux/reward/sweep/",
     "online_measure": "data/output/test/picky_eater/online_property/dqn_aux/reward/sweep/",
    'best': '0'
     },
    {"label": "ReLU+SF",
     "control": "data/output/test/picky_eater/control/early_stop/different_task/fix_rep/dqn_aux/successor_as/sweep/",
     "online_measure": "data/output/test/picky_eater/online_property/dqn_aux/successor_as/sweep/",
     'best': '0'
     },
    {"label": "FTA eta=2",
     "control": "data/output/test/picky_eater/control/early_stop/different_task/fix_rep/dqn_lta/sweep/",
     "online_measure": "data/output/test/picky_eater/online_property/dqn_lta/sweep/",
     'best': '0'
     },
    {"label": "FTA+Control",
     "control": "data/output/test/picky_eater/control/early_stop/different_task/fix_rep/dqn_lta_aux/aux_control/sweep/",
     "online_measure": "data/output/test/picky_eater/online_property/dqn_lta_aux/aux_control/sweep/",
     'best': '0'
     },
    {"label": "FTA+XY",
     "control": "data/output/test/picky_eater/control/early_stop/different_task/fix_rep/dqn_lta_aux/info/sweep_xy/",
     "online_measure": "data/output/test/picky_eater/online_property/dqn_lta_aux/info/sweep_xy/",
     'best': '0'
     },
    {"label": "FTA+Decoder",
     "control": "data/output/test/picky_eater/control/early_stop/different_task/fix_rep/dqn_lta_aux/input_decoder/sweep/",
     "online_measure": "data/output/test/picky_eater/online_property/dqn_lta_aux/input_decoder/sweep/",
     'best': '3'
     },
    {"label": "FTA+NAS",
     "control": "data/output/test/picky_eater/control/early_stop/different_task/fix_rep/dqn_lta_aux/nas_v2_delta/sweep/",
     "online_measure": "data/output/test/picky_eater/online_property/dqn_lta_aux/nas_v2_delta/sweep/",
     'best': '0'
     },
    {"label": "FTA+Reward",
     "control": "data/output/test/picky_eater/control/early_stop/different_task/fix_rep/dqn_lta_aux/reward/sweep/",
     "online_measure": "data/output/test/picky_eater/online_property/dqn_lta_aux/reward/sweep/",
     'best': '0'
     },
    {"label": "FTA+SF",
     "control": "data/output/test/picky_eater/control/early_stop/different_task/fix_rep/dqn_lta_aux/successor_as/sweep/",
     "online_measure": "data/output/test/picky_eater/online_property/dqn_lta_aux/successor_as/sweep/",
     'best': '0'
     },
    # {"label": "Random",
     # "control": "data/output/test/picky_eater/control/baseline/different_task/fix_rep/random/sweep/",
     # "online_measure": "data/output/test/picky_eater/fixrep_property/random/sweep",
    #  },
]



crgb_diff_fix_early = [
    {"label": "ReLU", "property": "data/output/test/",
     "control": "data/output/test/picky_eater/control/early_stop/different_task/fix_rep/dqn/best/",
     "online_measure": "data/output/test/picky_eater/online_property/dqn/best/",
     },
    {"label": "ReLU+Control",
     "control": "data/output/test/picky_eater/control/early_stop/different_task/fix_rep/dqn_aux/aux_control/best/",
     "online_measure": "data/output/test/picky_eater/online_property/dqn_aux/aux_control/best/",
     },
    {"label": "ReLU+XY",
     "control": "data/output/test/picky_eater/control/early_stop/different_task/fix_rep/dqn_aux/info/best_xy/",
     "online_measure": "data/output/test/picky_eater/online_property/dqn_aux/info/best_xy/",
     },
    {"label": "ReLU+Decoder",
     "control": "data/output/test/picky_eater/control/early_stop/different_task/fix_rep/dqn_aux/input_decoder/best/",
     "online_measure": "data/output/test/picky_eater/online_property/dqn_aux/input_decoder/best/",
     },
    {"label": "ReLU+NAS",
     "control": "data/output/test/picky_eater/control/early_stop/different_task/fix_rep/dqn_aux/nas_v2_delta/best/",
     "online_measure": "data/output/test/picky_eater/online_property/dqn_aux/nas_v2_delta/best/"
     },
    {"label": "ReLU+Reward",
     "control": "data/output/test/picky_eater/control/early_stop/different_task/fix_rep/dqn_aux/reward/best/",
     "online_measure": "data/output/test/picky_eater/online_property/dqn_aux/reward/best/"
     },
    {"label": "ReLU+SF",
     "control": "data/output/test/picky_eater/control/early_stop/different_task/fix_rep/dqn_aux/successor_as/best/",
     "online_measure": "data/output/test/picky_eater/online_property/dqn_aux/successor_as/best/"
     },
    {"label": "FTA eta=2",
     "control": "data/output/test/picky_eater/control/early_stop/different_task/fix_rep/dqn_lta/best/",
     "online_measure": "data/output/test/picky_eater/online_property/dqn_lta/best/"
     },
    {"label": "FTA+Control",
     "control": "data/output/test/picky_eater/control/early_stop/different_task/fix_rep/dqn_lta_aux/aux_control/best/",
     "online_measure": "data/output/test/picky_eater/online_property/dqn_lta_aux/aux_control/best/"
     },
    {"label": "FTA+XY",
     "control": "data/output/test/picky_eater/control/early_stop/different_task/fix_rep/dqn_lta_aux/info/best_xy/",
     "online_measure": "data/output/test/picky_eater/online_property/dqn_lta_aux/info/best_xy/"
     },
    {"label": "FTA+Decoder",
     "control": "data/output/test/picky_eater/control/early_stop/different_task/fix_rep/dqn_lta_aux/input_decoder/best/",
     "online_measure": "data/output/test/picky_eater/online_property/dqn_lta_aux/input_decoder/best/"
     },
    {"label": "FTA+NAS",
     "control": "data/output/test/picky_eater/control/early_stop/different_task/fix_rep/dqn_lta_aux/nas_v2_delta/best/",
     "online_measure": "data/output/test/picky_eater/online_property/dqn_lta_aux/nas_v2_delta/best/"
     },
    {"label": "FTA+Reward",
     "control": "data/output/test/picky_eater/control/early_stop/different_task/fix_rep/dqn_lta_aux/reward/best/",
     "online_measure": "data/output/test/picky_eater/online_property/dqn_lta_aux/reward/best/"
     },
    {"label": "FTA+SF",
     "control": "data/output/test/picky_eater/control/early_stop/different_task/fix_rep/dqn_lta_aux/successor_as/best/",
     "online_measure": "data/output/test/picky_eater/online_property/dqn_lta_aux/successor_as/best/"
     },
    # {"label": "Random",
     # "control": "data/output/test/picky_eater/control/baseline/different_task/fix_rep/random/best/",
     # "online_measure": "data/output/test/picky_eater/fixrep_property/random/best",
    #  },
]




crgb_diff_tune_early = [
    {"label": "ReLU", "property": "data/output/test/",
     "control": "data/output/test/picky_eater/control/early_stop/different_task/fine_tune/dqn/best/",
     "online_measure": "data/output/test/picky_eater/online_property/dqn/best/",
     },
    {"label": "ReLU+Control",
     "control": "data/output/test/picky_eater/control/early_stop/different_task/fine_tune/dqn_aux/aux_control/best/",
     "online_measure": "data/output/test/picky_eater/online_property/dqn_aux/aux_control/best/",
     },
    {"label": "ReLU+XY",
     "control": "data/output/test/picky_eater/control/early_stop/different_task/fine_tune/dqn_aux/info/best_xy/",
     "online_measure": "data/output/test/picky_eater/online_property/dqn_aux/info/best_xy/",
     },
    {"label": "ReLU+Decoder",
     "control": "data/output/test/picky_eater/control/early_stop/different_task/fine_tune/dqn_aux/input_decoder/best/",
     "online_measure": "data/output/test/picky_eater/online_property/dqn_aux/input_decoder/best/",
     },
    {"label": "ReLU+NAS",
     "control": "data/output/test/picky_eater/control/early_stop/different_task/fine_tune/dqn_aux/nas_v2_delta/best/",
     "online_measure": "data/output/test/picky_eater/online_property/dqn_aux/nas_v2_delta/best/"
     },
    {"label": "ReLU+Reward",
     "control": "data/output/test/picky_eater/control/early_stop/different_task/fine_tune/dqn_aux/reward/best/",
     "online_measure": "data/output/test/picky_eater/online_property/dqn_aux/reward/best/"
     },
    {"label": "ReLU+SF",
     "control": "data/output/test/picky_eater/control/early_stop/different_task/fine_tune/dqn_aux/successor_as/best/",
     "online_measure": "data/output/test/picky_eater/online_property/dqn_aux/successor_as/best/"
     },
    {"label": "FTA eta=2",
     "control": "data/output/test/picky_eater/control/early_stop/different_task/fine_tune/dqn_lta/best/",
     "online_measure": "data/output/test/picky_eater/online_property/dqn_lta/best/"
     },
    {"label": "FTA+Control",
     "control": "data/output/test/picky_eater/control/early_stop/different_task/fine_tune/dqn_lta_aux/aux_control/best/",
     "online_measure": "data/output/test/picky_eater/online_property/dqn_lta_aux/aux_control/best/"
     },
    {"label": "FTA+XY",
     "control": "data/output/test/picky_eater/control/early_stop/different_task/fine_tune/dqn_lta_aux/info/best_xy/",
     "online_measure": "data/output/test/picky_eater/online_property/dqn_lta_aux/info/best_xy/"
     },
    {"label": "FTA+Decoder",
     "control": "data/output/test/picky_eater/control/early_stop/different_task/fine_tune/dqn_lta_aux/input_decoder/best/",
     "online_measure": "data/output/test/picky_eater/online_property/dqn_lta_aux/input_decoder/best/"
     },
    {"label": "FTA+NAS",
     "control": "data/output/test/picky_eater/control/early_stop/different_task/fine_tune/dqn_lta_aux/nas_v2_delta/best/",
     "online_measure": "data/output/test/picky_eater/online_property/dqn_lta_aux/nas_v2_delta/best/"
     },
    {"label": "FTA+Reward",
     "control": "data/output/test/picky_eater/control/early_stop/different_task/fine_tune/dqn_lta_aux/reward/best/",
     "online_measure": "data/output/test/picky_eater/online_property/dqn_lta_aux/reward/best/"
     },
    {"label": "FTA+SF",
     "control": "data/output/test/picky_eater/control/early_stop/different_task/fine_tune/dqn_lta_aux/successor_as/best/",
     "online_measure": "data/output/test/picky_eater/online_property/dqn_lta_aux/successor_as/best/"
     },
    # {"label": "Random",
     # "control": "data/output/test/picky_eater/control/baseline/different_task/fine_tune/random/best/",
     # "online_measure": "data/output/test/picky_eater/fixrep_property/random/best",
    #  },
]


crgb_similar_early = [
    {"label": "ReLU", "property": "data/output/test/",
     "control": "data/output/test/gridhard/control/similar_task/fix_rep/dqn/best_early/",
     "online_measure": "data/output/test/gridhard/online_property/dqn/best/"
     },
#     {"label": "ReLU+Control1g",
     # "control": "data/output/test/gridhard/control/last/similar_task/fix_rep/dqn_aux/aux_control/best_1g/",
     # "online_measure": "data/output/test/gridhard/online_property/dqn_aux/freq8/aux_control/best_1g/"
     # },
    # {"label": "ReLU+Control5g",
     # "control": "data/output/test/gridhard/control/last/similar_task/fix_rep/dqn_aux/aux_control/best_5g/",
     # "online_measure": "data/output/test/gridhard/online_property/dqn_aux/freq128/aux_control/best_5g/"
     # },
    # {"label": "ReLU+XY",
     # "control": "data/output/test/gridhard/control/last/similar_task/fix_rep/dqn_aux/info/best/",
     # "online_measure": "data/output/test/gridhard/online_property/dqn_aux/freq8/info/best/"
     # },
    # {"label": "ReLU+Decoder",
     # "control": "data/output/test/gridhard/control/last/similar_task/fix_rep/dqn_aux/input_decoder/best/",
     # "online_measure": "data/output/test/gridhard/online_property/dqn_aux/freq8/info/best/"
     # },
    # {"label": "ReLU+NAS",
     # "control": "data/output/test/gridhard/control/last/similar_task/fix_rep/dqn_aux/nas_v2_delta/best/",
     # "online_measure": "data/output/test/gridhard/online_property/dqn_aux/freq8/nas_v2_delta/best/"
     # },
    # {"label": "ReLU+Reward",
     # "control": "data/output/test/gridhard/control/last/similar_task/fix_rep/dqn_aux/reward/best/",
     # "online_measure": "data/output/test/gridhard/online_property/dqn_aux/freq8/reward/best/"
     # },
    # {"label": "ReLU+SF",
     # "control": "data/output/test/gridhard/control/last/similar_task/fix_rep/dqn_aux/successor_as/best/",
     # "online_measure": "data/output/test/gridhard/online_property/dqn_aux/freq8/successor_as/best/"
     # },
    {"label": "LTA",
      "control": "data/output/test/picky_eater/control/similar_task/fix_rep/dqn_lta/best_early/",
      "online_measure": "data/output/test/picky_eater/online_property/dqn_lta/best/"
      },
    # {"label": "LTA+Control1g",
     # "control": "data/output/test/gridhard/control/last/similar_task/fix_rep/dqn_lta_aux/aux_control/best_1g/",
     # "online_measure": "data/output/test/gridhard/online_property/dqn_lta_aux/aux_control/best_1g/"
     # },
    # {"label": "LTA+Control5g",
     # "control": "data/output/test/gridhard/control/last/similar_task/fix_rep/dqn_lta_aux/aux_control/best_5g/",
     # "online_measure": "data/output/test/gridhard/online_property/dqn_lta_aux/aux_control/best_5g/"
     # },
    # {"label": "LTA+XY",
     # "control": "data/output/test/gridhard/control/last/similar_task/fix_rep/dqn_lta_aux/info/best/",
     # "online_measure": "data/output/test/gridhard/online_property/dqn_lta_aux/info/best/"
     # },
    # {"label": "LTA+Decoder",
     # "control": "data/output/test/gridhard/control/last/similar_task/fix_rep/dqn_lta_aux/input_decoder/best/",
     # "online_measure": "data/output/test/gridhard/online_property/dqn_lta_aux/input_decoder/best/"
     # },
    # {"label": "LTA+NAS",
     # "control": "data/output/test/gridhard/control/last/similar_task/fix_rep/dqn_lta_aux/nas_v2_delta/best/",
     # "online_measure": "data/output/test/gridhard/online_property/dqn_lta_aux/nas_v2_delta/best/"
     # },
    # {"label": "LTA+Reward",
     # "control": "data/output/test/gridhard/control/last/similar_task/fix_rep/dqn_lta_aux/reward/best/",
     # "online_measure": "data/output/test/gridhard/online_property/dqn_lta_aux/reward/best/"
     # },
    # {"label": "LTA+SF",
     # "control": "data/output/test/gridhard/control/last/similar_task/fix_rep/dqn_lta_aux/successor_as/best/",
     # "online_measure": "data/output/test/gridhard/online_property/dqn_lta_aux/successor_as/best/"
     # },
    # {"label": "Random",
     # "control": "data/output/test/gridhard/control/baseline/similar_task/fix_rep/random/best/",
     # "online_measure": "data/output/test/gridhard/fixrep_property/random/best",
     # },
    # {"label": "Input",
     # "control": "data/output/test/gridhard/control/baseline/similar_task/fix_rep/input/best/",
     # "online_measure": "data/output/test/gridhard/fixrep_property/input/best",
     # },
]


crgb_online_sweep = [

    # {"label": "LTA",
     # "control": "data/output/test/picky_eater/online_property/dqn_lta/best/",
     # },
    {"label": "LTA+Control",
     "control": "data/output/test/picky_eater/online_property/dqn_lta_aux/aux_control/sweep/",
     },    
    {"label": "LTA+XY",
     "control": "data/output/test/picky_eater/online_property/dqn_lta_aux/info/sweep_xy/",
     },
    {"label": "LTA+Decoder",
     "control": "data/output/test/picky_eater/online_property/dqn_lta_aux/input_decoder/sweep/",
     },
    {"label": "LTA+NAS",
     "control": "data/output/test/picky_eater/online_property/dqn_lta_aux/nas_v2_delta/sweep/",
     },
    {"label": "LTA+SF",
     "control": "data/output/test/picky_eater/online_property/dqn_lta_aux/successor_as/sweep/",
     },
    {"label": "LTA+Reward",
     "control": "data/output/test/picky_eater/online_property/dqn_lta_aux/reward/sweep/",
    },
 #    {"label": "ReLU",
     # "control": "data/output/test/picky_eater/online_property/dqn/sweep/",
     # },
   {"label": "ReLU+Reward",
     "control": "data/output/test/picky_eater/online_property/dqn_aux/reward/sweep/",
     },
    {"label": "ReLU+Control", "control": "data/output/test/picky_eater/online_property/dqn_aux/aux_control/sweep/",
     },    
    {"label": "ReLU+XY",
     "control": "data/output/test/picky_eater/online_property/dqn_aux/info/sweep_xy/",
     },
     {"label": "ReLU+Decoder",
     "control": "data/output/test/picky_eater/online_property/dqn_aux/input_decoder/sweep/",
     },
    {"label": "ReLU+NAS",
     "control": "data/output/test/picky_eater/online_property/dqn_aux/nas_v2_delta/sweep/",
     },
    {"label": "ReLU+SF",
     "control": "data/output/test/picky_eater/online_property/dqn_aux/successor_as/sweep/",
     },
        ]

crgb_online_best = [
    {"label": "FTA",
     "control": "data/output/test/picky_eater/online_property/dqn_lta/best/",
     },
    {"label": "FTA+Control",
     "control": "data/output/test/picky_eater/online_property/dqn_lta_aux/aux_control/best/",
     },    
    {"label": "FTA+XY",
     "control": "data/output/test/picky_eater/online_property/dqn_lta_aux/info/best_xy/",
     },
    {"label": "FTA+Decoder",
     "control": "data/output/test/picky_eater/online_property/dqn_lta_aux/input_decoder/best/",
     },
    {"label": "FTA+NAS",
     "control": "data/output/test/picky_eater/online_property/dqn_lta_aux/nas_v2_delta/best/",
     },
    {"label": "FTA+SF",
     "control": "data/output/test/picky_eater/online_property/dqn_lta_aux/successor_as/best/",
     },
    {"label": "FTA+Reward",
     "control": "data/output/test/picky_eater/online_property/dqn_lta_aux/reward/best/",
    },
    {"label": "ReLU",
     "control": "data/output/test/picky_eater/online_property/dqn/best/",
     },
   {"label": "ReLU+Reward",
     "control": "data/output/test/picky_eater/online_property/dqn_aux/reward/best/",
     },
    {"label": "ReLU+Control", "control": "data/output/test/picky_eater/online_property/dqn_aux/aux_control/best/",
     },    
    {"label": "ReLU+XY",
     "control": "data/output/test/picky_eater/online_property/dqn_aux/info/best_xy/",
     },
     {"label": "ReLU+Decoder",
     "control": "data/output/test/picky_eater/online_property/dqn_aux/input_decoder/best/",
     },
    {"label": "ReLU+NAS",
     "control": "data/output/test/picky_eater/online_property/dqn_aux/nas_v2_delta/best/",
     },
    {"label": "ReLU+SF",
     "control": "data/output/test/picky_eater/online_property/dqn_aux/successor_as/best/",
     },
    ]


crgb_online_sweep_1 = [
    {"label": "FTA+Control",
     "control": "data/output/test/picky_eater/online_property/dqn_lta_aux/aux_control/sweep_1/",
     },    
    {"label": "ReLU+Control", "control": "data/output/test/picky_eater/online_property/dqn_aux/aux_control/sweep_1/",
     },    
    ]

crgb_online_sweep_2 = [
    {"label": "FTA+Control",
     "control": "data/output/test/picky_eater/online_property/dqn_lta_aux/aux_control/sweep_2/",
     },    
    {"label": "ReLU+Control", "control": "data/output/test/picky_eater/online_property/dqn_aux/aux_control/sweep_2/",
     },
    {"label": "FTA+Control",
     "control": "data/output/test/picky_eater/online_property/dqn_lta_aux/aux_control/sweep_3/",
     },    
    {"label": "ReLU+Control", "control": "data/output/test/picky_eater/online_property/dqn_aux/aux_control/sweep_3/",
     },    
    {"label": "FTA+Control",
     "control": "data/output/test/picky_eater/online_property/dqn_lta_aux/aux_control/sweep_4/",
     },    
    {"label": "ReLU+Control", "control": "data/output/test/picky_eater/online_property/dqn_aux/aux_control/sweep_4/",
     }, 
    ]

crgb_online_sweep_1_f = [
    {"label": "FTA",
     "control": "data/output/test/picky_eater/online_property/dqn_lta/sweep_1f/",
     },        
    {"label": "FTA",
     "control": "data/output/test/picky_eater/online_property/dqn_lta/sweep_4f/",
     },
    {"label": "ReLU",
     "control": "data/output/test/picky_eater/online_property/dqn/sweep_1f/",
     },    
    {"label": "ReLU",
     "control": "data/output/test/picky_eater/online_property/dqn/sweep_4f/",
     },
    {"label": "FTA+Control",
     "control": "data/output/test/picky_eater/online_property/dqn_lta_aux/aux_control/sweep_1f/",
     },
    {"label": "FTA+Control",
     "control": "data/output/test/picky_eater/online_property/dqn_lta_aux/aux_control/sweep_4f/",
     },
    {"label": "ReLU+Control", "control": "data/output/test/picky_eater/online_property/dqn_aux/aux_control/sweep_1f/",
     },    
    {"label": "ReLU+Control", "control": "data/output/test/picky_eater/online_property/dqn_aux/aux_control/sweep_4f/",
     },
    ]

crgb_online_sweep_1f_ltatest = [
    {"label": "FTA",
     "control": "data/output/test/picky_eater/representation/dqn_aux/aux_control/sweep_1f/"
     },
    {"label": "FTA",
     "control": "data/output/test/picky_eater/representation/dqn_lta_aux/aux_control/sweep_1f_1/"
     }
    ]

crgb_online_sweep_34_f_test = [
    {"label": "FTA",
     "control": "data/output/test/picky_eater/representation/dqn_lta_aux/aux_control/sweep_1f/",
     "best": "1"
     },
    {"label": "FTA",
     "control": "data/output/test/picky_eater/representation/dqn_lta/sweep_3f/",
     "best": "1"
     },
    {"label": "FTA",
     "control": "data/output/test/picky_eater/representation/dqn_lta/sweep_4f/",
    "best": "1"
     },
   {"label": "ReLU",
     "control": "data/output/test/picky_eater/representation/dqn/sweep_3f/",
     "best": "1"
     },    
    {"label": "ReLU",
     "control": "data/output/test/picky_eater/representation/dqn/sweep_4f/",
     "best": "1"
     },
    {"label": "FTA+Control",
     "control": "data/output/test/picky_eater/representation/dqn_lta_aux/aux_control/sweep_3f/",
     "best": "1"
     },
    {"label": "FTA+Control",
     "control": "data/output/test/picky_eater/representation/dqn_lta_aux/aux_control/sweep_4f/",
     "best": "2"
     },
    {"label": "ReLU+Control", "control": "data/output/test/picky_eater/representation/dqn_aux/aux_control/sweep_3f/",
     "best": "1"
     },    
    {"label": "ReLU+Control", "control": "data/output/test/picky_eater/representation/dqn_aux/aux_control/sweep_4f/",
     "best": "2"
     },
    ]

crgb_online_sweep_14_f_control_test = [
    {"label": "FTA",
     "control": "data/output/test/picky_eater/control_test/decayep/last/different_task/fix_rep/dqn_lta/sweep_1f_1",
     "best": "3"
     },
    {"label": "FTA",
     "control": "data/output/test/picky_eater/control_test/decayep/last/different_task/fix_rep/dqn_lta/sweep_4f_1",
     "best": "3"
     },
   {"label": "ReLU",
     "control": "data/output/test/picky_eater/control_test/decayep/last/different_task/fix_rep/dqn/sweep_1f_1",
     "best": "0"
     },    
#     {"label": "ReLU",
     # "control": "data/output/test/picky_eater/control_test/decayep/last/different_task/fix_rep/dqn/sweep_4f_1",
     # "online_measure": "data/output/test/picky_eater/representation/dqn/sweep_4f/",
     # "best": "2"
     # },
    {"label": "FTA+Control",
     "control": "data/output/test/picky_eater/control_test/decayep/last/different_task/fix_rep/dqn_lta_aux/aux_control/sweep_1f_1",
     "best": "3"
     },
    {"label": "FTA+Control",
     "control": "data/output/test/picky_eater/control_test/decayep/last/different_task/fix_rep/dqn_lta_aux/aux_control/sweep_4f_1",
     "best": "3"
     },
    {"label": "ReLU+Control",
     "control": "data/output/test/picky_eater/control_test/decayep/last/different_task/fix_rep/dqn_aux/aux_control/sweep_1f_1",
     "best": "3"
     },    
#     {"label": "ReLU+Control", 
     # "control": "data/output/test/picky_eater/control_test/decayep/last/different_task/fix_rep/dqn_aux/aux_control/sweep_4f_1",
     # "best": "3"
     # },
    ]

crgb_online_best_14_f_control_test = [
    {"label": "FTA",
     "control": "data/output/test/picky_eater/control_test/decayep/last/different_task/fix_rep/dqn_lta/best_1f_1",
     "best": "3"
     },
#     {"label": "FTA",
     # "control": "data/output/test/picky_eater/control_test/decayep/last/different_task/fix_rep/dqn_lta/best_4f_1",
     # "best": "3"
#      },
   {"label": "ReLU",
     "control": "data/output/test/picky_eater/control_test/decayep/last/different_task/fix_rep/dqn/best_1f_1",
     "best": "1"
     },    
#     {"label": "ReLU",
     # "control": "data/output/test/picky_eater/control_test/decayep/last/different_task/fix_rep/dqn/best_4f_1",
     # "online_measure": "data/output/test/picky_eater/representation/dqn/best_4f/",
     # "best": "2"
     # },
    {"label": "FTA+Control",
     "control": "data/output/test/picky_eater/control_test/decayep/last/different_task/fix_rep/dqn_lta_aux/aux_control/best_1f_1",
     "best": "3"
     },
#     {"label": "FTA+Control",
     # "control": "data/output/test/picky_eater/control_test/decayep/last/different_task/fix_rep/dqn_lta_aux/aux_control/best_4f_1",
     # "best": "3"
#      },
    {"label": "ReLU+Control",
     "control": "data/output/test/picky_eater/control_test/decayep/last/different_task/fix_rep/dqn_aux/aux_control/best_1f_1",
     "best": "3"
     },    
#     {"label": "ReLU+Control", 
     # "control": "data/output/test/picky_eater/control_test/decayep/last/different_task/fix_rep/dqn_aux/aux_control/best_4f_1",
     # "best": "3"
     # },
    ]


crgb_online_sweep_34_f_control_test = [
    {"label": "FTA",
     "control": "data/output/test/picky_eater/control_test/decayep/last/different_task/fix_rep/dqn_lta/sweep_3f",
     "best": "3"
     },
    {"label": "FTA",
     "control": "data/output/test/picky_eater/control_test/decayep/last/different_task/fix_rep/dqn_lta/sweep_4f",
     "best": "3"
     },
   {"label": "ReLU",
     "control": "data/output/test/picky_eater/control_test/decayep/last/different_task/fix_rep/dqn/sweep_3f",
     "best": "2"
     },    
    {"label": "ReLU",
     "control": "data/output/test/picky_eater/control_test/decayep/last/different_task/fix_rep/dqn/sweep_4f",
     "online_measure": "data/output/test/picky_eater/representation/dqn/sweep_4f/",
     "best": "2"
     },
    {"label": "FTA+Control",
     "control": "data/output/test/picky_eater/control_test/decayep/last/different_task/fix_rep/dqn_lta_aux/aux_control/sweep_3f",
     "best": "3"
     },
    {"label": "FTA+Control",
     "control": "data/output/test/picky_eater/control_test/decayep/last/different_task/fix_rep/dqn_lta_aux/aux_control/sweep_4f",
     "best": "3"
     },
    {"label": "ReLU+Control",
     "control": "data/output/test/picky_eater/control_test/decayep/last/different_task/fix_rep/dqn_aux/aux_control/sweep_3f",
     "best": "3"
     },    
    {"label": "ReLU+Control", 
     "control": "data/output/test/picky_eater/control_test/decayep/last/different_task/fix_rep/dqn_aux/aux_control/sweep_4f",
     "best": "3"
     },
    ]


crgb_online_best_34_f_control_test = [
    {"label": "FTA",
     "control": "data/output/test/picky_eater/control_test/decayep/last/different_task/fix_rep/dqn_lta/best_3f",
     "best": "3"
     },
    {"label": "FTA",
     "control": "data/output/test/picky_eater/control_test/decayep/last/different_task/fix_rep/dqn_lta/best_4f",
     "best": "3"
     },
   {"label": "ReLU",
     "control": "data/output/test/picky_eater/control_test/decayep/last/different_task/fix_rep/dqn/best_3f",
     "best": "2"
     },    
    {"label": "ReLU",
     "control": "data/output/test/picky_eater/control_test/decayep/last/different_task/fix_rep/dqn/best_4f",
     "online_measure": "data/output/test/picky_eater/representation/dqn/best_4f/",
     "best": "2"
     },
    {"label": "FTA+Control",
     "control": "data/output/test/picky_eater/control_test/decayep/last/different_task/fix_rep/dqn_lta_aux/aux_control/best_3f",
     "best": "3"
     },
    {"label": "FTA+Control",
     "control": "data/output/test/picky_eater/control_test/decayep/last/different_task/fix_rep/dqn_lta_aux/aux_control/best_4f",
     "best": "3"
     },
    {"label": "ReLU+Control",
     "control": "data/output/test/picky_eater/control_test/decayep/last/different_task/fix_rep/dqn_aux/aux_control/best_3f",
     "best": "3"
     },    
    {"label": "ReLU+Control", 
     "control": "data/output/test/picky_eater/control_test/decayep/last/different_task/fix_rep/dqn_aux/aux_control/best_4f",
     "best": "3"
     },
    ]

crgb_online_best_3_f_control_test = [
    {"label": "FTA",
     "control": "data/output/test/picky_eater/control_test/decayep/last/different_task/fix_rep/dqn_lta/best_3f",
     "best": "3"
     },
  {"label": "ReLU",
     "control": "data/output/test/picky_eater/control_test/decayep/last/different_task/fix_rep/dqn/best_3f",
     "best": "2"
     },    
   {"label": "FTA+Control",
     "control": "data/output/test/picky_eater/control_test/decayep/last/different_task/fix_rep/dqn_lta_aux/aux_control/best_3f",
     "best": "3"
     },
   {"label": "ReLU+Control",
     "control": "data/output/test/picky_eater/control_test/decayep/last/different_task/fix_rep/dqn_aux/aux_control/best_3f",
     "best": "3"
     },    
   ]

crgb_online_best_4_f_control_test = [

    {"label": "FTA",
     "control": "data/output/test/picky_eater/control_test/decayep/last/different_task/fix_rep/dqn_lta/best_4f",
     "best": "3"
     },
   {"label": "ReLU",
     "control": "data/output/test/picky_eater/control_test/decayep/last/different_task/fix_rep/dqn/best_4f",
     "online_measure": "data/output/test/picky_eater/representation/dqn/best_4f/",
     "best": "2"
     },
   {"label": "FTA+Control",
     "control": "data/output/test/picky_eater/control_test/decayep/last/different_task/fix_rep/dqn_lta_aux/aux_control/best_4f",
     "best": "3"
     },
   {"label": "ReLU+Control", 
     "control": "data/output/test/picky_eater/control_test/decayep/last/different_task/fix_rep/dqn_aux/aux_control/best_4f",
     "best": "3"
     },
    ]

crgb_online_sweep_1f_control = [
    {"label": "ReLU+Control",
     "control": "data/output/test/picky_eater/control/different_task/fix_rep/dqn_aux/aux_control/sweep_1f",
     "best": "3"
     },
   {"label": "ReLU+Control",
     "control": "data/output/test/picky_eater/control/different_task/fix_rep/dqn_aux/aux_control/sweep_1f_99d",
     # "online_measure": "data/output/test/picky_eater/representation/dqn/best_4f/",
     "best": "2"
     },
   {"label": "FTA+Control",
     "control": "data/output/test/picky_eater/control/different_task/fix_rep/dqn_lta_aux/aux_control/sweep_1f",
     "best": "3"
     },
   {"label": "FTA+Control", 
     "control": "data/output/test/picky_eater/control/different_task/fix_rep/dqn_lta_aux/aux_control/sweep_1f_99d",
     "best": "3"
     },
    ]


crgb_online_best_4_f_randc = [
    {"label": "ReLU+Control",
    "control": "data/output/test/picky_eater_randc/representation/dqn_aux/aux_control/best_3f",
     },    
    {"label": "FTA+Control",
    "control": "data/output/test/picky_eater_randc/representation/dqn_lta_aux/aux_control/best_3f",
     },
   ]

crgb_online_best_1_f_99d_and_1d = [
    {"label": "ReLU+Control",
    "control": "data/output/test/picky_eater/representation/dqn_aux/aux_control/best_1f",
     },
    {"label": "ReLU+Control",
    "control": "data/output/test/picky_eater/representation/dqn_aux/aux_control/best_1f_99d",
     },
    {"label": "FTA+Control",
    "control": "data/output/test/picky_eater/representation/dqn_lta_aux/aux_control/best_1f",
     },
    {"label": "FTA+Control",
    "control": "data/output/test/picky_eater/representation/dqn_lta_aux/aux_control/best_1f_99d",
     },
   ]


pe_sweep_temp = [
    {"label": "ReLU",
     "control": "data/output/result/picky_eater/linear_vf/representation/dqn/sweep_3f/",
     },
    {"label": "ReLU+Control",
     "control": "data/output/result/picky_eater/linear_vf/representation/dqn_aux/aux_control/sweep_3f/",
     },
    {"label": "ReLU+XY",
     "control": "data/output/result/picky_eater/linear_vf/representation/dqn_aux/info/sweep_3f_xy/",
     },
    {"label": "ReLU+Color",
     "control": "data/output/result/picky_eater/linear_vf/representation/dqn_aux/info/sweep_3f_color/",
     },
    {"label": "ReLU+Decoder",
     "control": "data/output/result/picky_eater/linear_vf/representation/dqn_aux/input_decoder/sweep_3f/",
     },
    {"label": "ReLU+NAS",
     "control": "data/output/result/picky_eater/linear_vf/representation/dqn_aux/nas_v2_delta/sweep_3f/",
     },
    {"label": "ReLU+Reward",
     "control": "data/output/result/picky_eater/linear_vf/representation/dqn_aux/reward/sweep_3f/",
     },
    {"label": "ReLU+SF",
     "control": "data/output/result/picky_eater/linear_vf/representation/dqn_aux/successor_as/sweep_3f/",
     },
    {"label": "LTA",
     "control": "data/output/result/picky_eater/linear_vf/representation/dqn_lta/sweep_3f/",
     },
    {"label": "FTA+Control",
     "control": "data/output/result/picky_eater/linear_vf/representation/dqn_lta_aux/aux_control/sweep_3f/",
     },
    {"label": "FTA+XY",
     "control": "data/output/result/picky_eater/linear_vf/representation/dqn_lta_aux/info/sweep_3f_xy/",
     },
    {"label": "FTA+Color",
     "control": "data/output/result/picky_eater/linear_vf/representation/dqn_lta_aux/info/sweep_3f_color/",
     },
    {"label": "FTA+Decoder",
     "control": "data/output/result/picky_eater/linear_vf/representation/dqn_lta_aux/input_decoder/sweep_3f/",
     },
    {"label": "LTA+NAS",
     "control": "data/output/result/picky_eater/linear_vf/representation/dqn_lta_aux/nas_v2_delta/sweep_3f/",
     },
    {"label": "LTA+Reward",
     "control": "data/output/result/picky_eater/linear_vf/representation/dqn_lta_aux/reward/sweep_3f/",
     },
    {"label": "LTA+SF",
     "control": "data/output/result/picky_eater/linear_vf/representation/dqn_lta_aux/successor_as/sweep_3f/",
     },
    ]

pe_best_temp = [
    {"label": "ReLU+Control",
     "control": "data/output/test_v6/picky_eater/online_property/dqn_aux/aux_control/best_3f/",
     },
    {"label": "ReLU+Color",
     "control": "data/output/test_v6/picky_eater/online_property/dqn_aux/info/best_3f_color/",
     },
    {"label": "FTA+Control",
     "control": "data/output/test_v6/picky_eater/online_property/dqn_lta_aux/aux_control/best_3f/",
     },
    {"label": "FTA+Color",
     "control": "data/output/test_v6/picky_eater/online_property/dqn_lta_aux/info/best_3f_color/",
     },
]

pe_trans_sweep_temp = [
    {"label": "ReLU",
     "control": "data/output/result/picky_eater/linear_vf/control/early_stop/different_task/fix_rep/dqn/sweep_3f/",
     },
    {"label": "ReLU+Control",
     "control": "data/output/result/picky_eater/linear_vf/control/early_stop/different_task/fix_rep/dqn_aux/aux_control/sweep_3f/",
     },
    {"label": "ReLU+XY",
     "control": "data/output/result/picky_eater/linear_vf/control/early_stop/different_task/fix_rep/dqn_aux/info/sweep_3f_xy/",
     },
    {"label": "ReLU+Color",
     "control": "data/output/result/picky_eater/linear_vf/control/early_stop/different_task/fix_rep/dqn_aux/info/sweep_3f_color/",
     },
    {"label": "ReLU+Decoder",
     "control": "data/output/result/picky_eater/linear_vf/control/early_stop/different_task/fix_rep/dqn_aux/input_decoder/sweep_3f/",
     },
    {"label": "ReLU+NAS",
     "control": "data/output/result/picky_eater/linear_vf/control/early_stop/different_task/fix_rep/dqn_aux/nas_v2_delta/sweep_3f/",
     },
    {"label": "ReLU+Reward",
     "control": "data/output/result/picky_eater/linear_vf/control/early_stop/different_task/fix_rep/dqn_aux/reward/sweep_3f/",
     },
    {"label": "ReLU+SF",
     "control": "data/output/result/picky_eater/linear_vf/control/early_stop/different_task/fix_rep/dqn_aux/successor_as/sweep_3f/",
     },
    {"label": "LTA",
     "control": "data/output/result/picky_eater/linear_vf/control/early_stop/different_task/fix_rep/dqn_lta/sweep_3f/",
     },
    {"label": "FTA+Control",
     "control": "data/output/result/picky_eater/linear_vf/control/early_stop/different_task/fix_rep/dqn_lta_aux/aux_control/sweep_3f/",
     },
    {"label": "FTA+XY",
     "control": "data/output/result/picky_eater/linear_vf/control/early_stop/different_task/fix_rep/dqn_lta_aux/info/sweep_3f_xy/",
     },
    {"label": "FTA+Color",
     "control": "data/output/result/picky_eater/linear_vf/control/early_stop/different_task/fix_rep/dqn_lta_aux/info/sweep_3f_color/",
     },
    {"label": "FTA+Decoder",
     "control": "data/output/result/picky_eater/linear_vf/control/early_stop/different_task/fix_rep/dqn_lta_aux/input_decoder/sweep_3f/",
     },
    {"label": "LTA+NAS",
     "control": "data/output/result/picky_eater/linear_vf/control/early_stop/different_task/fix_rep/dqn_lta_aux/nas_v2_delta/sweep_3f/",
     },
    {"label": "LTA+Reward",
     "control": "data/output/result/picky_eater/linear_vf/control/early_stop/different_task/fix_rep/dqn_lta_aux/reward/sweep_3f/",
     },
    {"label": "LTA+SF",
     "control": "data/output/result/picky_eater/linear_vf/control/early_stop/different_task/fix_rep/dqn_lta_aux/successor_as/sweep_3f/",
     },
]
pe_trans_best_temp = [
    {"label": "ReLU+Control",
     "control": "data/output/test_v6/picky_eater/control/early_stop/different_task/fix_rep/dqn_aux/aux_control/best_3f_long/",
     "online_measure": "data/output/test_v6/picky_eater/online_property/dqn_aux/aux_control/best_3f/"
     },
    {"label": "ReLU+Color",
     "control": "data/output/test_v6/picky_eater/control/early_stop/different_task/fix_rep/dqn_aux/info/best_3f_color_long/",
     "online_measure": "data/output/test_v6/picky_eater/online_property/dqn_aux/info/best_3f_color/"
     },
    # {"label": "ReLU+Control+XY+Color",
    #  "control": "data/output/test_v6/picky_eater/control/early_stop/different_task/fix_rep/dqn_aux/control+info/best_3f_xy_color_linearAux/",
    #  "online_measure": "data/output/test_v6/picky_eater/online_property/dqn_aux/control+info/best_3f_xy_color/"
    #  },
    {"label": "FTA+Control",
     "control": "data/output/test_v6/picky_eater/control/early_stop/different_task/fix_rep/dqn_lta_aux/aux_control/best_3f_long/",
     "online_measure": "data/output/test_v6/picky_eater/online_property/dqn_aux/aux_control/best_3f/"
     },
    {"label": "FTA+Color",
     "control": "data/output/test_v6/picky_eater/control/early_stop/different_task/fix_rep/dqn_lta_aux/info/best_3f_color_long/",
     "online_measure": "data/output/test_v6/picky_eater/online_property/dqn_aux/info/best_3f_color/"
     },
    # {"label": "FTA+Control+XY+Color",
    #  "control": "data/output/test_v6/picky_eater/control/early_stop/different_task/fix_rep/dqn_lta_aux/control+info/best_3f_xy_color_linearAux/",
    #  "online_measure": "data/output/test_v6/picky_eater/online_property/dqn_aux/control+info/best_3f_xy_color/"
    #  },
]


perand_sweep_temp = [
    # {"label": "ReLU+Control",
    #  "control": "data/output/test/picky_eater_randc/representation/dqn_aux/aux_control/sweep_3f/",
    #  },
    # {"label": "LTA+Control",
    #  "control": "data/output/test/picky_eater_randc/representation/dqn_lta_aux/aux_control/sweep_3f/",
    #  },
    # {"label": "ReLU",
    #  "control": "data/output/test/picky_eater_randc/representation/dqn_switch/sweep_3f/",
    #  },
    # {"label": "FTA",
    #  "control": "data/output/test/picky_eater_randc/representation/dqn_lta_switch/sweep_3f/",
    #  },
    # {"label": "ReLU+Control+XY",
    #  "control": "data/output/test/picky_eater_randc/representation/dqn_aux/control+info/sweep_3f_xy/",
    #  },
    {"label": "ReLU+Control+XY+Color",
     "control": "data/output/test_v6/picky_eater_randc/representation/dqn_aux/control+info/sweep_3f_xy_color/",
     },
    # {"label": "ReLU+Control_Test",
    #  "control": "data/output/test/picky_eater_randc/representation/dqn_aux/aux_control/test_3f/",
    #  },
    # {"label": "ReLU+Control_Large",
    #  "control": "data/output/test/picky_eater_randc/representation/dqn_aux/aux_control/sweep_3f_large/",
    #  },
    # {"label": "FTA+Control+XY",
    #  "control": "data/output/test/picky_eater_randc/representation/dqn_lta_aux/control+info/sweep_3f_xy/",
    #  },
    {"label": "FTA+Control+XY+Color",
     "control": "data/output/test_v6/picky_eater_randc/representation/dqn_lta_aux/control+info/sweep_3f_xy_color/",
     },
    # {"label": "FTA+Control_Test",
    #  "control": "data/output/test/picky_eater_randc/representation/dqn_lta_aux/aux_control/test_3f/",
    #  },
    # {"label": "FTA+Control_Large",
    #  "control": "data/output/test/picky_eater_randc/representation/dqn_lta_aux/aux_control/sweep_3f_large/",
    #  },
    ]
# pe_trans_sweep_temp = [
#     {"label": "ReLU+Control",
#      "control": "data/output/test/picky_eater/control_test/decayep/last/different_task/fix_rep/dqn_aux/aux_control/sweep_3f/",
#      },
#     {"label": "FTA+Control",
#      "control": "data/output/test/picky_eater/control_test/decayep/last/different_task/fix_rep/dqn_lta_aux/aux_control/sweep_3f/",
#      },
#     {"label": "FTA+Control+Info",
#      "control": "data/output/test/picky_eater/control_test/decayep/last/different_task/fix_rep/dqn_lta_aux/aux_control/sweep_3f_xy_color/",
#      },
# ]

perand_trans_sweep_temp = [
    {"label": "ReLU+Control+Info",
     "control": "data/output/test_v6/picky_eater_randc/control/last/different_task/fix_rep/dqn_aux/control+info/sweep_3f_xy_color/",
     },
    {"label": "FTA+Control+Info",
     "control": "data/output/test_v6/picky_eater_randc/control/last/different_task/fix_rep/dqn_lta_aux/control+info/sweep_3f_xy_color/",
     },
]


pe_linear_rep = [
    {"label": "ReLU",
     "control": "data/output/result/picky_eater/linear_vf/representation/dqn/best_3f/",
     },
    {"label": "ReLU+Control",
     "control": "data/output/result/picky_eater/linear_vf/representation/dqn_aux/aux_control/best_3f/",
     },
    {"label": "ReLU+XY",
     "control": "data/output/result/picky_eater/linear_vf/representation/dqn_aux/info/best_3f_xy/",
     },
    {"label": "ReLU+Color",
     "control": "data/output/result/picky_eater/linear_vf/representation/dqn_aux/info/best_3f_color/",
     },
    {"label": "ReLU+Decoder",
     "control": "data/output/result/picky_eater/linear_vf/representation/dqn_aux/input_decoder/best_3f/",
     },
    {"label": "ReLU+NAS",
     "control": "data/output/result/picky_eater/linear_vf/representation/dqn_aux/nas_v2_delta/best_3f/",
     },
    {"label": "ReLU+Reward",
     "control": "data/output/result/picky_eater/linear_vf/representation/dqn_aux/reward/best_3f/",
     },
    {"label": "ReLU+SF",
     "control": "data/output/result/picky_eater/linear_vf/representation/dqn_aux/successor_as/best_3f/",
     },
    {"label": "FTA",
     "control": "data/output/result/picky_eater/linear_vf/representation/dqn_lta/best_3f/",
     },
    {"label": "FTA+Control",
     "control": "data/output/result/picky_eater/linear_vf/representation/dqn_lta_aux/aux_control/best_3f/",
     },
    {"label": "FTA+XY",
     "control": "data/output/result/picky_eater/linear_vf/representation/dqn_lta_aux/info/best_3f_xy/",
     },
    {"label": "FTA+Color",
     "control": "data/output/result/picky_eater/linear_vf/representation/dqn_lta_aux/info/best_3f_color/",
     },
    {"label": "FTA+Decoder",
     "control": "data/output/result/picky_eater/linear_vf/representation/dqn_lta_aux/input_decoder/best_3f/",
     },
    {"label": "FTA+NAS",
     "control": "data/output/result/picky_eater/linear_vf/representation/dqn_lta_aux/nas_v2_delta/best_3f/",
     },
    {"label": "FTA+Reward",
     "control": "data/output/result/picky_eater/linear_vf/representation/dqn_lta_aux/reward/best_3f/",
     },
    {"label": "FTA+SF",
     "control": "data/output/result/picky_eater/linear_vf/representation/dqn_lta_aux/successor_as/best_3f/",
     },
    ]


pe_linear_trans_diff = [
    {"label": "ReLU",
     "control": "data/output/result/picky_eater/linear_vf/control/early_stop/different_task/fix_rep/dqn/best_3f/",
     },
    {"label": "ReLU+Control",
     "control": "data/output/result/picky_eater/linear_vf/control/early_stop/different_task/fix_rep/dqn_aux/aux_control/best_3f/",
     },
    {"label": "ReLU+XY",
     "control": "data/output/result/picky_eater/linear_vf/control/early_stop/different_task/fix_rep/dqn_aux/info/best_3f_xy/",
     },
    {"label": "ReLU+Color",
     "control": "data/output/result/picky_eater/linear_vf/control/early_stop/different_task/fix_rep/dqn_aux/info/best_3f_color/",
     },
    {"label": "ReLU+Decoder",
     "control": "data/output/result/picky_eater/linear_vf/control/early_stop/different_task/fix_rep/dqn_aux/input_decoder/best_3f/",
     },
    {"label": "ReLU+NAS",
     "control": "data/output/result/picky_eater/linear_vf/control/early_stop/different_task/fix_rep/dqn_aux/nas_v2_delta/best_3f/",
     },
    {"label": "ReLU+Reward",
     "control": "data/output/result/picky_eater/linear_vf/control/early_stop/different_task/fix_rep/dqn_aux/reward/best_3f/",
     },
    {"label": "ReLU+SF",
     "control": "data/output/result/picky_eater/linear_vf/control/early_stop/different_task/fix_rep/dqn_aux/successor_as/best_3f/",
     },
    {"label": "FTA",
     "control": "data/output/result/picky_eater/linear_vf/control/early_stop/different_task/fix_rep/dqn_lta/best_3f/",
     },
    {"label": "FTA+Control",
     "control": "data/output/result/picky_eater/linear_vf/control/early_stop/different_task/fix_rep/dqn_lta_aux/aux_control/best_3f/",
     },
    {"label": "FTA+XY",
     "control": "data/output/result/picky_eater/linear_vf/control/early_stop/different_task/fix_rep/dqn_lta_aux/info/best_3f_xy/",
     },
    {"label": "FTA+Color",
     "control": "data/output/result/picky_eater/linear_vf/control/early_stop/different_task/fix_rep/dqn_lta_aux/info/best_3f_color/",
     },
    {"label": "FTA+Decoder",
     "control": "data/output/result/picky_eater/linear_vf/control/early_stop/different_task/fix_rep/dqn_lta_aux/input_decoder/best_3f/",
     },
    {"label": "FTA+NAS",
     "control": "data/output/result/picky_eater/linear_vf/control/early_stop/different_task/fix_rep/dqn_lta_aux/nas_v2_delta/best_3f/",
     },
    {"label": "FTA+Reward",
     "control": "data/output/result/picky_eater/linear_vf/control/early_stop/different_task/fix_rep/dqn_lta_aux/reward/best_3f/",
     },
    {"label": "FTA+SF",
     "control": "data/output/result/picky_eater/linear_vf/control/early_stop/different_task/fix_rep/dqn_lta_aux/successor_as/best_3f/",
     },

    {"label": "Scratch",
     "control": "data/output/result/picky_eater/linear_vf/representation/dqn/best_3f/",
     },
    ]
