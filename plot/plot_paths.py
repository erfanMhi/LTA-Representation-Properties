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
             "#BBBBBB", # grey
             "#F4A582"
             ]
s_default = ["-", "--", ":"]
m_default = ["o", "^"]
property_keys = {"lipschitz": "Complexity Reduction",
                 "distance": "Dynamics Awareness",
                 "ortho": "Orthogonality",
                 "interf":"Noninterference",
                 "diversity":"Diversity",
                 "sparsity":"Sparsity",
                 "return": ""}

normalize_prop = {
    "lipschitz": [0.00727, 0.35973],
    "interf": [0.0, 0.07027447]
}

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
    "Complexity Reduction": c_default[0],
    "Dynamics Awareness": c_default[2],
    "Orthogonality": c_default[4],
    "Noninterference": c_default[6],
    "Diversity": c_default[8],
    "Sparsity": c_default[10],

    "ReLU": c_default[0],
    "ReLU+Control": c_default[2],
    "ReLU+VirtualVF1": c_default[2],
    "ReLU+VirtualVF5": c_default[3],
    "ReLU+XY": c_default[4],
    "ReLU+XY+color": c_default[5],
    "ReLU+XY+count": c_default[6],
    "ReLU+Color": c_default[6],
    "ReLU+Control+XY+Color": c_default[8],

    "ReLU+Decoder": c_default[7],
    "ReLU+NAS": c_default[8],
    "ReLU+Reward": c_default[9],
    "ReLU+SF": c_default[10],
    "ReLU+ATC": c_default[11],
    
    "ReLU+divConstr": c_default[11],
    "ReLU+ATC": c_default[13],
    "ReLU+ATC delta=2": c_default[0],
    "ReLU+ATC delta=3": c_default[1],
    "ReLU+ATC delta=4": c_default[2],
    "ReLU+ATC delta=5": c_default[3],

    "ReLU+ATC shift-prob=0": c_default[0],
    "ReLU+ATC shift-prob=0.01": c_default[1],
    "ReLU+ATC shift-prob=0.1": c_default[2],
    "ReLU+ATC shift-prob=0.2": c_default[3],
    "ReLU+ATC shift-prob=0.3": c_default[4],

    "ReLU+ATC encoder-size=8": c_default[0],
    "ReLU+ATC encoder-size=16": c_default[1],
    "ReLU+ATC encoder-size=32": c_default[2],
    "ReLU+ATC encoder-size=64": c_default[3],
    "ReLU+ATC encoder-size=128": c_default[4],


    "ReLU(L)": c_default[0],
    "ReLU(L)+VirtualVF1": c_default[2],
    "ReLU(L)+VirtualVF5": c_default[3],
    "ReLU(L)+XY": c_default[4],
    "ReLU(L)+Decoder": c_default[7],
    "ReLU(L)+NAS": c_default[8],
    "ReLU(L)+Reward": c_default[9],
    "ReLU(L)+SF": c_default[10],
    "ReLU(L)+ATC": c_default[13],

    "FTA": c_default[0],
    "FTA(no target)": c_default[0],

    "FTA eta=2": c_default[1],
    
    "FTA eta=0.2": c_default[1],
    "FTA eta=0.4": c_default[0],
    "FTA eta=0.6": c_default[11],
    "FTA eta=0.8": c_default[12],

    "FTA+Control": c_default[2],
    "FTA+VirtualVF1": c_default[2],
    "FTA+VirtualVF5": c_default[3],
    "FTA+XY": c_default[4],
    "FTA+XY+color": c_default[5],
    "FTA+XY+count": c_default[6],
    "FTA+Color": c_default[6],
    "FTA+Control+XY+Color": c_default[8],

    "FTA+Decoder": c_default[7],
    "FTA+NAS": c_default[8],
    "FTA+Reward": c_default[9],
    "FTA+SF": c_default[10],
    "FTA+ATC": c_default[13],


    "Random": 'red',
    "Random(L)": 'red',
    "Input": 'brown',
    "Scratch": "black",
    "Scratch(ReLU)": "black",
    "Scratch(L)": "black",
    "Scratch(FTA)": "black",

    "ReLU close10": c_default[1],
    "ReLU close25": c_default[2],
    "ReLU close50": c_default[3],
    "ReLU close75": c_default[4],
    
    "Linear": "C0",
    "Nonlinear": "C1",
    "Other rep": "grey",
}

curve_styles = {
    "Complexity Reduction": s_default[0],
    "Dynamics Awareness": s_default[0],
    "Orthogonality": s_default[0],
    "Noninterference": s_default[0],
    "Diversity": s_default[0],
    "Sparsity": s_default[0],

    "ReLU": s_default[1],
    "ReLU+Control": s_default[1],
    "ReLU+VirtualVF1": s_default[1],
    "ReLU+VirtualVF5": s_default[1],
    "ReLU+XY": s_default[1],
    "ReLU+XY+color": s_default[1],
    "ReLU+XY+count": s_default[1],
    "ReLU+Decoder": s_default[1],
    "ReLU+NAS": s_default[1],
    "ReLU+Reward": s_default[1],
    "ReLU+SF": s_default[1],
    "ReLU+Color": s_default[1],
    "ReLU+Control+XY+Color": s_default[1],
    "ReLU+divConstr": s_default[1],
    "ReLU+ATC": s_default[1],

    "ReLU+ATC delta=2": s_default[1],
    "ReLU+ATC delta=3": s_default[1],
    "ReLU+ATC delta=4": s_default[1],
    "ReLU+ATC delta=5": s_default[1],

    "ReLU+ATC shift-prob=0": s_default[1],
    "ReLU+ATC shift-prob=0.01": s_default[1],
    "ReLU+ATC shift-prob=0.1": s_default[1],
    "ReLU+ATC shift-prob=0.2": s_default[1],
    "ReLU+ATC shift-prob=0.3": s_default[1],

    "ReLU+ATC encoder-size=8": s_default[1],
    "ReLU+ATC encoder-size=16": s_default[1],
    "ReLU+ATC encoder-size=32": s_default[1],
    "ReLU+ATC encoder-size=64": s_default[1],
    "ReLU+ATC encoder-size=128": s_default[1],


    "ReLU(L)": s_default[2],
    "ReLU(L)+VirtualVF1": s_default[2],
    "ReLU(L)+VirtualVF5": s_default[2],
    "ReLU(L)+XY": s_default[2],
    "ReLU(L)+Decoder": s_default[2],
    "ReLU(L)+NAS": s_default[2],
    "ReLU(L)+Reward": s_default[2],
    "ReLU(L)+SF": s_default[2],
    "ReLU(L)+ATC": s_default[2],

    "FTA": s_default[0],
    "FTA(no target)": s_default[0],

    "FTA eta=2": s_default[0],
    
    "FTA eta=0.2": s_default[0],
    "FTA eta=0.4": s_default[0],
    "FTA eta=0.6": s_default[0],
    "FTA eta=0.8": s_default[0],

    "FTA+Control": s_default[0],
    "FTA+VirtualVF1": s_default[0],
    "FTA+VirtualVF5": s_default[0],
    "FTA+XY": s_default[0],
    "FTA+XY+color": s_default[0],
    "FTA+XY+count": s_default[0],
    "FTA+Decoder": s_default[0],
    "FTA+NAS": s_default[0],
    "FTA+Reward": s_default[0],
    "FTA+SF": s_default[0],
    "FTA+Color": s_default[0],
    "FTA+Control+XY+Color": s_default[0],
    "FTA+ATC": s_default[0],

    "Random": s_default[0],
    "Random(L)": s_default[2],
    "Input": s_default[0],
    "Scratch": s_default[0],
    "Scratch(ReLU)": s_default[1],
    "Scratch(L)": s_default[2],
    "Scratch(FTA)": s_default[0],

    "ReLU close10": s_default[0],
    "ReLU close25": s_default[0],
    "ReLU close50": s_default[0],
    "ReLU close75": s_default[0],
    
    "Other rep": s_default[0],
}

marker_styles = {
        "ReLU": m_default[1],
        "ReLU+Color": m_default[1],
        "ReLU+Control": m_default[1],
        "ReLU+VirtualVF1": m_default[1],
        "ReLU+VirtualVF5": m_default[1],
        "ReLU+XY": m_default[1],
        "ReLU+XY+color": m_default[1],
        "ReLU+XY+count": m_default[1],
        "ReLU+Decoder": m_default[1],
        "ReLU+NAS": m_default[1],
        "ReLU+Reward": m_default[1],
        "ReLU+SF": m_default[1],
        "ReLU+divConstr": m_default[1],
    "ReLU+ATC": m_default[1],

    "ReLU+ATC delta=2": m_default[1],
    "ReLU+ATC delta=3": m_default[1],
    "ReLU+ATC delta=4": m_default[1],
    "ReLU+ATC delta=5": m_default[1],

    "ReLU+ATC shift-prob=0": m_default[1],
    "ReLU+ATC shift-prob=0.01": m_default[1],
    "ReLU+ATC shift-prob=0.1": m_default[1],
    "ReLU+ATC shift-prob=0.2": m_default[1],
    "ReLU+ATC shift-prob=0.3": m_default[1],

    "ReLU+ATC encoder-size=8": m_default[1],
    "ReLU+ATC encoder-size=16": m_default[1],
    "ReLU+ATC encoder-size=32": m_default[1],
    "ReLU+ATC encoder-size=64": m_default[1],
    "ReLU+ATC encoder-size=128": m_default[1],

    "FTA": m_default[0],
    "FTA(no target)": m_default[0],

    "FTA eta=2": m_default[0],
    
    "FTA eta=0.2": m_default[0],
    "FTA eta=0.4": m_default[0],
    "FTA eta=0.6": m_default[0],
    "FTA eta=0.8": m_default[0],

    "FTA+Color": m_default[0],
    "FTA+Control": m_default[0],
    "FTA+VirtualVF1": m_default[0],
    "FTA+VirtualVF5": m_default[0],
    "FTA+XY": m_default[0],
    "FTA+XY+color": m_default[0],
    "FTA+XY+count": m_default[0],
    "FTA+Decoder": m_default[0],
    "FTA+NAS": m_default[0],
    "FTA+Reward": m_default[0],
    "FTA+SF": m_default[0],
    "FTA+ATC": m_default[0],

    "Random": m_default[0],
    "Input": m_default[0],
    "Scratch": m_default[0],

    "ReLU close10": m_default[0],
    "ReLU close25": m_default[0],
    "ReLU close50": m_default[0],
    "ReLU close75": m_default[0],

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
    # {"label": "ReLU+VirtualVF1",
    #  "control": "data/output/test_v7/gridhard/linear_vf/representation/dqn_aux/aux_control/sweep_1g/",
    #  },
    # {"label": "ReLU+VirtualVF5",
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
    # {"label": "FTA+VirtualVF1",
    #  "control": "data/output/test_v7/gridhard/linear_vf/representation/dqn_lta_aux/aux_control/sweep_1g/",
    #  },
    # {"label": "FTA+VirtualVF5",
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
    {"label": "ReLU+VirtualVF1",
     "control": "data/output/test_v7/gridhard/linear_vf/control/early_stop/different_task/fix_rep/dqn_aux/aux_control/sweep_1g/",
     },
    {"label": "ReLU+VirtualVF5",
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
    {"label": "FTA+VirtualVF1",
     "control": "data/output/test_v7/gridhard/linear_vf/control/early_stop/different_task/fix_rep/dqn_lta_aux/aux_control/sweep_1g/",
     },
    {"label": "FTA+VirtualVF5",
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
    {"label": "ReLU+VirtualVF1",
     "control": "data/output/test/gridhard/control/different_task/fine_tune/dqn_aux/aux_control/sweep_1g/",
     },
    {"label": "ReLU+VirtualVF5",
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
    {"label": "FTA+VirtualVF1",
     "control": "data/output/test/gridhard/control/different_task/fine_tune/dqn_lta_aux/aux_control/sweep_1g/",
     },
    {"label": "FTA+VirtualVF5",
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
    {"label": "ReLU+VirtualVF1",
     "control": "data/output/result/gridhard/linear_vf/online_property/dqn_aux/aux_control/best_1g_gamma0.9/",
     },
    {"label": "ReLU+VirtualVF5",
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
    {"label": "FTA+VirtualVF1",
     "control": "data/output/result/gridhard/linear_vf/online_property/dqn_lta_aux/aux_control/best_1g_gamma0.9/",
     },
    {"label": "FTA+VirtualVF5",
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
    {"label": "ReLU+VirtualVF1",
     "control": "data/output/result/gridhard/linear_vf/control/early_stop/same_task/fix_rep/dqn_aux/aux_control/best_1g_gamma0.9/",
     "online_measure": "data/output/result/gridhard/linear_vf/online_property/dqn_aux/aux_control/best_1g_gamma0.9/"
     },
    {"label": "ReLU+VirtualVF5",
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
    {"label": "FTA+VirtualVF1",
     "control": "data/output/result/gridhard/linear_vf/control/early_stop/same_task/fix_rep/dqn_lta_aux/aux_control/best_1g_gamma0.9/",
     "online_measure": "data/output/result/gridhard/linear_vf/online_property/dqn_lta_aux/aux_control/best_1g_gamma0.9/"
     },
    {"label": "FTA+VirtualVF5",
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
    {"label": "Scratch",
     "control": "data/output/result/gridhard/linear_vf/control/early_stop/different_task/scratch/dqn/best/",
     },
]

gh_similar_early = [
    {"label": "ReLU", "property": "data/output/result/",
     "control": "data/output/result/gridhard/linear_vf/control/early_stop/similar_task/fix_rep/dqn/best/",
     "online_measure": "data/output/result/gridhard/linear_vf/online_property/dqn/best/"
     },
    {"label": "ReLU+VirtualVF1",
     "control": "data/output/result/gridhard/linear_vf/control/early_stop/similar_task/fix_rep/dqn_aux/aux_control/best_1g_gamma0.9/",
     "online_measure": "data/output/result/gridhard/linear_vf/online_property/dqn_aux/aux_control/best_1g_gamma0.9/"
     },
    {"label": "ReLU+VirtualVF5",
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
    {"label": "FTA+VirtualVF1",
     "control": "data/output/result/gridhard/linear_vf/control/early_stop/similar_task/fix_rep/dqn_lta_aux/aux_control/best_1g_gamma0.9/",
     "online_measure": "data/output/result/gridhard/linear_vf/online_property/dqn_lta_aux/aux_control/best_1g_gamma0.9/"
     },
    {"label": "FTA+VirtualVF5",
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
    {"label": "ReLU+VirtualVF1",
     "control": "data/output/result/gridhard/linear_vf/control/early_stop/different_task/fix_rep/dqn_aux/aux_control/best_1g_gamma0.9/",
     "online_measure": "data/output/result/gridhard/linear_vf/online_property/dqn_aux/aux_control/best_1g_gamma0.9/"
     },
    {"label": "ReLU+VirtualVF5",
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
    {"label": "FTA+VirtualVF1",
     "control": "data/output/result/gridhard/linear_vf/control/early_stop/different_task/fix_rep/dqn_lta_aux/aux_control/best_1g_gamma0.9/",
     "online_measure": "data/output/result/gridhard/linear_vf/online_property/dqn_lta_aux/aux_control/best_1g_gamma0.9/"
     },
    {"label": "FTA+VirtualVF5",
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


gh_diff_tune_early = [
    {"label": "ReLU", "property": "data/output/result/",
     "control": "data/output/result/gridhard/linear_vf/control/early_stop/different_task/fine_tune/dqn/best/",
     "online_measure": "data/output/result/gridhard/linear_vf/online_property/dqn/best/"
     },
    {"label": "ReLU+VirtualVF1",
     "control": "data/output/result/gridhard/linear_vf/control/early_stop/different_task/fine_tune/dqn_aux/aux_control/best_1g_gamma0.9/",
     "online_measure": "data/output/result/gridhard/linear_vf/online_property/dqn_aux/aux_control/best_1g_gamma0.9/"
     },
    {"label": "ReLU+VirtualVF5",
     "control": "data/output/result/gridhard/linear_vf/control/early_stop/different_task/fine_tune/dqn_aux/aux_control/best_5g_gamma0.9/",
     "online_measure": "data/output/result/gridhard/linear_vf/online_property/dqn_aux/aux_control/best_5g_gamma0.9/"
     },
    {"label": "ReLU+XY",
     "control": "data/output/result/gridhard/linear_vf/control/early_stop/different_task/fine_tune/dqn_aux/info/best/",
     "online_measure": "data/output/result/gridhard/linear_vf/online_property/dqn_aux/info/best/"
     },
    {"label": "ReLU+Decoder",
     "control": "data/output/result/gridhard/linear_vf/control/early_stop/different_task/fine_tune/dqn_aux/input_decoder/best/",
     "online_measure": "data/output/result/gridhard/linear_vf/online_property/dqn_aux/info/best/"
     },
    {"label": "ReLU+NAS",
     "control": "data/output/result/gridhard/linear_vf/control/early_stop/different_task/fine_tune/dqn_aux/nas_v2_delta/best/",
     "online_measure": "data/output/result/gridhard/linear_vf/online_property/dqn_aux/nas_v2_delta/best/"
     },
    {"label": "ReLU+Reward",
     "control": "data/output/result/gridhard/linear_vf/control/early_stop/different_task/fine_tune/dqn_aux/reward/best/",
     "online_measure": "data/output/result/gridhard/linear_vf/online_property/dqn_aux/reward/best/"
     },
    {"label": "ReLU+SF",
     "control": "data/output/result/gridhard/linear_vf/control/early_stop/different_task/fine_tune/dqn_aux/successor_as/best/",
     "online_measure": "data/output/result/gridhard/linear_vf/online_property/dqn_aux/successor_as/best/"
     },
    {"label": "FTA eta=0.2",
     "control": "data/output/result/gridhard/linear_vf/control/early_stop/different_task/fine_tune/dqn_lta/eta_study_0.2_best/",
     "online_measure": "data/output/result/gridhard/linear_vf/online_property/dqn_lta/eta_study_0.2_best/"
     },
    {"label": "FTA eta=0.4",
     "control": "data/output/result/gridhard/linear_vf/control/early_stop/different_task/fine_tune/dqn_lta/eta_study_0.4_best/",
     "online_measure": "data/output/result/gridhard/linear_vf/online_property/dqn_lta/eta_study_0.4_best/"
     },
    {"label": "FTA eta=0.6",
     "control": "data/output/result/gridhard/linear_vf/control/early_stop/different_task/fine_tune/dqn_lta/eta_study_0.6_best/",
     "online_measure": "data/output/result/gridhard/linear_vf/online_property/dqn_lta/eta_study_0.6_best/"
     },
    {"label": "FTA eta=0.8",
     "control": "data/output/result/gridhard/linear_vf/control/early_stop/different_task/fine_tune/dqn_lta/eta_study_0.8_best/",
     "online_measure": "data/output/result/gridhard/linear_vf/online_property/dqn_lta/eta_study_0.8_best/"
     },
    {"label": "FTA+VirtualVF1",
     "control": "data/output/result/gridhard/linear_vf/control/early_stop/different_task/fine_tune/dqn_lta_aux/aux_control/best_1g_gamma0.9/",
     "online_measure": "data/output/result/gridhard/linear_vf/online_property/dqn_lta_aux/aux_control/best_1g_gamma0.9/"
     },
    {"label": "FTA+VirtualVF5",
     "control": "data/output/result/gridhard/linear_vf/control/early_stop/different_task/fine_tune/dqn_lta_aux/aux_control/best_5g_gamma0.9/",
     "online_measure": "data/output/result/gridhard/linear_vf/online_property/dqn_lta_aux/aux_control/best_5g_gamma0.9/"
     },
    {"label": "FTA+XY",
     "control": "data/output/result/gridhard/linear_vf/control/early_stop/different_task/fine_tune/dqn_lta_aux/info/best/",
     "online_measure": "data/output/result/gridhard/linear_vf/online_property/dqn_lta_aux/info/best/"
     },
    {"label": "FTA+Decoder",
     "control": "data/output/result/gridhard/linear_vf/control/early_stop/different_task/fine_tune/dqn_lta_aux/input_decoder/best/",
     "online_measure": "data/output/result/gridhard/linear_vf/online_property/dqn_lta_aux/input_decoder/best/"
     },
    {"label": "FTA+NAS",
     "control": "data/output/result/gridhard/linear_vf/control/early_stop/different_task/fine_tune/dqn_lta_aux/nas_v2_delta/best/",
     "online_measure": "data/output/result/gridhard/linear_vf/online_property/dqn_lta_aux/nas_v2_delta/best/"
     },
    {"label": "FTA+Reward",
     "control": "data/output/result/gridhard/linear_vf/control/early_stop/different_task/fine_tune/dqn_lta_aux/reward/best/",
     "online_measure": "data/output/result/gridhard/linear_vf/online_property/dqn_lta_aux/reward/best/"
     },
    {"label": "FTA+SF",
     "control": "data/output/result/gridhard/linear_vf/control/early_stop/different_task/fine_tune/dqn_lta_aux/successor_as/best/",
     "online_measure": "data/output/result/gridhard/linear_vf/online_property/dqn_lta_aux/successor_as/best/"
     },
    {"label": "Random",
     "control": "data/output/result/gridhard/linear_vf/control/baseline/different_task/fine_tune/random/best/",
     "online_measure": "data/output/result/gridhard/linear_vf/fixrep_property/random/best",
     },
]

gh_same_last = [
    {"label": "ReLU", "property": "data/output/test/",
     "control": "data/output/test/gridhard/control/last/same_task/fix_rep/dqn/best/",
     "online_measure": "data/output/test/gridhard/online_property/dqn/best/"
     },
    {"label": "ReLU+VirtualVF1",
     "control": "data/output/test/gridhard/control/last/same_task/fix_rep/dqn_aux/aux_control/best_1g/",
     "online_measure": "data/output/test/gridhard/online_property/dqn_aux/freq8/aux_control/best_1g/"
     },
    {"label": "ReLU+VirtualVF5",
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
    {"label": "FTA+VirtualVF1",
     "control": "data/output/test/gridhard/control/last/same_task/fix_rep/dqn_lta_aux/aux_control/best_1g/",
     "online_measure": "data/output/test/gridhard/online_property/dqn_lta_aux/aux_control/best_1g/"
     },
    {"label": "FTA+VirtualVF5",
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
    {"label": "ReLU+VirtualVF1",
     "control": "data/output/test/gridhard/control/last/similar_task/fix_rep/dqn_aux/aux_control/best_1g/",
     "online_measure": "data/output/test/gridhard/online_property/dqn_aux/freq8/aux_control/best_1g/"
     },
    {"label": "ReLU+VirtualVF5",
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
    {"label": "FTA+VirtualVF1",
     "control": "data/output/test/gridhard/control/last/similar_task/fix_rep/dqn_lta_aux/aux_control/best_1g/",
     "online_measure": "data/output/test/gridhard/online_property/dqn_lta_aux/aux_control/best_1g/"
     },
    {"label": "FTA+VirtualVF5",
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
    {"label": "ReLU+VirtualVF1",
     "control": "data/output/test/gridhard/control/last/different_task/fix_rep/dqn_aux/aux_control/best_1g/",
     "online_measure": "data/output/test/gridhard/online_property/dqn_aux/freq8/aux_control/best_1g/"
     },
    {"label": "ReLU+VirtualVF5",
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
    {"label": "FTA+VirtualVF1",
     "control": "data/output/test/gridhard/control/last/different_task/fix_rep/dqn_lta_aux/aux_control/best_1g/",
     "online_measure": "data/output/test/gridhard/online_property/dqn_lta_aux/aux_control/best_1g/"
     },
    {"label": "FTA+VirtualVF5",
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
    {"label": "ReLU+VirtualVF1",
     "control": "data/output/test/gridhard/control/last/different_task/fine_tune/dqn_aux/aux_control/best_1g/",
     "online_measure": "data/output/test/gridhard/online_property/dqn_aux/freq8/aux_control/best_1g/"
     },
    {"label": "ReLU+VirtualVF5",
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
    {"label": "FTA+VirtualVF1",
     "control": "data/output/test/gridhard/control/last/different_task/fine_tune/dqn_lta_aux/aux_control/best_1g/",
     "online_measure": "data/output/test/gridhard/online_property/dqn_lta_aux/aux_control/best_1g/"
     },
    {"label": "FTA+VirtualVF5",
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
#     {"label": "ReLU+VirtualVF1",
     # "control": "data/output/test/gridhard/control/last/different_task/fix_rep/dqn_aux/aux_control/best_1g/",
     # "online_measure": "data/output/test/gridhard/online_property/dqn_aux/freq8/aux_control/best_1g/"
     # },
    # {"label": "ReLU+VirtualVF5",
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
#     {"label": "LTA+VirtualVF1",
     # "control": "data/output/test/gridhard/control/last/different_task/fix_rep/dqn_lta_aux/aux_control/best_1g/",
     # "online_measure": "data/output/test/gridhard/online_property/dqn_lta_aux/aux_control/best_1g/"
     # },
    # {"label": "LTA+VirtualVF5",
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
#     {"label": "ReLU+VirtualVF1",
     # "control": "data/output/test/gridhard/control/last/similar_task/fix_rep/dqn_aux/aux_control/best_1g/",
     # "online_measure": "data/output/test/gridhard/online_property/dqn_aux/freq8/aux_control/best_1g/"
     # },
    # {"label": "ReLU+VirtualVF5",
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
    # {"label": "LTA+VirtualVF1",
     # "control": "data/output/test/gridhard/control/last/similar_task/fix_rep/dqn_lta_aux/aux_control/best_1g/",
     # "online_measure": "data/output/test/gridhard/online_property/dqn_lta_aux/aux_control/best_1g/"
     # },
    # {"label": "LTA+VirtualVF5",
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
#     {"label": "ReLU+VirtualVF1",
     # "control": "data/output/test/gridhard/control/last/different_task/fix_rep/dqn_aux/aux_control/best_1g/",
     # "online_measure": "data/output/test/gridhard/online_property/dqn_aux/freq8/aux_control/best_1g/"
     # },
    # {"label": "ReLU+VirtualVF5",
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
#     {"label": "LTA+VirtualVF1",
     # "control": "data/output/test/gridhard/control/last/different_task/fix_rep/dqn_lta_aux/aux_control/best_1g/",
     # "online_measure": "data/output/test/gridhard/online_property/dqn_lta_aux/aux_control/best_1g/"
     # },
    # {"label": "LTA+VirtualVF5",
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
#     {"label": "ReLU+VirtualVF1",
     # "control": "data/output/test/gridhard/control/last/similar_task/fix_rep/dqn_aux/aux_control/best_1g/",
     # "online_measure": "data/output/test/gridhard/online_property/dqn_aux/freq8/aux_control/best_1g/"
     # },
    # {"label": "ReLU+VirtualVF5",
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
    # {"label": "LTA+VirtualVF1",
     # "control": "data/output/test/gridhard/control/last/similar_task/fix_rep/dqn_lta_aux/aux_control/best_1g/",
     # "online_measure": "data/output/test/gridhard/online_property/dqn_lta_aux/aux_control/best_1g/"
     # },
    # {"label": "LTA+VirtualVF5",
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
#     {"label": "FTA",
     # "control": "data/output/test/picky_eater/representation/dqn_aux/aux_control/sweep_1f/"
#      },
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

crgb_online_sweep_2f_representation = [
    {"label": "ReLU+Control",
     "control": "data/output/test/picky_eater/representation/dqn_aux/aux_control/sweep_2f",
     "best": "1"
     },
#    {"label": "FTA+Control",
    # "control": "data/output/test/picky_eater/representation/dqn_lta_aux/aux_control/best_2f",
    # "best": "2"
#      },
   {"label": "FTA+Control",
    "control": "data/output/test/picky_eater/representation/dqn_lta_aux/aux_control/sweep_2f",
    "best": "1"
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


pe_trans_sweep_2f =  [
    {"label": "FTA+Control",
     "control": "data/output/test/picky_eater/control_test/decayep/last/different_task/fix_rep/dqn_lta_aux/aux_control/sweep_2f",
     "best": '5'
     },
]

pe_trans_best_2f =  [
    {"label": "FTA+Control",
     "control": "data/output/test/picky_eater/control_test/decayep/last/different_task/fix_rep/dqn_lta_aux/aux_control/best_2f",
     },
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

switch_color_representation_sweep = [
#     {"label": "ReLU",
     # "control": "data/output/test/picky_eater_color_switch/representation/dqn/sweep_3f/",
#      },
    {"label": "ReLU+Control",
     "control": "data/output/test/picky_eater_color_switch/representation/dqn_aux/aux_control/sweep_3f/",
     },
    ]

switch_color_representation_best = [
#     {"label": "ReLU+Control",
     # "control": "data/output/test/picky_eater_color_switch/representation/dqn/best_3f/",
#      },
    {"label": "ReLU+Control",
     "control": "data/output/test/picky_eater_color_switch/representation/dqn_aux/aux_control/best_3f/",
     },
    ]

test_v7_representation_sweep = [
#     {"label": "ReLU+Control",
     # "control": "data/output/test/picky_eater_color_switch/representation/dqn/best_3f/",
#      },
    {"label": "ReLU+Control",
     "control": "data/output/test_v7/picky_eater/representation/dqn_aux/aux_control/sweep_3f/",
     },
    {"label": "FTA+Control",
     "control": "data/output/test_v7/picky_eater/representation/dqn_lta_aux/aux_control/sweep_3f/",
     },
    ]

test_v8_representation_sweep = [
    {"label": "ReLU+Control",
     "control": "data/output/test_v8/picky_eater/representation/dqn_aux/aux_control/sweep_3f/",
     },
    {"label": "FTA+Control",
     "control": "data/output/test_v8/picky_eater/representation/dqn_lta_aux/aux_control/sweep_3f/",
     },
    ]


test_v7_representation_sweep_v2 = [
#     {"label": "ReLU+Control",
     # "control": "data/output/test/picky_eater_color_switch/representation/dqn/best_3f/",
#      },
#     {"label": "ReLU+Control",
     # "control": "data/output/test_v7/picky_eater/representation/dqn_aux/aux_control/sweep_3f_v2/",
     # },
    # {"label": "FTA+Control",
     # "control": "data/output/test_v7/picky_eater/representation/dqn_lta_aux/aux_control/sweep_3f_v2/",
     # },
     {"label": "FTA+Control",
     "control": "data/output/test_v7/picky_eater/representation/dqn_lta_aux/aux_control/best_3f/",
     },
    ]

test_v7_control_sweep = [
#      {"label": "ReLU+Control",
       # "control": "data/output/test_v7/picky_eater/control/early_stop/different_task/fix_rep/dqn_aux/aux_control/sweep_3f/",
#        },
#    {"label": "FTA+Control",
#     "control": "data/output/test_v7/picky_eater/control/early_stop/different_task/fix_rep/dqn_lta_aux/aux_control/sweep_3f/",
#     },    
    {"label": "FTA+Control",
     "control": "data/output/test_v7/picky_eater/control/early_stop/different_task/fix_rep/dqn_lta_aux/aux_control/sweep_3f_v1/",
     },
    ]

test_v8_control_sweep = [
      {"label": "ReLU+Control",
       "control": "data/output/test_v8/picky_eater/control/early_stop/different_task/fix_rep/dqn_aux/aux_control/sweep_3f/",
        },
   {"label": "FTA+Control",
     "control": "data/output/test_v8/picky_eater/control/early_stop/different_task/fix_rep/dqn_lta_aux/aux_control/sweep_3f/",
     },
    ]



switch_color_control_sweep= [
#     {"label": "ReLU",
     # "control": "data/output/test/picky_eater_color_switch/control/last/different_task/fix_rep/dqn/sweep_3f",
     # "best": '3'
#     },
     {"label": "ReLU+Control",
     "control": "data/output/test/picky_eater_color_switch/control/last/different_task/fix_rep/dqn_aux/aux_control/sweep_3f",
     "best": '3'
    },
    ]

switch_color_control_best= [
    {"label": "ReLU",
     "control": "data/output/test/picky_eater_color_switch/control/last/different_task/fix_rep/dqn_aux/aux_control/best_3f",
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
     "control": "data/output/test/picky_eater/control_test/decayep/last/different_task/fix_rep/dqn_lta_aux/aux_control/best_3f/",
     },
    {"label": "Scratch",
     "control": "data/output/result/picky_eater/linear_vf/representation/dqn/best_3f/",
     },
]



#pe_rep_result = [
#    {"label": "ReLU", "property": "data/output/test/",
     # "control": "data/output/test/picky_eater/control/early_stop/different_task/fine_tune/dqn/best/",
#     "control": "data/output/result/picky_eater/nonlinear_vf/representation/dqn/sweep_3f",
     # },
#     {"label": "ReLU+Control",
     # "control": "data/output/test/picky_eater/control/early_stop/different_task/fine_tune/dqn_aux/aux_control/best/",
     # "online_measure": "data/output/test/picky_eater/online_property/dqn_aux/aux_control/best/",
     # },
    # {"label": "ReLU+XY",
     # "control": "data/output/test/picky_eater/control/early_stop/different_task/fine_tune/dqn_aux/info/best_xy/",
     # "online_measure": "data/output/test/picky_eater/online_property/dqn_aux/info/best_xy/",
     # },
    # {"label": "ReLU+Decoder",
     # "control": "data/output/test/picky_eater/control/early_stop/different_task/fine_tune/dqn_aux/input_decoder/best/",
     # "online_measure": "data/output/test/picky_eater/online_property/dqn_aux/input_decoder/best/",
     # },
    # {"label": "ReLU+NAS",
     # "control": "data/output/test/picky_eater/control/early_stop/different_task/fine_tune/dqn_aux/nas_v2_delta/best/",
     # "online_measure": "data/output/test/picky_eater/online_property/dqn_aux/nas_v2_delta/best/"
     # },
    # {"label": "ReLU+Reward",
     # "control": "data/output/test/picky_eater/control/early_stop/different_task/fine_tune/dqn_aux/reward/best/",
     # "online_measure": "data/output/test/picky_eater/online_property/dqn_aux/reward/best/"
     # },
    # {"label": "ReLU+SF",
     # "control": "data/output/test/picky_eater/control/early_stop/different_task/fine_tune/dqn_aux/successor_as/best/",
     # "online_measure": "data/output/test/picky_eater/online_property/dqn_aux/successor_as/best/"
     # },
    # {"label": "FTA eta=2",
     # "control": "data/output/test/picky_eater/control/early_stop/different_task/fine_tune/dqn_lta/best/",
     # "online_measure": "data/output/test/picky_eater/online_property/dqn_lta/best/"
     # },
    # {"label": "FTA+Control",
     # "control": "data/output/test/picky_eater/control/early_stop/different_task/fine_tune/dqn_lta_aux/aux_control/best/",
     # "online_measure": "data/output/test/picky_eater/online_property/dqn_lta_aux/aux_control/best/"
     # },
    # {"label": "FTA+XY",
     # "control": "data/output/test/picky_eater/control/early_stop/different_task/fine_tune/dqn_lta_aux/info/best_xy/",
     # "online_measure": "data/output/test/picky_eater/online_property/dqn_lta_aux/info/best_xy/"
     # },
    # {"label": "FTA+Decoder",
     # "control": "data/output/test/picky_eater/control/early_stop/different_task/fine_tune/dqn_lta_aux/input_decoder/best/",
     # "online_measure": "data/output/test/picky_eater/online_property/dqn_lta_aux/input_decoder/best/"
     # },
    # {"label": "FTA+NAS",
     # "control": "data/output/test/picky_eater/control/early_stop/different_task/fine_tune/dqn_lta_aux/nas_v2_delta/best/",
     # "online_measure": "data/output/test/picky_eater/online_property/dqn_lta_aux/nas_v2_delta/best/"
     # },
    # {"label": "FTA+Reward",
     # "control": "data/output/test/picky_eater/control/early_stop/different_task/fine_tune/dqn_lta_aux/reward/best/",
     # "online_measure": "data/output/test/picky_eater/online_property/dqn_lta_aux/reward/best/"
     # },
    # {"label": "FTA+SF",
     # "control": "data/output/test/picky_eater/control/early_stop/different_task/fine_tune/dqn_lta_aux/successor_as/best/",
     # "online_measure": "data/output/test/picky_eater/online_property/dqn_lta_aux/successor_as/best/"
#      },
    # {"label": "Random",
     # "control": "data/output/test/picky_eater/control/baseline/different_task/fine_tune/random/best/",
     # "online_measure": "data/output/test/picky_eater/fixrep_property/random/best",
    #  },
# ]

pe_rep_sweep = [
    {"label": "ReLU",
     "control": "data/output/result/picky_eater/nonlinear_vf/representation/dqn/sweep_3f/",
     "best": "2"
     },
    {"label": "ReLU+Control",
     "control": "data/output/result/picky_eater/nonlinear_vf/representation/dqn_aux/aux_control/sweep_3f/",
     "best": "2"
     },
     {"label": "ReLU+XY",
      "control": "data/output/result/picky_eater/nonlinear_vf/representation/dqn_aux/info/sweep_3f_xy/",
      },
      {"label": "ReLU+Color",
      "control": "data/output/result/picky_eater/nonlinear_vf/representation/dqn_aux/info/sweep_3f_color/",
     "best": "2"
      },   
      {"label": "ReLU+Decoder",
     "control": "data/output/result/picky_eater/nonlinear_vf/representation/dqn_aux/input_decoder/sweep_3f/",
     "best": "1"
     },
    {"label": "ReLU+NAS",
     "control": "data/output/result/picky_eater/nonlinear_vf/representation/dqn_aux/nas_v2_delta/sweep_3f/",
     "best": "2"
     },
    {"label": "ReLU+Reward",
     "control": "data/output/result/picky_eater/nonlinear_vf/representation/dqn_aux/reward/sweep_3f/",
     "best": "2"
     },
    {"label": "ReLU+SF",
     "control": "data/output/result/picky_eater/nonlinear_vf/representation/dqn_aux/successor_as/sweep_3f/",
     "best": "2"
     },
    {"label": "FTA",
     "control": "data/output/result/picky_eater/nonlinear_vf/representation/dqn_lta/sweep_3f/",
     "best": "2"
     },
    {"label": "FTA+Control",
     "control": "data/output/result/picky_eater/nonlinear_vf/representation/dqn_lta_aux/aux_control/sweep_3f/",
     "best": "2"
     },
     {"label": "FTA+XY",
      "control": "data/output/result/picky_eater/nonlinear_vf/representation/dqn_lta_aux/info/sweep_3f_xy/",
     "best": "2"
      },
      {"label": "FTA+Color",
      "control": "data/output/result/picky_eater/nonlinear_vf/representation/dqn_lta_aux/info/sweep_3f_color/",
     "best": "2"
      },
      {"label": "FTA+Decoder",
     "control": "data/output/result/picky_eater/nonlinear_vf/representation/dqn_lta_aux/input_decoder/sweep_3f/",
     "best": "2"
     },
    {"label": "FTA+NAS",
     "control": "data/output/result/picky_eater/nonlinear_vf/representation/dqn_lta_aux/nas_v2_delta/sweep_3f/",
     "best": "2"
     },
    {"label": "FTA+Reward",
     "control": "data/output/result/picky_eater/nonlinear_vf/representation/dqn_lta_aux/reward/sweep_3f/",
     "best": "2"
     },
    {"label": "FTA+SF",
     "control": "data/output/result/picky_eater/nonlinear_vf/representation/dqn_lta_aux/successor_as/sweep_3f/",
     "best": "2"
     }
]


pe_rep_best = [
    {"label": "ReLU",
     "control": "data/output/result/picky_eater/nonlinear_vf/representation/dqn/best_3f/",
     },
    {"label": "ReLU+Control",
     "control": "data/output/result/picky_eater/nonlinear_vf/representation/dqn_aux/aux_control/best_3f/",
     },
      {"label": "ReLU+XY",
      "control": "data/output/result/picky_eater/nonlinear_vf/representation/dqn_aux/info/best_3f_xy/",
      },
      {"label": "ReLU+Color",
      "control": "data/output/result/picky_eater/nonlinear_vf/representation/dqn_aux/info/best_3f_color/",
      },   
      {"label": "ReLU+Decoder",
     "control": "data/output/result/picky_eater/nonlinear_vf/representation/dqn_aux/input_decoder/best_3f/",
     },
    {"label": "ReLU+NAS",
     "control": "data/output/result/picky_eater/nonlinear_vf/representation/dqn_aux/nas_v2_delta/best_3f/",
     },
    {"label": "ReLU+Reward",
     "control": "data/output/result/picky_eater/nonlinear_vf/representation/dqn_aux/reward/best_3f/",
     },
    {"label": "ReLU+SF",
     "control": "data/output/result/picky_eater/nonlinear_vf/representation/dqn_aux/successor_as/best_3f/",
     },
    {"label": "FTA",
     "control": "data/output/result/picky_eater/nonlinear_vf/representation/dqn_lta/best_3f/",
     },
   {"label": "FTA+Control",
    "control": "data/output/result/picky_eater/nonlinear_vf/representation/dqn_lta_aux/aux_control/best_3f/",
    },
     {"label": "FTA+XY",
      "control": "data/output/result/picky_eater/nonlinear_vf/representation/dqn_lta_aux/info/best_3f_xy/",
      },
      {"label": "FTA+Color",
      "control": "data/output/result/picky_eater/nonlinear_vf/representation/dqn_lta_aux/info/best_3f_color/",
      },
      {"label": "FTA+Decoder",
     "control": "data/output/result/picky_eater/nonlinear_vf/representation/dqn_lta_aux/input_decoder/best_3f/",
     },
    {"label": "FTA+NAS",
     "control": "data/output/result/picky_eater/nonlinear_vf/representation/dqn_lta_aux/nas_v2_delta/best_3f/",
     },
    {"label": "FTA+Reward",
     "control": "data/output/result/picky_eater/nonlinear_vf/representation/dqn_lta_aux/reward/best_3f/",
     },
    {"label": "FTA+SF",
     "control": "data/output/result/picky_eater/nonlinear_vf/representation/dqn_lta_aux/successor_as/best_3f/",
     }
]

pe_transfer_sweep = [
    {"label": "ReLU",
     "control": "data/output/result/picky_eater/nonlinear_vf/control/early_stop/different_task/fix_rep/dqn/sweep_3f/",
     "best": "5"
     },
   {"label": "ReLU+Control",
    "control": "data/output/result/picky_eater/nonlinear_vf/control/early_stop/different_task/fix_rep/dqn_aux/aux_control/sweep_3f/",
    "best": "7"
    },
     {"label": "ReLU+XY",
      "control": "data/output/result/picky_eater/nonlinear_vf/control/early_stop/different_task/fix_rep/dqn_aux/info/sweep_3f_xy/",
     "best": "6"
      },
      {"label": "ReLU+Color",
      "control": "data/output/result/picky_eater/nonlinear_vf/control/early_stop/different_task/fix_rep/dqn_aux/info/sweep_3f_color/",
     "best": "6"
      },   
      {"label": "ReLU+Decoder",
     "control": "data/output/result/picky_eater/nonlinear_vf/control/early_stop/different_task/fix_rep/dqn_aux/input_decoder/sweep_3f/",
     "best": "4"
     },
    {"label": "ReLU+NAS",
     "control": "data/output/result/picky_eater/nonlinear_vf/control/early_stop/different_task/fix_rep/dqn_aux/nas_v2_delta/sweep_3f/",
     "best": "6"
     },
    {"label": "ReLU+Reward",
     "control": "data/output/result/picky_eater/nonlinear_vf/control/early_stop/different_task/fix_rep/dqn_aux/reward/sweep_3f/",
     "best": "5"
     },
    {"label": "ReLU+SF",
     "control": "data/output/result/picky_eater/nonlinear_vf/control/early_stop/different_task/fix_rep/dqn_aux/successor_as/sweep_3f/",
     "best": "6"
     },
    {"label": "FTA",
     "control": "data/output/result/picky_eater/nonlinear_vf/control/early_stop/different_task/fix_rep/dqn_lta/sweep_3f/",
     "best": "7"
     },
    {"label": "FTA+Control",
     "control": "data/output/result/picky_eater/nonlinear_vf/control/early_stop/different_task/fix_rep/dqn_lta_aux/aux_control/sweep_3f/",
     "best": "4"
     },
     {"label": "FTA+XY",
      "control": "data/output/result/picky_eater/nonlinear_vf/control/early_stop/different_task/fix_rep/dqn_lta_aux/info/sweep_3f_xy/",
     "best": "7"
      },
      {"label": "FTA+Color",
      "control": "data/output/result/picky_eater/nonlinear_vf/control/early_stop/different_task/fix_rep/dqn_lta_aux/info/sweep_3f_color/",
     "best": "4"
      },
      {"label": "FTA+Decoder",
     "control": "data/output/result/picky_eater/nonlinear_vf/control/early_stop/different_task/fix_rep/dqn_lta_aux/input_decoder/sweep_3f/",
     "best": "4"
     },
    {"label": "FTA+NAS",
     "control": "data/output/result/picky_eater/nonlinear_vf/control/early_stop/different_task/fix_rep/dqn_lta_aux/nas_v2_delta/sweep_3f/",
     "best": "7"
     },
    {"label": "FTA+Reward",
     "control": "data/output/result/picky_eater/nonlinear_vf/control/early_stop/different_task/fix_rep/dqn_lta_aux/reward/sweep_3f/",
     "best": "7"
     },
    {"label": "FTA+SF",
     "control": "data/output/result/picky_eater/nonlinear_vf/control/early_stop/different_task/fix_rep/dqn_lta_aux/successor_as/sweep_3f/",
     "best": "7"
     }
]



pe_t_sweep_v2 = [
  {"label": "FTA+Decoder",
     "control": "data/output/result/picky_eater/nonlinear_vf/control/early_stop/different_task/fix_rep/dqn_lta_aux/input_decoder/sweep_3f/",
     "best": "4"
   },
 #  {"label": "ReLU+Control",
    # "control": "data/output/result/picky_eater/nonlinear_vf/control/early_stop/different_task/fix_rep/dqn_aux/aux_control/sweep_3f/",
    # "best": "2"
    # },
   # {"label": "FTA",
     # "control": "data/output/result/picky_eater/nonlinear_vf/control/early_stop/different_task/fix_rep/dqn_lta/sweep_3f_v2/",
     # "best": "7"
     # },
    # {"label": "FTA+Control",
     # "control": "data/output/result/picky_eater/nonlinear_vf/control/early_stop/different_task/fix_rep/dqn_lta_aux/aux_control/sweep_3f/",
     # "best": "4"
     # },
     # {"label": "FTA+XY",
      # "control": "data/output/result/picky_eater/nonlinear_vf/control/early_stop/different_task/fix_rep/dqn_lta_aux/info/sweep_3f_xy_v2/",
     # "best": "7"
      # },
    # {"label": "FTA+NAS",
     # "control": "data/output/result/picky_eater/nonlinear_vf/control/early_stop/different_task/fix_rep/dqn_lta_aux/nas_v2_delta/sweep_3f_v2/",
     # "best": "7"
     # },
    # {"label": "FTA+Reward",
     # "control": "data/output/result/picky_eater/nonlinear_vf/control/early_stop/different_task/fix_rep/dqn_lta_aux/reward/sweep_3f_v2/",
     # "best": "7"
     # },
    # {"label": "FTA+SF",
     # "control": "data/output/result/picky_eater/nonlinear_vf/control/early_stop/different_task/fix_rep/dqn_lta_aux/successor_as/sweep_3f_v2/",
     # "best": "7"
  #    }
]


pe_transfer_sweep_v2 = [
  {"label": "ReLU+Control",
    "control": "data/output/result/picky_eater/nonlinear_vf/control/early_stop/different_task/fix_rep/dqn_aux/aux_control/sweep_3f/",
    "best": "2"
    },
   {"label": "FTA",
     "control": "data/output/result/picky_eater/nonlinear_vf/control/early_stop/different_task/fix_rep/dqn_lta/sweep_3f_v2/",
     "best": "7"
     },
    {"label": "FTA+Control",
     "control": "data/output/result/picky_eater/nonlinear_vf/control/early_stop/different_task/fix_rep/dqn_lta_aux/aux_control/sweep_3f/",
     "best": "4"
     },
     {"label": "FTA+XY",
      "control": "data/output/result/picky_eater/nonlinear_vf/control/early_stop/different_task/fix_rep/dqn_lta_aux/info/sweep_3f_xy_v2/",
     "best": "7"
      },
    {"label": "FTA+NAS",
     "control": "data/output/result/picky_eater/nonlinear_vf/control/early_stop/different_task/fix_rep/dqn_lta_aux/nas_v2_delta/sweep_3f_v2/",
     "best": "7"
     },
    {"label": "FTA+Reward",
     "control": "data/output/result/picky_eater/nonlinear_vf/control/early_stop/different_task/fix_rep/dqn_lta_aux/reward/sweep_3f_v2/",
     "best": "7"
     },
    {"label": "FTA+SF",
     "control": "data/output/result/picky_eater/nonlinear_vf/control/early_stop/different_task/fix_rep/dqn_lta_aux/successor_as/sweep_3f_v2/",
     "best": "7"
     }
]


pe_transfer_sweep_same_v2 = [
  {"label": "ReLU+Control",
    "control": "data/output/result/picky_eater/nonlinear_vf/control/early_stop/same_task/fix_rep/dqn_aux/aux_control/sweep_3f/",
    "best": "4"
    },
    {"label": "FTA+Control",
     "control": "data/output/result/picky_eater/nonlinear_vf/control/early_stop/same_task/fix_rep/dqn_lta_aux/aux_control/sweep_3f/",
     "best": "4"
     },
]

pe_transfer_sweep_same = [
    {"label": "ReLU",
     "control": "data/output/result/picky_eater/nonlinear_vf/control/early_stop/same_task/fix_rep/dqn/sweep_3f/",
     "best": "3"
     },
   {"label": "ReLU+Control",
    "control": "data/output/result/picky_eater/nonlinear_vf/control/early_stop/same_task/fix_rep/dqn_aux/aux_control/sweep_3f/",
    "best": "4"
    },
     {"label": "ReLU+XY",
      "control": "data/output/result/picky_eater/nonlinear_vf/control/early_stop/same_task/fix_rep/dqn_aux/info/sweep_3f_xy/",
     "best": "3"
      },
      {"label": "ReLU+Color",
      "control": "data/output/result/picky_eater/nonlinear_vf/control/early_stop/same_task/fix_rep/dqn_aux/info/sweep_3f_color/",
     "best": "4"
      },   
      {"label": "ReLU+Decoder",
     "control": "data/output/result/picky_eater/nonlinear_vf/control/early_stop/same_task/fix_rep/dqn_aux/input_decoder/sweep_3f/",
     "best": "3"
     },
    {"label": "ReLU+NAS",
     "control": "data/output/result/picky_eater/nonlinear_vf/control/early_stop/same_task/fix_rep/dqn_aux/nas_v2_delta/sweep_3f/",
     "best": "4"
     },
    {"label": "ReLU+Reward",
     "control": "data/output/result/picky_eater/nonlinear_vf/control/early_stop/same_task/fix_rep/dqn_aux/reward/sweep_3f/",
     "best": "4"
     },
    {"label": "ReLU+SF",
     "control": "data/output/result/picky_eater/nonlinear_vf/control/early_stop/same_task/fix_rep/dqn_aux/successor_as/sweep_3f/",
     "best": "3"
     },
    {"label": "FTA",
     "control": "data/output/result/picky_eater/nonlinear_vf/control/early_stop/same_task/fix_rep/dqn_lta/sweep_3f/",
     "best": "4"
     },
    {"label": "FTA+Control",
     "control": "data/output/result/picky_eater/nonlinear_vf/control/early_stop/same_task/fix_rep/dqn_lta_aux/aux_control/sweep_3f/",
     "best": "4"
     },
     {"label": "FTA+XY",
      "control": "data/output/result/picky_eater/nonlinear_vf/control/early_stop/same_task/fix_rep/dqn_lta_aux/info/sweep_3f_xy/",
     "best": "4"
      },
      {"label": "FTA+Color",
      "control": "data/output/result/picky_eater/nonlinear_vf/control/early_stop/same_task/fix_rep/dqn_lta_aux/info/sweep_3f_color/",
     "best": "4"
      },
      {"label": "FTA+Decoder",
     "control": "data/output/result/picky_eater/nonlinear_vf/control/early_stop/same_task/fix_rep/dqn_lta_aux/input_decoder/sweep_3f/",
     "best": "4"
     },
    {"label": "FTA+NAS",
     "control": "data/output/result/picky_eater/nonlinear_vf/control/early_stop/same_task/fix_rep/dqn_lta_aux/nas_v2_delta/sweep_3f/",
     "best": "4"
     },
    {"label": "FTA+Reward",
     "control": "data/output/result/picky_eater/nonlinear_vf/control/early_stop/same_task/fix_rep/dqn_lta_aux/reward/sweep_3f/",
     "best": "4"
     },
    {"label": "FTA+SF",
     "control": "data/output/result/picky_eater/nonlinear_vf/control/early_stop/same_task/fix_rep/dqn_lta_aux/successor_as/sweep_3f/",
     "best": "4"
     }
]


pe_in_rand_sweep = [
    {"label": "Input",
     "control": "data/output/result/picky_eater/nonlinear_vf/control/early_stop/different_task/fix_rep/input/sweep_3f/",
     "best": "4"
     },
       {"label": "Random",
     "control": "data/output/result/picky_eater/nonlinear_vf/control/early_stop/same_task/fix_rep/random/sweep_3f/",
     "best": "3"
     },

]



pe_transfer_best = [
    {"label": "ReLU",
     "control": "data/output/result/picky_eater/nonlinear_vf/control/early_stop/different_task/fix_rep/dqn/best_3f/",
     "best": "2"
     },
    {"label": "ReLU+Control",
     "control": "data/output/result/picky_eater/nonlinear_vf/representation/dqn_aux/aux_control/best_3f/",
     "online_measure": "data/output/result/picky_eater/nonlinear_vf/representation/dqn_aux/aux_control/best_3f/",
     "best": "2"
     },
      {"label": "ReLU+XY",
      "control": "data/output/result/picky_eater/nonlinear_vf/representation/dqn_aux/info/best_3f_xy/",
     "online_measure": "data/output/result/picky_eater/nonlinear_vf/representation/dqn_aux/info/best_3f_xy/",
      },
      {"label": "ReLU+Color",
      "control": "data/output/result/picky_eater/nonlinear_vf/representation/dqn_aux/info/best_3f_color/",
     "online_measure": "data/output/result/picky_eater/nonlinear_vf/representation/dqn_aux/info/best_3f_color/",
      },   
      {"label": "ReLU+Decoder",
     "control": "data/output/result/picky_eater/nonlinear_vf/representation/dqn_aux/input_decoder/best_3f/",
     "online_measure": "data/output/result/picky_eater/nonlinear_vf/representation/dqn_aux/input_decoder/best_3f/",
     },
    {"label": "ReLU+NAS",
     "control": "data/output/result/picky_eater/nonlinear_vf/representation/dqn_aux/nas_v2_delta/best_3f/",
     "online_measure": "data/output/result/picky_eater/nonlinear_vf/representation/dqn_aux/nas_v2_delta/best_3f/",
     },
    {"label": "ReLU+Reward",
     "control": "data/output/result/picky_eater/nonlinear_vf/representation/dqn_aux/reward/best_3f/",
     "online_measure": "data/output/result/picky_eater/nonlinear_vf/representation/dqn_aux/reward/best_3f/",
     },
    {"label": "ReLU+SF",
     "control": "data/output/result/picky_eater/nonlinear_vf/representation/dqn_aux/successor_as/best_3f/",
     "online_measure": "data/output/result/picky_eater/nonlinear_vf/representation/dqn_aux/successor_as/best_3f/",
     },
    {"label": "FTA",
     "control": "data/output/result/picky_eater/nonlinear_vf/representation/dqn_lta/best_3f/",
     "online_measure": "data/output/result/picky_eater/nonlinear_vf/representation/dqn_lta/best_3f/",
     },
   {"label": "FTA+Control",
    "control": "data/output/result/picky_eater/nonlinear_vf/representation/dqn_lta_aux/aux_control/best_3f/",
     "online_measure": "data/output/result/picky_eater/nonlinear_vf/representation/dqn_lta_aux/aux_control/best_3f/",
    "best": "2"
    },
     {"label": "FTA+XY",
      "control": "data/output/result/picky_eater/nonlinear_vf/representation/dqn_lta_aux/info/best_3f_xy/",
     "online_measure": "data/output/result/picky_eater/nonlinear_vf/representation/dqn_lta_aux/aux_control/best_3f/",
     "best": "2"
      },
      {"label": "FTA+Color",
      "control": "data/output/result/picky_eater/nonlinear_vf/representation/dqn_lta_aux/info/best_3f_color/",
     "best": "2"
      },
      {"label": "FTA+Decoder",
     "control": "data/output/result/picky_eater/nonlinear_vf/representation/dqn_lta_aux/input_decoder/best_3f/",
     "best": "2"
     },
    {"label": "FTA+NAS",
     "control": "data/output/result/picky_eater/nonlinear_vf/representation/dqn_lta_aux/nas_v2_delta/best_3f/",
     "best": "2"
     },
    {"label": "FTA+Reward",
     "control": "data/output/result/picky_eater/nonlinear_vf/representation/dqn_lta_aux/reward/best_3f/",
     "best": "2"
     },
    {"label": "FTA+SF",
     "control": "data/output/result/picky_eater/nonlinear_vf/representation/dqn_lta_aux/successor_as/best_3f/",
     "best": "2"
     }
]

pe_transfer_sweep = [
    {"label": "ReLU",
     "control": "data/output/result/picky_eater/nonlinear_vf/control/early_stop/different_task/fix_rep/dqn/sweep_3f/",
     "best": "5"
     },
   {"label": "ReLU+Control",
    "control": "data/output/result/picky_eater/nonlinear_vf/control/early_stop/different_task/fix_rep/dqn_aux/aux_control/sweep_3f/",
    "best": "7"
    },
     {"label": "ReLU+XY",
      "control": "data/output/result/picky_eater/nonlinear_vf/control/early_stop/different_task/fix_rep/dqn_aux/info/sweep_3f_xy/",
     "best": "6"
      },
      {"label": "ReLU+Color",
      "control": "data/output/result/picky_eater/nonlinear_vf/control/early_stop/different_task/fix_rep/dqn_aux/info/sweep_3f_color/",
     "best": "6"
      },   
      {"label": "ReLU+Decoder",
     "control": "data/output/result/picky_eater/nonlinear_vf/control/early_stop/different_task/fix_rep/dqn_aux/input_decoder/sweep_3f/",
     "best": "4"
     },
    {"label": "ReLU+NAS",
     "control": "data/output/result/picky_eater/nonlinear_vf/control/early_stop/different_task/fix_rep/dqn_aux/nas_v2_delta/sweep_3f/",
     "best": "6"
     },
    {"label": "ReLU+Reward",
     "control": "data/output/result/picky_eater/nonlinear_vf/control/early_stop/different_task/fix_rep/dqn_aux/reward/sweep_3f/",
     "best": "5"
     },
    {"label": "ReLU+SF",
     "control": "data/output/result/picky_eater/nonlinear_vf/control/early_stop/different_task/fix_rep/dqn_aux/successor_as/sweep_3f/",
     "best": "6"
     },
    {"label": "FTA",
     "control": "data/output/result/picky_eater/nonlinear_vf/control/early_stop/different_task/fix_rep/dqn_lta/sweep_3f/",
     "best": "7"
     },
    {"label": "FTA+Control",
     "control": "data/output/result/picky_eater/nonlinear_vf/control/early_stop/different_task/fix_rep/dqn_lta_aux/aux_control/sweep_3f/",
     "best": "4"
     },
     {"label": "FTA+XY",
      "control": "data/output/result/picky_eater/nonlinear_vf/control/early_stop/different_task/fix_rep/dqn_lta_aux/info/sweep_3f_xy/",
     "best": "7"
      },
      {"label": "FTA+Color",
      "control": "data/output/result/picky_eater/nonlinear_vf/control/early_stop/different_task/fix_rep/dqn_lta_aux/info/sweep_3f_color/",
     "best": "4"
      },
      {"label": "FTA+Decoder",
     "control": "data/output/result/picky_eater/nonlinear_vf/control/early_stop/different_task/fix_rep/dqn_lta_aux/input_decoder/sweep_3f/",
     "best": "4"
     },
    {"label": "FTA+NAS",
     "control": "data/output/result/picky_eater/nonlinear_vf/control/early_stop/different_task/fix_rep/dqn_lta_aux/nas_v2_delta/sweep_3f/",
     "best": "7"
     },
    {"label": "FTA+Reward",
     "control": "data/output/result/picky_eater/nonlinear_vf/control/early_stop/different_task/fix_rep/dqn_lta_aux/reward/sweep_3f/",
     "best": "7"
     },
    {"label": "FTA+SF",
     "control": "data/output/result/picky_eater/nonlinear_vf/control/early_stop/different_task/fix_rep/dqn_lta_aux/successor_as/sweep_3f/",
     "best": "7"
     }
]



pe_t_sweep_v2 = [
  {"label": "FTA+Decoder",
     "control": "data/output/result/picky_eater/nonlinear_vf/control/early_stop/different_task/fix_rep/dqn_lta_aux/input_decoder/sweep_3f/",
     "best": "4"
   },
 #  {"label": "ReLU+Control",
    # "control": "data/output/result/picky_eater/nonlinear_vf/control/early_stop/different_task/fix_rep/dqn_aux/aux_control/sweep_3f/",
    # "best": "2"
    # },
   # {"label": "FTA",
     # "control": "data/output/result/picky_eater/nonlinear_vf/control/early_stop/different_task/fix_rep/dqn_lta/sweep_3f_v2/",
     # "best": "7"
     # },
    # {"label": "FTA+Control",
     # "control": "data/output/result/picky_eater/nonlinear_vf/control/early_stop/different_task/fix_rep/dqn_lta_aux/aux_control/sweep_3f/",
     # "best": "4"
     # },
     # {"label": "FTA+XY",
      # "control": "data/output/result/picky_eater/nonlinear_vf/control/early_stop/different_task/fix_rep/dqn_lta_aux/info/sweep_3f_xy_v2/",
     # "best": "7"
      # },
    # {"label": "FTA+NAS",
     # "control": "data/output/result/picky_eater/nonlinear_vf/control/early_stop/different_task/fix_rep/dqn_lta_aux/nas_v2_delta/sweep_3f_v2/",
     # "best": "7"
     # },
    # {"label": "FTA+Reward",
     # "control": "data/output/result/picky_eater/nonlinear_vf/control/early_stop/different_task/fix_rep/dqn_lta_aux/reward/sweep_3f_v2/",
     # "best": "7"
     # },
    # {"label": "FTA+SF",
     # "control": "data/output/result/picky_eater/nonlinear_vf/control/early_stop/different_task/fix_rep/dqn_lta_aux/successor_as/sweep_3f_v2/",
     # "best": "7"
  #    }
]


pe_transfer_sweep_v2 = [
  {"label": "ReLU+Control",
    "control": "data/output/result/picky_eater/nonlinear_vf/control/early_stop/different_task/fix_rep/dqn_aux/aux_control/sweep_3f/",
    "best": "2"
    },
   {"label": "FTA",
     "control": "data/output/result/picky_eater/nonlinear_vf/control/early_stop/different_task/fix_rep/dqn_lta/sweep_3f_v2/",
     "best": "7"
     },
    {"label": "FTA+Control",
     "control": "data/output/result/picky_eater/nonlinear_vf/control/early_stop/different_task/fix_rep/dqn_lta_aux/aux_control/sweep_3f/",
     "best": "4"
     },
     {"label": "FTA+XY",
      "control": "data/output/result/picky_eater/nonlinear_vf/control/early_stop/different_task/fix_rep/dqn_lta_aux/info/sweep_3f_xy_v2/",
     "best": "7"
      },
    {"label": "FTA+NAS",
     "control": "data/output/result/picky_eater/nonlinear_vf/control/early_stop/different_task/fix_rep/dqn_lta_aux/nas_v2_delta/sweep_3f_v2/",
     "best": "7"
     },
    {"label": "FTA+Reward",
     "control": "data/output/result/picky_eater/nonlinear_vf/control/early_stop/different_task/fix_rep/dqn_lta_aux/reward/sweep_3f_v2/",
     "best": "7"
     },
    {"label": "FTA+SF",
     "control": "data/output/result/picky_eater/nonlinear_vf/control/early_stop/different_task/fix_rep/dqn_lta_aux/successor_as/sweep_3f_v2/",
     "best": "7"
     }
]


pe_transfer_sweep_same_v2 = [
  {"label": "ReLU+Control",
    "control": "data/output/result/picky_eater/nonlinear_vf/control/early_stop/same_task/fix_rep/dqn_aux/aux_control/sweep_3f/",
    "best": "4"
    },
    {"label": "FTA+Control",
     "control": "data/output/result/picky_eater/nonlinear_vf/control/early_stop/same_task/fix_rep/dqn_lta_aux/aux_control/sweep_3f/",
     "best": "4"
     },
]

pe_transfer_sweep_same = [
    {"label": "ReLU",
     "control": "data/output/result/picky_eater/nonlinear_vf/control/early_stop/same_task/fix_rep/dqn/sweep_3f/",
     "best": "3"
     },
   {"label": "ReLU+Control",
    "control": "data/output/result/picky_eater/nonlinear_vf/control/early_stop/same_task/fix_rep/dqn_aux/aux_control/sweep_3f/",
    "best": "4"
    },
     {"label": "ReLU+XY",
      "control": "data/output/result/picky_eater/nonlinear_vf/control/early_stop/same_task/fix_rep/dqn_aux/info/sweep_3f_xy/",
     "best": "3"
      },
      {"label": "ReLU+Color",
      "control": "data/output/result/picky_eater/nonlinear_vf/control/early_stop/same_task/fix_rep/dqn_aux/info/sweep_3f_color/",
     "best": "4"
      },   
      {"label": "ReLU+Decoder",
     "control": "data/output/result/picky_eater/nonlinear_vf/control/early_stop/same_task/fix_rep/dqn_aux/input_decoder/sweep_3f/",
     "best": "3"
     },
    {"label": "ReLU+NAS",
     "control": "data/output/result/picky_eater/nonlinear_vf/control/early_stop/same_task/fix_rep/dqn_aux/nas_v2_delta/sweep_3f/",
     "best": "4"
     },
    {"label": "ReLU+Reward",
     "control": "data/output/result/picky_eater/nonlinear_vf/control/early_stop/same_task/fix_rep/dqn_aux/reward/sweep_3f/",
     "best": "4"
     },
    {"label": "ReLU+SF",
     "control": "data/output/result/picky_eater/nonlinear_vf/control/early_stop/same_task/fix_rep/dqn_aux/successor_as/sweep_3f/",
     "best": "3"
     },
    {"label": "FTA",
     "control": "data/output/result/picky_eater/nonlinear_vf/control/early_stop/same_task/fix_rep/dqn_lta/sweep_3f/",
     "best": "4"
     },
    {"label": "FTA+Control",
     "control": "data/output/result/picky_eater/nonlinear_vf/control/early_stop/same_task/fix_rep/dqn_lta_aux/aux_control/sweep_3f/",
     "best": "4"
     },
     {"label": "FTA+XY",
      "control": "data/output/result/picky_eater/nonlinear_vf/control/early_stop/same_task/fix_rep/dqn_lta_aux/info/sweep_3f_xy/",
     "best": "4"
      },
      {"label": "FTA+Color",
      "control": "data/output/result/picky_eater/nonlinear_vf/control/early_stop/same_task/fix_rep/dqn_lta_aux/info/sweep_3f_color/",
     "best": "4"
      },
      {"label": "FTA+Decoder",
     "control": "data/output/result/picky_eater/nonlinear_vf/control/early_stop/same_task/fix_rep/dqn_lta_aux/input_decoder/sweep_3f/",
     "best": "4"
     },
    {"label": "FTA+NAS",
     "control": "data/output/result/picky_eater/nonlinear_vf/control/early_stop/same_task/fix_rep/dqn_lta_aux/nas_v2_delta/sweep_3f/",
     "best": "4"
     },
    {"label": "FTA+Reward",
     "control": "data/output/result/picky_eater/nonlinear_vf/control/early_stop/same_task/fix_rep/dqn_lta_aux/reward/sweep_3f/",
     "best": "4"
     },
    {"label": "FTA+SF",
     "control": "data/output/result/picky_eater/nonlinear_vf/control/early_stop/same_task/fix_rep/dqn_lta_aux/successor_as/sweep_3f/",
     "best": "4"
     }
]


pe_in_rand_sweep = [
    {"label": "Input",
     "control": "data/output/result/picky_eater/nonlinear_vf/control/early_stop/different_task/fix_rep/input/sweep_3f/",
     "best": "4"
     },
       {"label": "Random",
     "control": "data/output/result/picky_eater/nonlinear_vf/control/early_stop/same_task/fix_rep/random/sweep_3f/",
     "best": "3"
     },

]



pe_transfer_best_dissimilar = [
#    {"label": "Scratch",
#     "control": "data/output/result/picky_eater/nonlinear_vf/control/early_stop/different_task/fix_rep/scratch/dqn/best_3f/",
#     "online_measure": "data/output/result/picky_eater/nonlinear_vf/representation/dqn/best_3f/",
#     },
    {"label": "ReLU",
     "control": "data/output/result/picky_eater/nonlinear_vf/control/early_stop/different_task/fix_rep/dqn/best_3f/",
     "online_measure": "data/output/result/picky_eater/nonlinear_vf/representation/dqn/best_3f/",
     },
   {"label": "ReLU+Control",
    "control": "data/output/result/picky_eater/nonlinear_vf/control/early_stop/different_task/fix_rep/dqn_aux/aux_control/best_3f/",
     "online_measure": "data/output/result/picky_eater/nonlinear_vf/representation/dqn_aux/aux_control/best_3f/",
    },
     {"label": "ReLU+XY",
      "control": "data/output/result/picky_eater/nonlinear_vf/control/early_stop/different_task/fix_rep/dqn_aux/info/best_3f_xy/",
     "online_measure": "data/output/result/picky_eater/nonlinear_vf/representation/dqn_aux/info/best_3f_xy/",
      },
      {"label": "ReLU+Color",
      "control": "data/output/result/picky_eater/nonlinear_vf/control/early_stop/different_task/fix_rep/dqn_aux/info/best_3f_color/",
      "online_measure": "data/output/result/picky_eater/nonlinear_vf/representation/dqn_aux/info/best_3f_color/",
      },   
      {"label": "ReLU+Decoder",
     "control": "data/output/result/picky_eater/nonlinear_vf/control/early_stop/different_task/fix_rep/dqn_aux/input_decoder/best_3f/",
      "online_measure": "data/output/result/picky_eater/nonlinear_vf/representation/dqn_aux/input_decoder/best_3f/",
     },
    {"label": "ReLU+NAS",
     "control": "data/output/result/picky_eater/nonlinear_vf/control/early_stop/different_task/fix_rep/dqn_aux/nas_v2_delta/best_3f/",
      "online_measure": "data/output/result/picky_eater/nonlinear_vf/representation/dqn_aux/nas_v2_delta/best_3f/",
     },
    {"label": "ReLU+Reward",
     "control": "data/output/result/picky_eater/nonlinear_vf/control/early_stop/different_task/fix_rep/dqn_aux/reward/best_3f/",
      "online_measure": "data/output/result/picky_eater/nonlinear_vf/representation/dqn_aux/reward/best_3f/",
     },
    {"label": "ReLU+SF",
     "control": "data/output/result/picky_eater/nonlinear_vf/control/early_stop/different_task/fix_rep/dqn_aux/successor_as/best_3f/",
      "online_measure": "data/output/result/picky_eater/nonlinear_vf/representation/dqn_aux/successor_as/best_3f/",
     },
    {"label": "FTA",
     "control": "data/output/result/picky_eater/nonlinear_vf/control/early_stop/different_task/fix_rep/dqn_lta/best_3f/",
      "online_measure": "data/output/result/picky_eater/nonlinear_vf/representation/dqn_lta/best_3f/",
     },
    {"label": "FTA+Control",
     "control": "data/output/result/picky_eater/nonlinear_vf/control/early_stop/different_task/fix_rep/dqn_lta_aux/aux_control/best_3f/",
      "online_measure": "data/output/result/picky_eater/nonlinear_vf/representation/dqn_lta_aux/aux_control/best_3f/",
     },
     {"label": "FTA+XY",
      "control": "data/output/result/picky_eater/nonlinear_vf/control/early_stop/different_task/fix_rep/dqn_lta_aux/info/best_3f_xy/",
      "online_measure": "data/output/result/picky_eater/nonlinear_vf/representation/dqn_lta_aux/info/best_3f_xy/",
      },
      {"label": "FTA+Color",
      "control": "data/output/result/picky_eater/nonlinear_vf/control/early_stop/different_task/fix_rep/dqn_lta_aux/info/best_3f_color/",
      "online_measure": "data/output/result/picky_eater/nonlinear_vf/representation/dqn_lta_aux/info/best_3f_color/",
      },
      {"label": "FTA+Decoder",
     "control": "data/output/result/picky_eater/nonlinear_vf/control/early_stop/different_task/fix_rep/dqn_lta_aux/input_decoder/best_3f/",
      "online_measure": "data/output/result/picky_eater/nonlinear_vf/representation/dqn_lta_aux/input_decoder/best_3f/",
     },
    {"label": "FTA+NAS",
     "control": "data/output/result/picky_eater/nonlinear_vf/control/early_stop/different_task/fix_rep/dqn_lta_aux/nas_v2_delta/best_3f/",
      "online_measure": "data/output/result/picky_eater/nonlinear_vf/representation/dqn_lta_aux/nas_v2_delta/best_3f/",
     },
    {"label": "FTA+Reward",
     "control": "data/output/result/picky_eater/nonlinear_vf/control/early_stop/different_task/fix_rep/dqn_lta_aux/reward/best_3f/",
      "online_measure": "data/output/result/picky_eater/nonlinear_vf/representation/dqn_lta_aux/reward/best_3f/",
     },
    {"label": "FTA+SF",
     "control": "data/output/result/picky_eater/nonlinear_vf/control/early_stop/different_task/fix_rep/dqn_lta_aux/successor_as/best_3f/",
      "online_measure": "data/output/result/picky_eater/nonlinear_vf/representation/dqn_lta_aux/successor_as/best_3f/",
     "best": "7"
     },
    {"label": "Input",
     "control": "data/output/result/picky_eater/nonlinear_vf/control/early_stop/different_task/fix_rep/input/best_3f/",
     "best": "4"
    },
    {"label": "Random",
     "control": "data/output/result/picky_eater/nonlinear_vf/control/early_stop/different_task/fix_rep/random/best_3f/",
     "best": "4"
    },
]

pe_transfer_best_similar = [
    {"label": "ReLU",
     "control": "data/output/result/picky_eater/nonlinear_vf/control/early_stop/same_task/fix_rep/dqn/best_3f/",
     "online_measure": "data/output/result/picky_eater/nonlinear_vf/representation/dqn/best_3f/",
     },
   {"label": "ReLU+Control",
    "control": "data/output/result/picky_eater/nonlinear_vf/control/early_stop/same_task/fix_rep/dqn_aux/aux_control/best_3f/",
     "online_measure": "data/output/result/picky_eater/nonlinear_vf/representation/dqn_aux/aux_control/best_3f/",
    },
     {"label": "ReLU+XY",
      "control": "data/output/result/picky_eater/nonlinear_vf/control/early_stop/same_task/fix_rep/dqn_aux/info/best_3f_xy/",
     "online_measure": "data/output/result/picky_eater/nonlinear_vf/representation/dqn_aux/info/best_3f_xy/",
      },
      {"label": "ReLU+Color",
      "control": "data/output/result/picky_eater/nonlinear_vf/control/early_stop/same_task/fix_rep/dqn_aux/info/best_3f_color/",
      "online_measure": "data/output/result/picky_eater/nonlinear_vf/representation/dqn_aux/info/best_3f_color/",
      },   
      {"label": "ReLU+Decoder",
     "control": "data/output/result/picky_eater/nonlinear_vf/control/early_stop/same_task/fix_rep/dqn_aux/input_decoder/best_3f/",
      "online_measure": "data/output/result/picky_eater/nonlinear_vf/representation/dqn_aux/input_decoder/best_3f/",
     },
    {"label": "ReLU+NAS",
     "control": "data/output/result/picky_eater/nonlinear_vf/control/early_stop/same_task/fix_rep/dqn_aux/nas_v2_delta/best_3f/",
      "online_measure": "data/output/result/picky_eater/nonlinear_vf/representation/dqn_aux/nas_v2_delta/best_3f/",
     },
    {"label": "ReLU+Reward",
     "control": "data/output/result/picky_eater/nonlinear_vf/control/early_stop/same_task/fix_rep/dqn_aux/reward/best_3f/",
      "online_measure": "data/output/result/picky_eater/nonlinear_vf/representation/dqn_aux/reward/best_3f/",
     },
    {"label": "ReLU+SF",
     "control": "data/output/result/picky_eater/nonlinear_vf/control/early_stop/same_task/fix_rep/dqn_aux/successor_as/best_3f/",
      "online_measure": "data/output/result/picky_eater/nonlinear_vf/representation/dqn_aux/successor_as/best_3f/",
     },
    {"label": "FTA",
     "control": "data/output/result/picky_eater/nonlinear_vf/control/early_stop/same_task/fix_rep/dqn_lta/best_3f/",
      "online_measure": "data/output/result/picky_eater/nonlinear_vf/representation/dqn_lta/best_3f/",
     },
    {"label": "FTA+Control",
     "control": "data/output/result/picky_eater/nonlinear_vf/control/early_stop/same_task/fix_rep/dqn_lta_aux/aux_control/best_3f/",
      "online_measure": "data/output/result/picky_eater/nonlinear_vf/representation/dqn_lta_aux/aux_control/best_3f/",
     },
     {"label": "FTA+XY",
      "control": "data/output/result/picky_eater/nonlinear_vf/control/early_stop/same_task/fix_rep/dqn_lta_aux/info/best_3f_xy/",
      "online_measure": "data/output/result/picky_eater/nonlinear_vf/representation/dqn_lta_aux/info/best_3f_xy/",
      },
      {"label": "FTA+Color",
      "control": "data/output/result/picky_eater/nonlinear_vf/control/early_stop/same_task/fix_rep/dqn_lta_aux/info/best_3f_color/",
      "online_measure": "data/output/result/picky_eater/nonlinear_vf/representation/dqn_lta_aux/info/best_3f_color/",
      },
      {"label": "FTA+Decoder",
     "control": "data/output/result/picky_eater/nonlinear_vf/control/early_stop/same_task/fix_rep/dqn_lta_aux/input_decoder/best_3f/",
      "online_measure": "data/output/result/picky_eater/nonlinear_vf/representation/dqn_lta_aux/input_decoder/best_3f/",
     },
    {"label": "FTA+NAS",
     "control": "data/output/result/picky_eater/nonlinear_vf/control/early_stop/same_task/fix_rep/dqn_lta_aux/nas_v2_delta/best_3f/",
      "online_measure": "data/output/result/picky_eater/nonlinear_vf/representation/dqn_lta_aux/nas_v2_delta/best_3f/",
     },
    {"label": "FTA+Reward",
     "control": "data/output/result/picky_eater/nonlinear_vf/control/early_stop/same_task/fix_rep/dqn_lta_aux/reward/best_3f/",
      "online_measure": "data/output/result/picky_eater/nonlinear_vf/representation/dqn_lta_aux/reward/best_3f/",
     },
    {"label": "FTA+SF",
     "control": "data/output/result/picky_eater/nonlinear_vf/control/early_stop/same_task/fix_rep/dqn_lta_aux/successor_as/best_3f/",
      "online_measure": "data/output/result/picky_eater/nonlinear_vf/representation/dqn_lta_aux/successor_as/best_3f/",
     "best": "7"
     },
    {"label": "Input",
     "control": "data/output/result/picky_eater/nonlinear_vf/control/early_stop/same_task/fix_rep/input/best_3f/",
     "best": "4"
    },
    {"label": "Random",
     "control": "data/output/result/picky_eater/nonlinear_vf/control/early_stop/same_task/fix_rep/random/best_3f/",
     "best": "4"
    },

]

maze_source_sweep = [
    {"label": "ReLU",
     "control": "data/output/test_v10/maze_multigoal/linear_vf/source_task/dqn/sweep/",
     },
    {"label": "ReLU+XY",
     "control": "data/output/test_v10/maze_multigoal/linear_vf/source_task/dqn_aux/info/sweep/",
     },
    {"label": "ReLU+Decoder",
     "control": "data/output/test_v10/maze_multigoal/linear_vf/source_task/dqn_aux/input_decoder/sweep/",
     },
    {"label": "ReLU+NAS",
     "control": "data/output/test_v10/maze_multigoal/linear_vf/source_task/dqn_aux/nas_v2_delta/sweep/",
     },
    {"label": "ReLU+Reward",
     # "control": "data/output/test_v10/maze_multigoal/linear_vf/source_task/dqn_aux/reward/sweep/",
     "control": "data/output/test_v10/maze_multigoal/linear_vf/source_task/dqn_aux/reward/sweep_smallw/",
     },
    {"label": "ReLU+SF",
     "control": "data/output/test_v10/maze_multigoal/linear_vf/source_task/dqn_aux/successor_as/sweep/",
     },
    {"label": "FTA eta=0.2",
     "control": "data/output/test_v10/maze_multigoal/linear_vf/source_task/dqn_lta/eta_study_0.2_sweep/",
     },
    {"label": "FTA eta=0.4",
     "control": "data/output/test_v10/maze_multigoal/linear_vf/source_task/dqn_lta/eta_study_0.4_sweep/",
     },
    {"label": "FTA eta=0.6",
     "control": "data/output/test_v10/maze_multigoal/linear_vf/source_task/dqn_lta/eta_study_0.6_sweep/",
     },
    {"label": "FTA eta=0.8",
     "control": "data/output/test_v10/maze_multigoal/linear_vf/source_task/dqn_lta/eta_study_0.8_sweep/",
     },
    {"label": "FTA+XY",
     "control": "data/output/test_v10/maze_multigoal/linear_vf/source_task/dqn_lta_aux/info/sweep/",
     },
    {"label": "FTA+Decoder",
     "control": "data/output/test_v10/maze_multigoal/linear_vf/source_task/dqn_lta_aux/input_decoder/sweep/",
     },
    {"label": "FTA+NAS",
     "control": "data/output/test_v10/maze_multigoal/linear_vf/source_task/dqn_lta_aux/nas_v2_delta/sweep/",
     },
    {"label": "FTA+Reward",
     "control": "data/output/test_v10/maze_multigoal/linear_vf/source_task/dqn_lta_aux/reward/sweep/",
     },
    {"label": "FTA+SF",
     "control": "data/output/test_v10/maze_multigoal/linear_vf/source_task/dqn_lta_aux/successor_as/sweep/",
     }
]

maze_source_best = [
    {"label": "ReLU",
     "control": "data/output/test_v10/maze_multigoal/linear_vf/source_task/dqn/best/",
     },
    {"label": "ReLU+XY",
     "control": "data/output/test_v10/maze_multigoal/linear_vf/source_task/dqn_aux/info/best/",
     },
    {"label": "ReLU+Decoder",
     "control": "data/output/test_v10/maze_multigoal/linear_vf/source_task/dqn_aux/input_decoder/best/",
     },
    {"label": "ReLU+NAS",
     "control": "data/output/test_v10/maze_multigoal/linear_vf/source_task/dqn_aux/nas_v2_delta/best/",
     },
    {"label": "ReLU+Reward",
     "control": "data/output/test_v10/maze_multigoal/linear_vf/source_task/dqn_aux/reward/best/",
     },
    {"label": "ReLU+SF",
     "control": "data/output/test_v10/maze_multigoal/linear_vf/source_task/dqn_aux/successor_as/best/",
     },
    {"label": "FTA eta=0.2",
     "control": "data/output/test_v10/maze_multigoal/linear_vf/source_task/dqn_lta/eta_study_0.2_best/",
     },
    {"label": "FTA eta=0.4",
     "control": "data/output/test_v10/maze_multigoal/linear_vf/source_task/dqn_lta/eta_study_0.4_best/",
     },
    {"label": "FTA eta=0.6",
     "control": "data/output/test_v10/maze_multigoal/linear_vf/source_task/dqn_lta/eta_study_0.6_best/",
     },
    {"label": "FTA eta=0.8",
     "control": "data/output/test_v10/maze_multigoal/linear_vf/source_task/dqn_lta/eta_study_0.8_best/",
     },
    {"label": "FTA+XY",
     "control": "data/output/test_v10/maze_multigoal/linear_vf/source_task/dqn_lta_aux/info/best/",
     },
    {"label": "FTA+Decoder",
     "control": "data/output/test_v10/maze_multigoal/linear_vf/source_task/dqn_lta_aux/input_decoder/best/",
     },
    {"label": "FTA+NAS",
     "control": "data/output/test_v10/maze_multigoal/linear_vf/source_task/dqn_lta_aux/nas_v2_delta/best/",
     },
    {"label": "FTA+Reward",
     "control": "data/output/test_v10/maze_multigoal/linear_vf/source_task/dqn_lta_aux/reward/best/",
     },
    {"label": "FTA+SF",
     "control": "data/output/test_v10/maze_multigoal/linear_vf/source_task/dqn_lta_aux/successor_as/best/",
     }
]

maze_target_same_sweep = [
    {"label": "ReLU",
     "control": "data/output/test_v10/maze_multigoal/linear_vf/target_task/same/fix_rep/dqn/sweep/",
     },
    {"label": "ReLU+XY",
     "control": "data/output/test_v10/maze_multigoal/linear_vf/target_task/same/fix_rep/dqn_aux/info/sweep/",
     },
    {"label": "ReLU+Decoder",
     "control": "data/output/test_v10/maze_multigoal/linear_vf/target_task/same/fix_rep/dqn_aux/input_decoder/sweep/",
     },
    {"label": "ReLU+NAS",
     "control": "data/output/test_v10/maze_multigoal/linear_vf/target_task/same/fix_rep/dqn_aux/nas_v2_delta/sweep/",
     },
    {"label": "ReLU+Reward",
     "control": "data/output/test_v10/maze_multigoal/linear_vf/target_task/same/fix_rep/dqn_aux/reward/sweep/",
     },
    {"label": "ReLU+SF",
     "control": "data/output/test_v10/maze_multigoal/linear_vf/target_task/same/fix_rep/dqn_aux/successor_as/sweep/",
     },
    {"label": "FTA eta=0.2",
     "control": "data/output/test_v10/maze_multigoal/linear_vf/target_task/same/fix_rep/dqn_lta/eta_study_0.2_sweep/",
     },
    {"label": "FTA eta=0.4",
     "control": "data/output/test_v10/maze_multigoal/linear_vf/target_task/same/fix_rep/dqn_lta/eta_study_0.4_sweep/",
     },
    {"label": "FTA eta=0.6",
     "control": "data/output/test_v10/maze_multigoal/linear_vf/target_task/same/fix_rep/dqn_lta/eta_study_0.6_sweep/",
     },
    {"label": "FTA eta=0.8",
     "control": "data/output/test_v10/maze_multigoal/linear_vf/target_task/same/fix_rep/dqn_lta/eta_study_0.8_sweep/",
     },
    {"label": "FTA+XY",
     "control": "data/output/test_v10/maze_multigoal/linear_vf/target_task/same/fix_rep/dqn_lta_aux/info/sweep/",
     },
    {"label": "FTA+Decoder",
     "control": "data/output/test_v10/maze_multigoal/linear_vf/target_task/same/fix_rep/dqn_lta_aux/input_decoder/sweep/",
     },
    {"label": "FTA+NAS",
     "control": "data/output/test_v10/maze_multigoal/linear_vf/target_task/same/fix_rep/dqn_lta_aux/nas_v2_delta/sweep/",
     },
    {"label": "FTA+Reward",
     "control": "data/output/test_v10/maze_multigoal/linear_vf/target_task/same/fix_rep/dqn_lta_aux/reward/sweep/",
     },
    {"label": "FTA+SF",
     "control": "data/output/test_v10/maze_multigoal/linear_vf/target_task/same/fix_rep/dqn_lta_aux/successor_as/sweep/",
     },

    {"label": "Random",
     "control": "data/output/test_v10/maze_multigoal/linear_vf/target_task/same/fix_rep/random/sweep/",
     },
    {"label": "Input",
     "control": "data/output/test_v10/maze_multigoal/linear_vf/target_task/same/fix_rep/input/sweep/",
     },
    {"label": "Scratch",
     "control": "data/output/test_v10/maze_multigoal/linear_vf/source_task/dqn/sweep/",
     },
]

maze_target_same_best = [
    {"label": "ReLU",
     "control": "data/output/test_v10/maze_multigoal/linear_vf/target_task/same/fix_rep/dqn/best/",
     },
    {"label": "ReLU+XY",
     "control": "data/output/test_v10/maze_multigoal/linear_vf/target_task/same/fix_rep/dqn_aux/info/best/",
     },
    {"label": "ReLU+Decoder",
     "control": "data/output/test_v10/maze_multigoal/linear_vf/target_task/same/fix_rep/dqn_aux/input_decoder/best/",
     },
    {"label": "ReLU+NAS",
     "control": "data/output/test_v10/maze_multigoal/linear_vf/target_task/same/fix_rep/dqn_aux/nas_v2_delta/best/",
     },
    {"label": "ReLU+Reward",
     "control": "data/output/test_v10/maze_multigoal/linear_vf/target_task/same/fix_rep/dqn_aux/reward/best/",
     },
    {"label": "ReLU+SF",
     "control": "data/output/test_v10/maze_multigoal/linear_vf/target_task/same/fix_rep/dqn_aux/successor_as/best/",
     },
    {"label": "FTA eta=0.2",
     "control": "data/output/test_v10/maze_multigoal/linear_vf/target_task/same/fix_rep/dqn_lta/eta_study_0.2_best/",
     },
    {"label": "FTA eta=0.4",
     "control": "data/output/test_v10/maze_multigoal/linear_vf/target_task/same/fix_rep/dqn_lta/eta_study_0.4_best/",
     },
    {"label": "FTA eta=0.6",
     "control": "data/output/test_v10/maze_multigoal/linear_vf/target_task/same/fix_rep/dqn_lta/eta_study_0.6_best/",
     },
    {"label": "FTA eta=0.8",
     "control": "data/output/test_v10/maze_multigoal/linear_vf/target_task/same/fix_rep/dqn_lta/eta_study_0.8_best/",
     },
    {"label": "FTA+XY",
     "control": "data/output/test_v10/maze_multigoal/linear_vf/target_task/same/fix_rep/dqn_lta_aux/info/best/",
     },
    {"label": "FTA+Decoder",
     "control": "data/output/test_v10/maze_multigoal/linear_vf/target_task/same/fix_rep/dqn_lta_aux/input_decoder/best/",
     },
    {"label": "FTA+NAS",
     "control": "data/output/test_v10/maze_multigoal/linear_vf/target_task/same/fix_rep/dqn_lta_aux/nas_v2_delta/best/",
     },
    {"label": "FTA+Reward",
     "control": "data/output/test_v10/maze_multigoal/linear_vf/target_task/same/fix_rep/dqn_lta_aux/reward/best/",
     },
    {"label": "FTA+SF",
     "control": "data/output/test_v10/maze_multigoal/linear_vf/target_task/same/fix_rep/dqn_lta_aux/successor_as/best/",
     },

    {"label": "Random",
     "control": "data/output/test_v10/maze_multigoal/linear_vf/target_task/same/fix_rep/random/best/",
     },
    {"label": "Input",
     "control": "data/output/test_v10/maze_multigoal/linear_vf/target_task/same/fix_rep/input/best/",
     },
    {"label": "Scratch",
     "control": "data/output/test_v10/maze_multigoal/linear_vf/source_task/dqn/best/",
     },
]

maze_target_diff_sweep = [
    {"label": "ReLU",
     "control": "data/output/test_v10/maze_multigoal/linear_vf/target_task/dissimilar/fix_rep/dqn/sweep/",
     },
    {"label": "ReLU+XY",
     "control": "data/output/test_v10/maze_multigoal/linear_vf/target_task/dissimilar/fix_rep/dqn_aux/info/sweep/",
     },
    {"label": "ReLU+Decoder",
     "control": "data/output/test_v10/maze_multigoal/linear_vf/target_task/dissimilar/fix_rep/dqn_aux/input_decoder/sweep/",
     },
    {"label": "ReLU+NAS",
     "control": "data/output/test_v10/maze_multigoal/linear_vf/target_task/dissimilar/fix_rep/dqn_aux/nas_v2_delta/sweep/",
     },
    {"label": "ReLU+Reward",
     "control": "data/output/test_v10/maze_multigoal/linear_vf/target_task/dissimilar/fix_rep/dqn_aux/reward/sweep/",
     },
    {"label": "ReLU+SF",
     "control": "data/output/test_v10/maze_multigoal/linear_vf/target_task/dissimilar/fix_rep/dqn_aux/successor_as/sweep/",
     },
    {"label": "FTA eta=0.2",
     "control": "data/output/test_v10/maze_multigoal/linear_vf/target_task/dissimilar/fix_rep/dqn_lta/eta_study_0.2_sweep/",
     },
    {"label": "FTA eta=0.4",
     "control": "data/output/test_v10/maze_multigoal/linear_vf/target_task/dissimilar/fix_rep/dqn_lta/eta_study_0.4_sweep/",
     },
    {"label": "FTA eta=0.6",
     "control": "data/output/test_v10/maze_multigoal/linear_vf/target_task/dissimilar/fix_rep/dqn_lta/eta_study_0.6_sweep/",
     },
    {"label": "FTA eta=0.8",
     "control": "data/output/test_v10/maze_multigoal/linear_vf/target_task/dissimilar/fix_rep/dqn_lta/eta_study_0.8_sweep/",
     },
    {"label": "FTA+XY",
     "control": "data/output/test_v10/maze_multigoal/linear_vf/target_task/dissimilar/fix_rep/dqn_lta_aux/info/sweep/",
     },
    {"label": "FTA+Decoder",
     "control": "data/output/test_v10/maze_multigoal/linear_vf/target_task/dissimilar/fix_rep/dqn_lta_aux/input_decoder/sweep/",
     },
    {"label": "FTA+NAS",
     "control": "data/output/test_v10/maze_multigoal/linear_vf/target_task/dissimilar/fix_rep/dqn_lta_aux/nas_v2_delta/sweep/",
     },
    {"label": "FTA+Reward",
     "control": "data/output/test_v10/maze_multigoal/linear_vf/target_task/dissimilar/fix_rep/dqn_lta_aux/reward/sweep/",
     },
    {"label": "FTA+SF",
     "control": "data/output/test_v10/maze_multigoal/linear_vf/target_task/dissimilar/fix_rep/dqn_lta_aux/successor_as/sweep/",
     },

    {"label": "Random",
     "control": "data/output/test_v10/maze_multigoal/linear_vf/target_task/dissimilar/fix_rep/random/sweep/",
     },
    {"label": "Input",
     "control": "data/output/test_v10/maze_multigoal/linear_vf/target_task/dissimilar/fix_rep/input/sweep/",
     },
    {"label": "Scratch",
     "control": "data/output/test_v10/maze_multigoal/linear_vf/target_task/dissimilar/scratch/dqn/sweep/",
     },
]

maze_source_sweep_v11 = [
    {"label": "ReLU",
     "control": "data/output/test_v11/maze_multigoal/linear_vf/source_task/dqn/sweep/",
     },
#    {"label": "ReLU+XY",
#     "control": "data/output/test_v11/maze_multigoal/linear_vf/source_task/dqn_aux/info/sweep/",
#     },
    {"label": "ReLU+Decoder",
     "control": "data/output/test_v11/maze_multigoal/linear_vf/source_task/dqn_aux/input_decoder/sweep/",
     },
    {"label": "ReLU+NAS",
     "control": "data/output/test_v11/maze_multigoal/linear_vf/source_task/dqn_aux/nas_v2_delta/sweep/",
     },
    {"label": "ReLU+Reward",
     "control": "data/output/test_v11/maze_multigoal/linear_vf/source_task/dqn_aux/reward/sweep/",
     },
    {"label": "ReLU+SF",
     "control": "data/output/test_v11/maze_multigoal/linear_vf/source_task/dqn_aux/successor_as/sweep/",
     },
    {"label": "FTA eta=0.2",
     "control": "data/output/test_v11/maze_multigoal/linear_vf/source_task/dqn_lta/eta_study_0.2_sweep/",
     },
    {"label": "FTA eta=0.4",
     "control": "data/output/test_v11/maze_multigoal/linear_vf/source_task/dqn_lta/eta_study_0.4_sweep/",
     },
    {"label": "FTA eta=0.6",
     "control": "data/output/test_v11/maze_multigoal/linear_vf/source_task/dqn_lta/eta_study_0.6_sweep/",
     },
    {"label": "FTA eta=0.8",
     "control": "data/output/test_v11/maze_multigoal/linear_vf/source_task/dqn_lta/eta_study_0.8_sweep/",
     },
    {"label": "FTA+XY",
     "control": "data/output/test_v11/maze_multigoal/linear_vf/source_task/dqn_lta_aux/info/sweep/",
     },
    {"label": "FTA+Decoder",
     "control": "data/output/test_v11/maze_multigoal/linear_vf/source_task/dqn_lta_aux/input_decoder/sweep/",
     },
    {"label": "FTA+NAS",
     "control": "data/output/test_v11/maze_multigoal/linear_vf/source_task/dqn_lta_aux/nas_v2_delta/sweep/",
     },
    {"label": "FTA+Reward",
     "control": "data/output/test_v11/maze_multigoal/linear_vf/source_task/dqn_lta_aux/reward/sweep/",
     },
    {"label": "FTA+SF",
     "control": "data/output/test_v11/maze_multigoal/linear_vf/source_task/dqn_lta_aux/successor_as/sweep/",
     }
]

maze_source_sweep_v12 = [
    {"label": "ReLU",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/source_task/dqn/sweep/",
     "best": "2"
     },
#    {"label": "ReLU+XY",
#     "control": "data/output/test_v12/maze_multigoal/linear_vf/source_task/dqn_aux/info/sweep/",
#     },
    {"label": "ReLU+Decoder",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/source_task/dqn_aux/input_decoder/sweep/",
     "best": "2"
     },
    {"label": "ReLU+NAS",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/source_task/dqn_aux/nas_v2_delta/sweep/",
     "best": "2"
     },
    {"label": "ReLU+Reward",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/source_task/dqn_aux/reward/sweep/",
     "best": "1"
     },
    {"label": "ReLU+SF",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/source_task/dqn_aux/successor_as/sweep/",
     "best": "1"
     },
    {"label": "FTA eta=0.2",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/source_task/dqn_lta/eta_study_0.2_sweep/",
     "best": "2"
     },
    {"label": "FTA eta=0.4",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/source_task/dqn_lta/eta_study_0.4_sweep/",
     "best": "2"
     },
    {"label": "FTA eta=0.6",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/source_task/dqn_lta/eta_study_0.6_sweep/",
     "best": "2"
     },
    {"label": "FTA eta=0.8",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/source_task/dqn_lta/eta_study_0.8_sweep/",
     "best": "2"
     },
    {"label": "FTA+XY",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/source_task/dqn_lta_aux/info/sweep/",
     "best": "2"
     },
    {"label": "FTA+Decoder",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/source_task/dqn_lta_aux/input_decoder/sweep/",
     "best": "3"
     },
    {"label": "FTA+NAS",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/source_task/dqn_lta_aux/nas_v2_delta/sweep/",
     },
    {"label": "FTA+Reward",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/source_task/dqn_lta_aux/reward/sweep/",
     "best": "2"
     },
    {"label": "FTA+SF",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/source_task/dqn_lta_aux/successor_as/sweep/",
     }
]
maze_source_best_v12 = [
    {"label": "ReLU",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/source_task/dqn/best/",
     },
    {"label": "ReLU+XY",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/source_task/dqn_aux/info/best/",
     },
    {"label": "ReLU+Decoder",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/source_task/dqn_aux/input_decoder/best/",
     },
    {"label": "ReLU+NAS",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/source_task/dqn_aux/nas_v2_delta/best/",
     },
    {"label": "ReLU+Reward",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/source_task/dqn_aux/reward/best/",
     },
    {"label": "ReLU+SF",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/source_task/dqn_aux/successor_as/best/",
     },
    {"label": "FTA eta=0.2",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/source_task/dqn_lta/eta_study_0.2_best/",
     },
    {"label": "FTA eta=0.4",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/source_task/dqn_lta/eta_study_0.4_best/",
     },
    {"label": "FTA eta=0.6",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/source_task/dqn_lta/eta_study_0.6_best/",
     },
    {"label": "FTA eta=0.8",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/source_task/dqn_lta/eta_study_0.8_best/",
     },
    {"label": "FTA+XY",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/source_task/dqn_lta_aux/info/best/",
     },
    {"label": "FTA+Decoder",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/source_task/dqn_lta_aux/input_decoder/best/",
     },
    {"label": "FTA+NAS",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/source_task/dqn_lta_aux/nas_v2_delta/best/",
     },
    {"label": "FTA+Reward",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/source_task/dqn_lta_aux/reward/best/",
     },
    {"label": "FTA+SF",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/source_task/dqn_lta_aux/successor_as/best/",
     }
]

maze_target_same_sweep = [
    {"label": "ReLU",
     "control": "data/output/test_v10/maze_multigoal/linear_vf/target_task/same/fix_rep/dqn/sweep/",
     },
    {"label": "ReLU+XY",
     "control": "data/output/test_v10/maze_multigoal/linear_vf/target_task/same/fix_rep/dqn_aux/info/sweep/",
     },
    {"label": "ReLU+Decoder",
     "control": "data/output/test_v10/maze_multigoal/linear_vf/target_task/same/fix_rep/dqn_aux/input_decoder/sweep/",
     },
    {"label": "ReLU+NAS",
     "control": "data/output/test_v10/maze_multigoal/linear_vf/target_task/same/fix_rep/dqn_aux/nas_v2_delta/sweep/",
     },
    {"label": "ReLU+Reward",
     "control": "data/output/test_v10/maze_multigoal/linear_vf/target_task/same/fix_rep/dqn_aux/reward/sweep/",
     },
    {"label": "ReLU+SF",
     "control": "data/output/test_v10/maze_multigoal/linear_vf/target_task/same/fix_rep/dqn_aux/successor_as/sweep/",
     },
    {"label": "FTA eta=0.2",
     "control": "data/output/test_v10/maze_multigoal/linear_vf/target_task/same/fix_rep/dqn_lta/eta_study_0.2_sweep/",
     },
    {"label": "FTA eta=0.4",
     "control": "data/output/test_v10/maze_multigoal/linear_vf/target_task/same/fix_rep/dqn_lta/eta_study_0.4_sweep/",
     },
    {"label": "FTA eta=0.6",
     "control": "data/output/test_v10/maze_multigoal/linear_vf/target_task/same/fix_rep/dqn_lta/eta_study_0.6_sweep/",
     },
    {"label": "FTA eta=0.8",
     "control": "data/output/test_v10/maze_multigoal/linear_vf/target_task/same/fix_rep/dqn_lta/eta_study_0.8_sweep/",
     },
    {"label": "FTA+XY",
     "control": "data/output/test_v10/maze_multigoal/linear_vf/target_task/same/fix_rep/dqn_lta_aux/info/sweep/",
     },
    {"label": "FTA+Decoder",
     "control": "data/output/test_v10/maze_multigoal/linear_vf/target_task/same/fix_rep/dqn_lta_aux/input_decoder/sweep/",
     },
    {"label": "FTA+NAS",
     "control": "data/output/test_v10/maze_multigoal/linear_vf/target_task/same/fix_rep/dqn_lta_aux/nas_v2_delta/sweep/",
     },
    {"label": "FTA+Reward",
     "control": "data/output/test_v10/maze_multigoal/linear_vf/target_task/same/fix_rep/dqn_lta_aux/reward/sweep/",
     },
    {"label": "FTA+SF",
     "control": "data/output/test_v10/maze_multigoal/linear_vf/target_task/same/fix_rep/dqn_lta_aux/successor_as/sweep/",
     },

    {"label": "Random",
     "control": "data/output/test_v10/maze_multigoal/linear_vf/target_task/same/fix_rep/random/sweep/",
     },
    {"label": "Input",
     "control": "data/output/test_v10/maze_multigoal/linear_vf/target_task/same/fix_rep/input/sweep/",
     },
    {"label": "Scratch",
     "control": "data/output/test_v10/maze_multigoal/linear_vf/source_task/dqn/sweep/",
     },
]

maze_target_same_sweep_v12 = [
    {"label": "ReLU",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task/same/fix_rep/dqn/sweep/",
     "best": "2"
     },
    {"label": "ReLU+XY",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task/same/fix_rep/dqn_aux/info/sweep/",
     "best": "2"
     },
    {"label": "ReLU+Decoder",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task/same/fix_rep/dqn_aux/input_decoder/sweep/",
     "best": "2"
     },
    {"label": "ReLU+NAS",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task/same/fix_rep/dqn_aux/nas_v2_delta/sweep/",
     "best": "2"
     },
    {"label": "ReLU+Reward",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task/same/fix_rep/dqn_aux/reward/sweep/",
     "best": "2"
     },
    {"label": "ReLU+SF",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task/same/fix_rep/dqn_aux/successor_as/sweep/",
     "best": "1"
     },
    {"label": "FTA eta=0.2",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task/same/fix_rep/dqn_lta/eta_study_0.2_sweep/",
     "best": "2"
     },
    {"label": "FTA eta=0.4",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task/same/fix_rep/dqn_lta/eta_study_0.4_sweep/",
     "best": "3"
     },
    {"label": "FTA eta=0.6",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task/same/fix_rep/dqn_lta/eta_study_0.6_sweep/",
     "best": "3"
     },
    {"label": "FTA eta=0.8",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task/same/fix_rep/dqn_lta/eta_study_0.8_sweep/",
     "best": "2"
     },
    {"label": "FTA+XY",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task/same/fix_rep/dqn_lta_aux/info/sweep/",
     "best": "4"
     },
    {"label": "FTA+Decoder",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task/same/fix_rep/dqn_lta_aux/input_decoder/sweep/",
     "best": "4"
     },
    {"label": "FTA+NAS",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task/same/fix_rep/dqn_lta_aux/nas_v2_delta/sweep/",
     "best": "3"
     },
    {"label": "FTA+Reward",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task/same/fix_rep/dqn_lta_aux/reward/sweep/",
     "best": "3"
     },
    {"label": "FTA+SF",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task/same/fix_rep/dqn_lta_aux/successor_as/sweep/",
     "best": "4"
     },

    {"label": "Random",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task/same/fix_rep/random/sweep/",
     "best": "0"
     },
    {"label": "Input",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task/same/fix_rep/input/sweep/",
     "best": "2"
     },
    {"label": "Scratch",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/source_task/dqn/sweep/",
     "best": "1"
     },
]

maze_target_diff_sweep_v12 = [
    {"label": "ReLU",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task/dissimilar/fix_rep/dqn/sweep/",
     "best": "2"
     },
    {"label": "ReLU+XY",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task/dissimilar/fix_rep/dqn_aux/info/sweep/",
     "best": "2"
     },
    {"label": "ReLU+Decoder",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task/dissimilar/fix_rep/dqn_aux/input_decoder/sweep/",
     "best": "2"
     },
    {"label": "ReLU+NAS",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task/dissimilar/fix_rep/dqn_aux/nas_v2_delta/sweep/",
     "best": "2"
     },
    {"label": "ReLU+Reward",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task/dissimilar/fix_rep/dqn_aux/reward/sweep/",
     "best": "2"
     },
    {"label": "ReLU+SF",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task/dissimilar/fix_rep/dqn_aux/successor_as/sweep/",
     "best": "1"
     },
    {"label": "FTA eta=0.2",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task/dissimilar/fix_rep/dqn_lta/eta_study_0.2_sweep/",
     "best": "2"
     },
    {"label": "FTA eta=0.4",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task/dissimilar/fix_rep/dqn_lta/eta_study_0.4_sweep/",
     "best": "4"
     },
    {"label": "FTA eta=0.6",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task/dissimilar/fix_rep/dqn_lta/eta_study_0.6_sweep/",
     "best": "4"
     },
    {"label": "FTA eta=0.8",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task/dissimilar/fix_rep/dqn_lta/eta_study_0.8_sweep/",
     "best": "2"
     },
    {"label": "FTA+XY",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task/dissimilar/fix_rep/dqn_lta_aux/info/sweep/",
     "best": "4"
     },
    {"label": "FTA+Decoder",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task/dissimilar/fix_rep/dqn_lta_aux/input_decoder/sweep/",
     "best": "4"
     },
    {"label": "FTA+NAS",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task/dissimilar/fix_rep/dqn_lta_aux/nas_v2_delta/sweep/",
     "best": "3"
     },
    {"label": "FTA+Reward",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task/dissimilar/fix_rep/dqn_lta_aux/reward/sweep/",
     "best": "3"
     },
    {"label": "FTA+SF",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task/dissimilar/fix_rep/dqn_lta_aux/successor_as/sweep/",
     "best": "4"
     },

    {"label": "Random",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task/dissimilar/fix_rep/random/sweep/",
     "best": "0"
     },
    {"label": "Input",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task/dissimilar/fix_rep/input/sweep/",
     "best": "2"
     },
    {"label": "Scratch",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task/dissimilar/scratch/dqn/sweep/",
     "best": "1"
     },
]

maze_target_same_best_v12 = [
    {"label": "ReLU",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task/same/fix_rep/dqn/best/",
     "best": "2"
     },
    {"label": "ReLU+XY",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task/same/fix_rep/dqn_aux/info/best/",
     },
    {"label": "ReLU+Decoder",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task/same/fix_rep/dqn_aux/input_decoder/best/",
     "best": "2"
     },
    {"label": "ReLU+NAS",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task/same/fix_rep/dqn_aux/nas_v2_delta/best/",
     "best": "2"
     },
    {"label": "ReLU+Reward",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task/same/fix_rep/dqn_aux/reward/best/",
     },
    {"label": "ReLU+SF",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task/same/fix_rep/dqn_aux/successor_as/best/",
     "best": "2"
     },
    {"label": "FTA eta=0.2",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task/same/fix_rep/dqn_lta/eta_study_0.2_best/",
     "best": "2"
     },
    {"label": "FTA eta=0.4",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task/same/fix_rep/dqn_lta/eta_study_0.4_best/",
     "best": "3"
     },
    {"label": "FTA eta=0.6",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task/same/fix_rep/dqn_lta/eta_study_0.6_best/",
     "best": "3"
     },
    {"label": "FTA eta=0.8",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task/same/fix_rep/dqn_lta/eta_study_0.8_best/",
     "best": "2"
     },
    {"label": "FTA+XY",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task/same/fix_rep/dqn_lta_aux/info/best/",
     "best": "2"
     },
    {"label": "FTA+Decoder",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task/same/fix_rep/dqn_lta_aux/input_decoder/best/",
     "best": "2"
     },
    {"label": "FTA+NAS",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task/same/fix_rep/dqn_lta_aux/nas_v2_delta/best/",
     },
    {"label": "FTA+Reward",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task/same/fix_rep/dqn_lta_aux/reward/best/",
     "best": "2"
     },
    {"label": "FTA+SF",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task/same/fix_rep/dqn_lta_aux/successor_as/best/",
     },

    {"label": "Random",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task/same/fix_rep/random/best/",
     "best": "0"
     },
    {"label": "Input",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task/same/fix_rep/input/best/",
     "best": "3"
     },
    {"label": "Scratch",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/source_task/dqn/best/",
     "best": "2"
     },
]

maze_target_diff_best_v12 = [
    {"label": "ReLU",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task/dissimilar/fix_rep/dqn/best/",
     "best": "2"
     },
    {"label": "ReLU+XY",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task/dissimilar/fix_rep/dqn_aux/info/best/",
     },
    {"label": "ReLU+Decoder",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task/dissimilar/fix_rep/dqn_aux/input_decoder/best/",
     "best": "2"
     },
    {"label": "ReLU+NAS",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task/dissimilar/fix_rep/dqn_aux/nas_v2_delta/best/",
     "best": "2"
     },
    {"label": "ReLU+Reward",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task/dissimilar/fix_rep/dqn_aux/reward/best/",
     },
    {"label": "ReLU+SF",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task/dissimilar/fix_rep/dqn_aux/successor_as/best/",
     "best": "1"
     },
    {"label": "FTA eta=0.2",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task/dissimilar/fix_rep/dqn_lta/eta_study_0.2_best/",
     "best": "2"
     },
    {"label": "FTA eta=0.4",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task/dissimilar/fix_rep/dqn_lta/eta_study_0.4_best/",
     "best": "4"
     },
    {"label": "FTA eta=0.6",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task/dissimilar/fix_rep/dqn_lta/eta_study_0.6_best/",
     "best": "4"
     },
    {"label": "FTA eta=0.8",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task/dissimilar/fix_rep/dqn_lta/eta_study_0.8_best/",
     "best": "2"
     },
    {"label": "FTA+XY",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task/dissimilar/fix_rep/dqn_lta_aux/info/best/",
     "best": "4"
     },
    {"label": "FTA+Decoder",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task/dissimilar/fix_rep/dqn_lta_aux/input_decoder/best/",
     "best": "4"
     },
    {"label": "FTA+NAS",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task/dissimilar/fix_rep/dqn_lta_aux/nas_v2_delta/best/",
     },
    {"label": "FTA+Reward",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task/dissimilar/fix_rep/dqn_lta_aux/reward/best/",
     "best": "3"
     },
    {"label": "FTA+SF",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task/dissimilar/fix_rep/dqn_lta_aux/successor_as/best/",
     },

    {"label": "Random",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task/dissimilar/fix_rep/random/best/",
     "best": "0"
     },
    {"label": "Input",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task/dissimilar/fix_rep/input/best/",
     "best": "2"
     },
    {"label": "Scratch",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task/dissimilar/scratch/dqn/best/",
     },
]


maze_target_same_sweep_v12_3g = [
    {"label": "ReLU",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_3g/same/fix_rep/dqn/sweep/",
     "best": "2"
     },
    {"label": "ReLU+XY",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_3g/same/fix_rep/dqn_aux/info/sweep/",
     "best": "2"
     },
    {"label": "ReLU+Decoder",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_3g/same/fix_rep/dqn_aux/input_decoder/sweep/",
     "best": "2"
     },
    {"label": "ReLU+NAS",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_3g/same/fix_rep/dqn_aux/nas_v2_delta/sweep/",
     "best": "2"
     },
    {"label": "ReLU+Reward",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_3g/same/fix_rep/dqn_aux/reward/sweep/",
     "best": "2"
     },
    {"label": "ReLU+SF",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_3g/same/fix_rep/dqn_aux/successor_as/sweep/",
     "best": "1"
     },
    {"label": "FTA eta=0.2",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_3g/same/fix_rep/dqn_lta/eta_study_0.2_sweep/",
     "best": "2"
     },
    {"label": "FTA eta=0.4",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_3g/same/fix_rep/dqn_lta/eta_study_0.4_sweep/",
     "best": "3"
     },
    {"label": "FTA eta=0.6",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_3g/same/fix_rep/dqn_lta/eta_study_0.6_sweep/",
     "best": "3"
     },
    {"label": "FTA eta=0.8",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_3g/same/fix_rep/dqn_lta/eta_study_0.8_sweep/",
     "best": "2"
     },
    {"label": "FTA+XY",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_3g/same/fix_rep/dqn_lta_aux/info/sweep/",
     "best": "4"
     },
    {"label": "FTA+Decoder",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_3g/same/fix_rep/dqn_lta_aux/input_decoder/sweep/",
     "best": "4"
     },
    {"label": "FTA+NAS",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_3g/same/fix_rep/dqn_lta_aux/nas_v2_delta/sweep/",
     "best": "3"
     },
    {"label": "FTA+Reward",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_3g/same/fix_rep/dqn_lta_aux/reward/sweep/",
     "best": "3"
     },
    {"label": "FTA+SF",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_3g/same/fix_rep/dqn_lta_aux/successor_as/sweep/",
     "best": "4"
     },

    {"label": "Random",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_3g/same/fix_rep/random/sweep/",
     "best": "0"
     },
    {"label": "Input",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_3g/same/fix_rep/input/sweep/",
     "best": "2"
     },
    {"label": "Scratch",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/source_task/dqn/sweep/",
     "best": "1"
     },
]

maze_target_diff_sweep_v12_3g = [
    {"label": "ReLU",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_3g/dissimilar/fix_rep/dqn/sweep/",
     "best": "3"
     },
    {"label": "ReLU+XY",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_3g/dissimilar/fix_rep/dqn_aux/info/sweep/",
     "best": "2"
     },
    {"label": "ReLU+Decoder",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_3g/dissimilar/fix_rep/dqn_aux/input_decoder/sweep/",
     "best": "2"
     },
    {"label": "ReLU+NAS",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_3g/dissimilar/fix_rep/dqn_aux/nas_v2_delta/sweep/",
     "best": "3"
     },
    {"label": "ReLU+Reward",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_3g/dissimilar/fix_rep/dqn_aux/reward/sweep/",
     "best": "3"
     },
    {"label": "ReLU+SF",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_3g/dissimilar/fix_rep/dqn_aux/successor_as/sweep/",
     "best": "3"
     },
    {"label": "FTA eta=0.2",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_3g/dissimilar/fix_rep/dqn_lta/eta_study_0.2_sweep/",
     "best": "3"
     },
    {"label": "FTA eta=0.4",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_3g/dissimilar/fix_rep/dqn_lta/eta_study_0.4_sweep/",
     "best": "3"
     },
    {"label": "FTA eta=0.6",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_3g/dissimilar/fix_rep/dqn_lta/eta_study_0.6_sweep/",
     "best": "2"
     },
    {"label": "FTA eta=0.8",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_3g/dissimilar/fix_rep/dqn_lta/eta_study_0.8_sweep/",
     "best": "2"
     },
    {"label": "FTA+XY",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_3g/dissimilar/fix_rep/dqn_lta_aux/info/sweep/",
     "best": "3"
     },
    {"label": "FTA+Decoder",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_3g/dissimilar/fix_rep/dqn_lta_aux/input_decoder/sweep/",
     "best": "3"
     },
    {"label": "FTA+NAS",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_3g/dissimilar/fix_rep/dqn_lta_aux/nas_v2_delta/sweep/",
     "best": "3"
     },
    {"label": "FTA+Reward",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_3g/dissimilar/fix_rep/dqn_lta_aux/reward/sweep/",
     "best": "3"
     },
    {"label": "FTA+SF",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_3g/dissimilar/fix_rep/dqn_lta_aux/successor_as/sweep/",
     "best": "4"
     },

    {"label": "Random",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_3g/dissimilar/fix_rep/random/sweep/",
     "best": "0"
     },
    {"label": "Input",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_3g/dissimilar/fix_rep/input/sweep/",
     "best": "3"
     },
    {"label": "Scratch",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_3g/dissimilar/scratch/dqn/sweep/",
     "best": "1"
     },
]

maze_target_same_best_v12_3g = [
    {"label": "ReLU",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_3g/same/fix_rep/dqn/best/",
     "best": "2"
     },
    {"label": "ReLU+XY",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_3g/same/fix_rep/dqn_aux/info/best/",
     },
    {"label": "ReLU+Decoder",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_3g/same/fix_rep/dqn_aux/input_decoder/best/",
     "best": "2"
     },
    {"label": "ReLU+NAS",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_3g/same/fix_rep/dqn_aux/nas_v2_delta/best/",
     "best": "2"
     },
    {"label": "ReLU+Reward",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_3g/same/fix_rep/dqn_aux/reward/best/",
     },
    {"label": "ReLU+SF",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_3g/same/fix_rep/dqn_aux/successor_as/best/",
     "best": "2"
     },
    {"label": "FTA eta=0.2",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_3g/same/fix_rep/dqn_lta/eta_study_0.2_best/",
     "best": "2"
     },
    {"label": "FTA eta=0.4",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_3g/same/fix_rep/dqn_lta/eta_study_0.4_best/",
     "best": "3"
     },
    {"label": "FTA eta=0.6",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_3g/same/fix_rep/dqn_lta/eta_study_0.6_best/",
     "best": "3"
     },
    {"label": "FTA eta=0.8",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_3g/same/fix_rep/dqn_lta/eta_study_0.8_best/",
     "best": "2"
     },
    {"label": "FTA+XY",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_3g/same/fix_rep/dqn_lta_aux/info/best/",
     "best": "2"
     },
    {"label": "FTA+Decoder",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_3g/same/fix_rep/dqn_lta_aux/input_decoder/best/",
     "best": "2"
     },
    {"label": "FTA+NAS",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_3g/same/fix_rep/dqn_lta_aux/nas_v2_delta/best/",
     },
    {"label": "FTA+Reward",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_3g/same/fix_rep/dqn_lta_aux/reward/best/",
     "best": "2"
     },
    {"label": "FTA+SF",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_3g/same/fix_rep/dqn_lta_aux/successor_as/best/",
     },

    {"label": "Random",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_3g/same/fix_rep/random/best/",
     "best": "0"
     },
    {"label": "Input",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_3g/same/fix_rep/input/best/",
     "best": "3"
     },
    {"label": "Scratch",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/source_task/dqn/best/",
     "best": "2"
     },
]

maze_target_diff_best_v12_3g = [
    {"label": "ReLU",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_3g/dissimilar/fix_rep/dqn/best/",
     "best": "2"
     },
    {"label": "ReLU+XY",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_3g/dissimilar/fix_rep/dqn_aux/info/best/",
     },
    {"label": "ReLU+Decoder",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_3g/dissimilar/fix_rep/dqn_aux/input_decoder/best/",
     "best": "2"
     },
    {"label": "ReLU+NAS",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_3g/dissimilar/fix_rep/dqn_aux/nas_v2_delta/best/",
     "best": "2"
     },
    {"label": "ReLU+Reward",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_3g/dissimilar/fix_rep/dqn_aux/reward/best/",
     },
    {"label": "ReLU+SF",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_3g/dissimilar/fix_rep/dqn_aux/successor_as/best/",
     "best": "1"
     },
    {"label": "FTA eta=0.2",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_3g/dissimilar/fix_rep/dqn_lta/eta_study_0.2_best/",
     "best": "2"
     },
    {"label": "FTA eta=0.4",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_3g/dissimilar/fix_rep/dqn_lta/eta_study_0.4_best/",
     "best": "4"
     },
    {"label": "FTA eta=0.6",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_3g/dissimilar/fix_rep/dqn_lta/eta_study_0.6_best/",
     "best": "4"
     },
    {"label": "FTA eta=0.8",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_3g/dissimilar/fix_rep/dqn_lta/eta_study_0.8_best/",
     "best": "2"
     },
    {"label": "FTA+XY",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_3g/dissimilar/fix_rep/dqn_lta_aux/info/best/",
     "best": "4"
     },
    {"label": "FTA+Decoder",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_3g/dissimilar/fix_rep/dqn_lta_aux/input_decoder/best/",
     "best": "4"
     },
    {"label": "FTA+NAS",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_3g/dissimilar/fix_rep/dqn_lta_aux/nas_v2_delta/best/",
     },
    {"label": "FTA+Reward",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_3g/dissimilar/fix_rep/dqn_lta_aux/reward/best/",
     "best": "3"
     },
    {"label": "FTA+SF",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_3g/dissimilar/fix_rep/dqn_lta_aux/successor_as/best/",
     },

    {"label": "Random",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_3g/dissimilar/fix_rep/random/best/",
     "best": "0"
     },
    {"label": "Input",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_3g/dissimilar/fix_rep/input/best/",
     "best": "2"
     },
    {"label": "Scratch",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_3g/dissimilar/scratch/dqn/best/",
     },
]



maze_target_diff_sweep_v12_5g = [
    {"label": "ReLU",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_5g/dissimilar/fix_rep/dqn/sweep/",
     "best": "2"
     },
    {"label": "ReLU+XY",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_5g/dissimilar/fix_rep/dqn_aux/info/sweep/",
     "best": "2"
     },
    {"label": "ReLU+Decoder",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_5g/dissimilar/fix_rep/dqn_aux/input_decoder/sweep/",
     "best": "2"
     },
    {"label": "ReLU+NAS",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_5g/dissimilar/fix_rep/dqn_aux/nas_v2_delta/sweep/",
     "best": "2"
     },
    {"label": "ReLU+Reward",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_5g/dissimilar/fix_rep/dqn_aux/reward/sweep/",
     "best": "2"
     },
    {"label": "ReLU+SF",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_5g/dissimilar/fix_rep/dqn_aux/successor_as/sweep/",
     "best": "1"
     },
    {"label": "FTA eta=0.2",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_5g/dissimilar/fix_rep/dqn_lta/eta_study_0.2_sweep/",
     "best": "2"
     },
    {"label": "FTA eta=0.4",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_5g/dissimilar/fix_rep/dqn_lta/eta_study_0.4_sweep/",
     "best": "4"
     },
    {"label": "FTA eta=0.6",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_5g/dissimilar/fix_rep/dqn_lta/eta_study_0.6_sweep/",
     "best": "4"
     },
    {"label": "FTA eta=0.8",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_5g/dissimilar/fix_rep/dqn_lta/eta_study_0.8_sweep/",
     "best": "2"
     },
    {"label": "FTA+XY",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_5g/dissimilar/fix_rep/dqn_lta_aux/info/sweep/",
     "best": "4"
     },
    {"label": "FTA+Decoder",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_5g/dissimilar/fix_rep/dqn_lta_aux/input_decoder/sweep/",
     "best": "4"
     },
    {"label": "FTA+NAS",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_5g/dissimilar/fix_rep/dqn_lta_aux/nas_v2_delta/sweep/",
     "best": "3"
     },
    {"label": "FTA+Reward",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_5g/dissimilar/fix_rep/dqn_lta_aux/reward/sweep/",
     "best": "3"
     },
    {"label": "FTA+SF",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_5g/dissimilar/fix_rep/dqn_lta_aux/successor_as/sweep/",
     "best": "4"
     },

    {"label": "Random",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_5g/dissimilar/fix_rep/random/sweep/",
     "best": "0"
     },
    {"label": "Input",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_5g/dissimilar/fix_rep/input/sweep/",
     "best": "2"
     },
    {"label": "Scratch",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_5g/dissimilar/scratch/dqn/sweep/",
     "best": "1"
     },
]

maze_target_same_best_v12_5g = [
    {"label": "ReLU",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_5g/same/fix_rep/dqn/best/",
     "best": "2"
     },
    {"label": "ReLU+XY",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_5g/same/fix_rep/dqn_aux/info/best/",
     },
    {"label": "ReLU+Decoder",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_5g/same/fix_rep/dqn_aux/input_decoder/best/",
     "best": "2"
     },
    {"label": "ReLU+NAS",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_5g/same/fix_rep/dqn_aux/nas_v2_delta/best/",
     "best": "2"
     },
    {"label": "ReLU+Reward",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_5g/same/fix_rep/dqn_aux/reward/best/",
     },
    {"label": "ReLU+SF",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_5g/same/fix_rep/dqn_aux/successor_as/best/",
     "best": "2"
     },
    {"label": "FTA eta=0.2",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_5g/same/fix_rep/dqn_lta/eta_study_0.2_best/",
     "best": "2"
     },
    {"label": "FTA eta=0.4",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_5g/same/fix_rep/dqn_lta/eta_study_0.4_best/",
     "best": "3"
     },
    {"label": "FTA eta=0.6",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_5g/same/fix_rep/dqn_lta/eta_study_0.6_best/",
     "best": "3"
     },
    {"label": "FTA eta=0.8",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_5g/same/fix_rep/dqn_lta/eta_study_0.8_best/",
     "best": "2"
     },
    {"label": "FTA+XY",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_5g/same/fix_rep/dqn_lta_aux/info/best/",
     "best": "2"
     },
    {"label": "FTA+Decoder",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_5g/same/fix_rep/dqn_lta_aux/input_decoder/best/",
     "best": "2"
     },
    {"label": "FTA+NAS",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_5g/same/fix_rep/dqn_lta_aux/nas_v2_delta/best/",
     },
    {"label": "FTA+Reward",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_5g/same/fix_rep/dqn_lta_aux/reward/best/",
     "best": "2"
     },
    {"label": "FTA+SF",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_5g/same/fix_rep/dqn_lta_aux/successor_as/best/",
     },

    {"label": "Random",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_5g/same/fix_rep/random/best/",
     "best": "0"
     },
    {"label": "Input",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_5g/same/fix_rep/input/best/",
     "best": "3"
     },
    {"label": "Scratch",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/source_task/dqn/best/",
     "best": "2"
     },
]

maze_target_diff_best_v12_5g = [
    {"label": "ReLU",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_5g/dissimilar/fix_rep/dqn/best/",
     "best": "1"
     },
    {"label": "ReLU+XY",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_5g/dissimilar/fix_rep/dqn_aux/info/best/",
     "best": "1"
     },
    {"label": "ReLU+Decoder",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_5g/dissimilar/fix_rep/dqn_aux/input_decoder/best/",
     "best": "1"
     },
    {"label": "ReLU+NAS",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_5g/dissimilar/fix_rep/dqn_aux/nas_v2_delta/best/",
     "best": "1"
     },
    {"label": "ReLU+Reward",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_5g/dissimilar/fix_rep/dqn_aux/reward/best/",
     "best": "1"
     },
    {"label": "ReLU+SF",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_5g/dissimilar/fix_rep/dqn_aux/successor_as/best/",
     "best": "1"
     },
    {"label": "FTA eta=0.2",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_5g/dissimilar/fix_rep/dqn_lta/eta_study_0.2_best/",
     "best": "3"
     },
    {"label": "FTA eta=0.4",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_5g/dissimilar/fix_rep/dqn_lta/eta_study_0.4_best/",
     "best": "2"
     },
    {"label": "FTA eta=0.6",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_5g/dissimilar/fix_rep/dqn_lta/eta_study_0.6_best/",
     "best": "2"
     },
    {"label": "FTA eta=0.8",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_5g/dissimilar/fix_rep/dqn_lta/eta_study_0.8_best/",
     "best": "2"
     },
    {"label": "FTA+XY",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_5g/dissimilar/fix_rep/dqn_lta_aux/info/best/",
     "best": "2"
     },
    {"label": "FTA+Decoder",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_5g/dissimilar/fix_rep/dqn_lta_aux/input_decoder/best/",
     "best": "2"
     },
    {"label": "FTA+NAS",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_5g/dissimilar/fix_rep/dqn_lta_aux/nas_v2_delta/best/",
     "best": "3"
     },
    {"label": "FTA+Reward",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_5g/dissimilar/fix_rep/dqn_lta_aux/reward/best/",
     "best": "2"
     },
    {"label": "FTA+SF",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_5g/dissimilar/fix_rep/dqn_lta_aux/successor_as/best/",
     "best": "3"
     },

    {"label": "Random",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_5g/dissimilar/fix_rep/random/best/",
     "best": "0"
     },
    {"label": "Input",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_5g/dissimilar/fix_rep/input/best/",
     "best": "3"
     },
    {"label": "Scratch",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_5g/dissimilar/scratch/dqn/best/",
     "best": "2"
     },
]


maze_target_diff_fine_tune_best_v12_5g = [
    {"label": "ReLU",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_5g/dissimilar/fine_tune/dqn/best/",
     "best": "4"
     },
    {"label": "ReLU+XY",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_5g/dissimilar/fine_tune/dqn_aux/info/best/",
     "best": "4"
     },
    {"label": "ReLU+Decoder",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_5g/dissimilar/fine_tune/dqn_aux/input_decoder/best/",
     "best": "4"
     },
    {"label": "ReLU+NAS",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_5g/dissimilar/fine_tune/dqn_aux/nas_v2_delta/best/",
     "best": "4"
     },
    {"label": "ReLU+Reward",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_5g/dissimilar/fine_tune/dqn_aux/reward/best/",
     "best": "4"
     },
    {"label": "ReLU+SF",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_5g/dissimilar/fine_tune/dqn_aux/successor_as/best/",
     "best": "0"
     },
    {"label": "FTA eta=0.2",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_5g/dissimilar/fine_tune/dqn_lta/eta_study_0.2_best/",
     "best": "2"
     },
    {"label": "FTA eta=0.4",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_5g/dissimilar/fine_tune/dqn_lta/eta_study_0.4_best/",
     "best": "2"
     },
    {"label": "FTA eta=0.6",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_5g/dissimilar/fine_tune/dqn_lta/eta_study_0.6_best/",
     "best": "2"
     },
    {"label": "FTA eta=0.8",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_5g/dissimilar/fine_tune/dqn_lta/eta_study_0.8_best/",
     "best": "2"
     },
    {"label": "FTA+XY",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_5g/dissimilar/fine_tune/dqn_lta_aux/info/best/",
     "best": "2"
     },
    {"label": "FTA+Decoder",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_5g/dissimilar/fine_tune/dqn_lta_aux/input_decoder/best/",
     "best": "2"
     },
    {"label": "FTA+NAS",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_5g/dissimilar/fine_tune/dqn_lta_aux/nas_v2_delta/best/",
     "best": "2"
     },
    {"label": "FTA+Reward",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_5g/dissimilar/fine_tune/dqn_lta_aux/reward/best/",
     "best": "2"
     },
    {"label": "FTA+SF",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_5g/dissimilar/fine_tune/dqn_lta_aux/successor_as/best/",
     "best": "2"
     },

    {"label": "Random",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_5g/dissimilar/fix_rep/random/best/",
     "best": "0"
     },
    {"label": "Input",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_5g/dissimilar/fix_rep/input/best/",
     "best": "2"
     },
    {"label": "Scratch",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_5g/dissimilar/scratch/dqn/best/",
     "best": "1"
     },
]


maze_target_diff_fine_tune_best_v12_3g = [
    {"label": "ReLU",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_3g/dissimilar/fine_tune/dqn/best/",
     "best": "4"
     },
    {"label": "ReLU+XY",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_3g/dissimilar/fine_tune/dqn_aux/info/best/",
     "best": "4"
     },
    {"label": "ReLU+Decoder",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_3g/dissimilar/fine_tune/dqn_aux/input_decoder/best/",
     "best": "4"
     },
    {"label": "ReLU+NAS",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_3g/dissimilar/fine_tune/dqn_aux/nas_v2_delta/best/",
     "best": "3"
     },
    {"label": "ReLU+Reward",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_3g/dissimilar/fine_tune/dqn_aux/reward/best/",
     "best": "3"
     },
    {"label": "ReLU+SF",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_3g/dissimilar/fine_tune/dqn_aux/successor_as/best/",
     "best": "3"
     },
    {"label": "FTA eta=0.2",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_3g/dissimilar/fine_tune/dqn_lta/eta_study_0.2_best/",
     "best": "2"
     },
    {"label": "FTA eta=0.4",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_3g/dissimilar/fine_tune/dqn_lta/eta_study_0.4_best/",
     "best": "2"
     },
    {"label": "FTA eta=0.6",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_3g/dissimilar/fine_tune/dqn_lta/eta_study_0.6_best/",
     "best": "2"
     },
    {"label": "FTA eta=0.8",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_3g/dissimilar/fine_tune/dqn_lta/eta_study_0.8_best/",
     "best": "2"
     },
    {"label": "FTA+XY",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_3g/dissimilar/fine_tune/dqn_lta_aux/info/best/",
     "best": "2"
     },
    {"label": "FTA+Decoder",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_3g/dissimilar/fine_tune/dqn_lta_aux/input_decoder/best/",
     "best": "2"
     },
    {"label": "FTA+NAS",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_3g/dissimilar/fine_tune/dqn_lta_aux/nas_v2_delta/best/",
     "best": "2"
     },
    {"label": "FTA+Reward",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_3g/dissimilar/fine_tune/dqn_lta_aux/reward/best/",
     "best": "2"
     },
    {"label": "FTA+SF",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_3g/dissimilar/fine_tune/dqn_lta_aux/successor_as/best/",
     "best": "2"
     },

    {"label": "Random",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_3g/dissimilar/fix_rep/random/best/",
     "best": "0"
     },
    {"label": "Input",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_3g/dissimilar/fix_rep/input/best/",
     "best": "2"
     },
    {"label": "Scratch",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_3g/dissimilar/scratch/dqn/best/",
     "best": "1"
     },
]


maze_target_diff_fine_tune_best_v12 = [
    {"label": "ReLU",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task/dissimilar/fine_tune/dqn/best/",
     "best": "4"
     },
    {"label": "ReLU+XY",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task/dissimilar/fine_tune/dqn_aux/info/best/",
     "best": "4"
     },
    {"label": "ReLU+Decoder",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task/dissimilar/fine_tune/dqn_aux/input_decoder/best/",
     "best": "4"
     },
    {"label": "ReLU+NAS",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task/dissimilar/fine_tune/dqn_aux/nas_v2_delta/best/",
     "best": "4"
     },
    {"label": "ReLU+Reward",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task/dissimilar/fine_tune/dqn_aux/reward/best/",
     "best": "4"
     },
    {"label": "ReLU+SF",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task/dissimilar/fine_tune/dqn_aux/successor_as/best/",
     "best": "3"
     },
    {"label": "FTA eta=0.2",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task/dissimilar/fine_tune/dqn_lta/eta_study_0.2_best/",
     "best": "2"
     },
    {"label": "FTA eta=0.4",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task/dissimilar/fine_tune/dqn_lta/eta_study_0.4_best/",
     "best": "2"
     },
    {"label": "FTA eta=0.6",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task/dissimilar/fine_tune/dqn_lta/eta_study_0.6_best/",
     "best": "2"
     },
    {"label": "FTA eta=0.8",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task/dissimilar/fine_tune/dqn_lta/eta_study_0.8_best/",
     "best": "2"
     },
    {"label": "FTA+XY",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task/dissimilar/fine_tune/dqn_lta_aux/info/best/",
     "best": "2"
     },
    {"label": "FTA+Decoder",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task/dissimilar/fine_tune/dqn_lta_aux/input_decoder/best/",
     "best": "2"
     },
    {"label": "FTA+NAS",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task/dissimilar/fine_tune/dqn_lta_aux/nas_v2_delta/best/",
     "best": "2"
     },
    {"label": "FTA+Reward",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task/dissimilar/fine_tune/dqn_lta_aux/reward/best/",
     "best": "2"
     },
    {"label": "FTA+SF",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task/dissimilar/fine_tune/dqn_lta_aux/successor_as/best/",
     "best": "2"
     },

    {"label": "Random",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task/dissimilar/fix_rep/random/best/",
     "best": "0"
     },
    {"label": "Input",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task/dissimilar/fix_rep/input/best/",
     "best": "2"
     },
    {"label": "Scratch",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task/dissimilar/scratch/dqn/best/",
     "best": "1"
     },
]

maze_target_diff_sweep_v12_15g = [
    {"label": "ReLU",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_15g/dissimilar/fix_rep/dqn/sweep/",
     "best": "2"
     },
    {"label": "ReLU+XY",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_15g/dissimilar/fix_rep/dqn_aux/info/sweep/",
     "best": "2"
     },
    {"label": "ReLU+Decoder",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_15g/dissimilar/fix_rep/dqn_aux/input_decoder/sweep/",
     "best": "2"
     },
    {"label": "ReLU+NAS",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_15g/dissimilar/fix_rep/dqn_aux/nas_v2_delta/sweep/",
     "best": "2"
     },
    {"label": "ReLU+Reward",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_15g/dissimilar/fix_rep/dqn_aux/reward/sweep/",
     "best": "1"
     },
    {"label": "ReLU+SF",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_15g/dissimilar/fix_rep/dqn_aux/successor_as/sweep/",
     "best": "1"
     },
    {"label": "FTA eta=0.2",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_15g/dissimilar/fix_rep/dqn_lta/eta_study_0.2_sweep/",
     "best": "2"
     },
    {"label": "FTA eta=0.4",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_15g/dissimilar/fix_rep/dqn_lta/eta_study_0.4_sweep/",
     "best": "2"
     },
    {"label": "FTA eta=0.6",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_15g/dissimilar/fix_rep/dqn_lta/eta_study_0.6_sweep/",
     "best": "2"
     },
    {"label": "FTA eta=0.8",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_15g/dissimilar/fix_rep/dqn_lta/eta_study_0.8_sweep/",
     "best": "2"
     },
    {"label": "FTA+XY",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_15g/dissimilar/fix_rep/dqn_lta_aux/info/sweep/",
     "best": "2"
     },
    {"label": "FTA+Decoder",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_15g/dissimilar/fix_rep/dqn_lta_aux/input_decoder/sweep/",
     "best": "2"
     },
    {"label": "FTA+NAS",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_15g/dissimilar/fix_rep/dqn_lta_aux/nas_v2_delta/sweep/",
     "best": "2"
     },
    {"label": "FTA+Reward",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_15g/dissimilar/fix_rep/dqn_lta_aux/reward/sweep/",
     "best": "2"
     },
    {"label": "FTA+SF",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_15g/dissimilar/fix_rep/dqn_lta_aux/successor_as/sweep/",
     "best": "2"
     },

    {"label": "Random",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_15g/dissimilar/fix_rep/random/sweep/",
     "best": "0"
     },
    {"label": "Input",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_15g/dissimilar/fix_rep/input/sweep/",
     "best": "3"
     },
    {"label": "Scratch",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_15g/dissimilar/scratch/dqn/sweep/",
     "best": "2"
     },
]

maze_target_diff_sweep_v12_30g = [
   {"label": "ReLU",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g/dissimilar/fix_rep/dqn/sweep/",
     "best": "2"
     },
    {"label": "ReLU+XY",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g/dissimilar/fix_rep/dqn_aux/info/sweep/",
     "best": "2"
     },
    {"label": "ReLU+Decoder",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g/dissimilar/fix_rep/dqn_aux/input_decoder/sweep/",
     "best": "2"
     },
    {"label": "ReLU+NAS",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g/dissimilar/fix_rep/dqn_aux/nas_v2_delta/sweep/",
     "best": "2"
     },
    {"label": "ReLU+Reward",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g/dissimilar/fix_rep/dqn_aux/reward/sweep/",
     "best": "1"
     },
    {"label": "ReLU+SF",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g/dissimilar/fix_rep/dqn_aux/successor_as/sweep/",
     "best": "1"
     },
    {"label": "FTA eta=0.2",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g/dissimilar/fix_rep/dqn_lta/eta_study_0.2_sweep/",
     "best": "2"
     },
    {"label": "FTA eta=0.4",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g/dissimilar/fix_rep/dqn_lta/eta_study_0.4_sweep/",
     "best": "2"
     },
    {"label": "FTA eta=0.6",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g/dissimilar/fix_rep/dqn_lta/eta_study_0.6_sweep/",
     "best": "2"
     },
    {"label": "FTA eta=0.8",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g/dissimilar/fix_rep/dqn_lta/eta_study_0.8_sweep/",
     "best": "2"
     },
    {"label": "FTA+XY",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g/dissimilar/fix_rep/dqn_lta_aux/info/sweep/",
     "best": "2"
     },
    {"label": "FTA+Decoder",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g/dissimilar/fix_rep/dqn_lta_aux/input_decoder/sweep/",
     "best": "2"
     },
    {"label": "FTA+NAS",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g/dissimilar/fix_rep/dqn_lta_aux/nas_v2_delta/sweep/",
     "best": "2"
     },
    {"label": "FTA+Reward",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g/dissimilar/fix_rep/dqn_lta_aux/reward/sweep/",
     "best": "2"
     },
    {"label": "FTA+SF",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g/dissimilar/fix_rep/dqn_lta_aux/successor_as/sweep/",
     "best": "2"
     },

    {"label": "Random",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g/dissimilar/fix_rep/random/sweep/",
     "best": "0"
     },
    {"label": "Input",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g/dissimilar/fix_rep/input/sweep/",
     "best": "3"
     },
    {"label": "Scratch",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g/dissimilar/scratch/dqn/sweep/",
     "best": "2"
     },
]

maze_target_diff_best_v12_15g = [
    {"label": "ReLU",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_15g/dissimilar/fix_rep/dqn/best/",
     "best": "2"
     },
    {"label": "ReLU+XY",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_15g/dissimilar/fix_rep/dqn_aux/info/best/",
     "best": "2"
     },
    {"label": "ReLU+Decoder",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_15g/dissimilar/fix_rep/dqn_aux/input_decoder/best/",
     "best": "2"
     },
    {"label": "ReLU+NAS",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_15g/dissimilar/fix_rep/dqn_aux/nas_v2_delta/best/",
     "best": "2"
     },
    {"label": "ReLU+Reward",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_15g/dissimilar/fix_rep/dqn_aux/reward/best/",
     "best": "1"
     },
    {"label": "ReLU+SF",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_15g/dissimilar/fix_rep/dqn_aux/successor_as/best/",
     "best": "1"
     },
    {"label": "FTA eta=0.2",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_15g/dissimilar/fix_rep/dqn_lta/eta_study_0.2_best/",
     "best": "2"
     },
    {"label": "FTA eta=0.4",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_15g/dissimilar/fix_rep/dqn_lta/eta_study_0.4_best/",
     "best": "2"
     },
    {"label": "FTA eta=0.6",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_15g/dissimilar/fix_rep/dqn_lta/eta_study_0.6_best/",
     "best": "2"
     },
    {"label": "FTA eta=0.8",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_15g/dissimilar/fix_rep/dqn_lta/eta_study_0.8_best/",
     "best": "2"
     },
    {"label": "FTA+XY",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_15g/dissimilar/fix_rep/dqn_lta_aux/info/best/",
     "best": "2"
     },
    {"label": "FTA+Decoder",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_15g/dissimilar/fix_rep/dqn_lta_aux/input_decoder/best/",
     "best": "2"
     },
    {"label": "FTA+NAS",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_15g/dissimilar/fix_rep/dqn_lta_aux/nas_v2_delta/best/",
     "best": "2"
     },
    {"label": "FTA+Reward",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_15g/dissimilar/fix_rep/dqn_lta_aux/reward/best/",
     "best": "2"
     },
    {"label": "FTA+SF",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_15g/dissimilar/fix_rep/dqn_lta_aux/successor_as/best/",
     "best": "2"
     },

    {"label": "Random",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_15g/dissimilar/fix_rep/random/best/",
     "best": "0"
     },
    {"label": "Input",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_15g/dissimilar/fix_rep/input/best/",
     "best": "3"
     },
    {"label": "Scratch",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_15g/dissimilar/scratch/dqn/best/",
     "best": "2"
     },
]

maze_target_diff_best_v12_30g = [
   {"label": "ReLU",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g/dissimilar/fix_rep/dqn/best/",
     "best": "2"
     },
    {"label": "ReLU+XY",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g/dissimilar/fix_rep/dqn_aux/info/best/",
     "best": "2"
     },
    {"label": "ReLU+Decoder",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g/dissimilar/fix_rep/dqn_aux/input_decoder/best/",
     "best": "2"
     },
    {"label": "ReLU+NAS",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g/dissimilar/fix_rep/dqn_aux/nas_v2_delta/best/",
     "best": "2"
     },
    {"label": "ReLU+Reward",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g/dissimilar/fix_rep/dqn_aux/reward/best/",
     "best": "1"
     },
    {"label": "ReLU+SF",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g/dissimilar/fix_rep/dqn_aux/successor_as/best/",
     "best": "1"
     },
    {"label": "FTA eta=0.2",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g/dissimilar/fix_rep/dqn_lta/eta_study_0.2_best/",
     "best": "2"
     },
    {"label": "FTA eta=0.4",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g/dissimilar/fix_rep/dqn_lta/eta_study_0.4_best/",
     "best": "2"
     },
    {"label": "FTA eta=0.6",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g/dissimilar/fix_rep/dqn_lta/eta_study_0.6_best/",
     "best": "2"
     },
    {"label": "FTA eta=0.8",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g/dissimilar/fix_rep/dqn_lta/eta_study_0.8_best/",
     "best": "2"
     },
    {"label": "FTA+XY",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g/dissimilar/fix_rep/dqn_lta_aux/info/best/",
     "best": "2"
     },
    {"label": "FTA+Decoder",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g/dissimilar/fix_rep/dqn_lta_aux/input_decoder/best/",
     "best": "2"
     },
    {"label": "FTA+NAS",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g/dissimilar/fix_rep/dqn_lta_aux/nas_v2_delta/best/",
     "best": "2"
     },
    {"label": "FTA+Reward",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g/dissimilar/fix_rep/dqn_lta_aux/reward/best/",
     "best": "2"
     },
    {"label": "FTA+SF",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g/dissimilar/fix_rep/dqn_lta_aux/successor_as/best/",
     "best": "2"
     },

    {"label": "Random",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g/dissimilar/fix_rep/random/best/",
     "best": "0"
     },
    {"label": "Input",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g/dissimilar/fix_rep/input/best/",
     "best": "3"
     },
    {"label": "Scratch",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g/dissimilar/scratch/dqn/best/",
     "best": "2"
     },
]


maze_target_diff_sweep_v12_30g_v7 = [
   {"label": "ReLU",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g_v7/dissimilar/fix_rep/dqn/sweep/",
     "best": "2"
     },
    {"label": "ReLU+XY",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g_v7/dissimilar/fix_rep/dqn_aux/info/sweep/",
     "best": "2"
     },
    {"label": "ReLU+Decoder",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g_v7/dissimilar/fix_rep/dqn_aux/input_decoder/sweep/",
     "best": "2"
     },
    {"label": "ReLU+NAS",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g_v7/dissimilar/fix_rep/dqn_aux/nas_v2_delta/sweep/",
     "best": "2"
     },
    {"label": "ReLU+Reward",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g_v7/dissimilar/fix_rep/dqn_aux/reward/sweep/",
     "best": "1"
     },
    {"label": "ReLU+SF",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g_v7/dissimilar/fix_rep/dqn_aux/successor_as/sweep/",
     "best": "1"
     },
    {"label": "FTA eta=0.2",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g_v7/dissimilar/fix_rep/dqn_lta/eta_study_0.2_sweep/",
     "best": "2"
     },
    {"label": "FTA eta=0.4",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g_v7/dissimilar/fix_rep/dqn_lta/eta_study_0.4_sweep/",
     "best": "2"
     },
    {"label": "FTA eta=0.6",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g_v7/dissimilar/fix_rep/dqn_lta/eta_study_0.6_sweep/",
     "best": "2"
     },
    {"label": "FTA eta=0.8",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g_v7/dissimilar/fix_rep/dqn_lta/eta_study_0.8_sweep/",
     "best": "2"
     },
    {"label": "FTA+XY",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g_v7/dissimilar/fix_rep/dqn_lta_aux/info/sweep/",
     "best": "2"
     },
    {"label": "FTA+Decoder",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g_v7/dissimilar/fix_rep/dqn_lta_aux/input_decoder/sweep/",
     "best": "2"
     },
    {"label": "FTA+NAS",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g_v7/dissimilar/fix_rep/dqn_lta_aux/nas_v2_delta/sweep/",
     "best": "2"
     },
    {"label": "FTA+Reward",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g_v7/dissimilar/fix_rep/dqn_lta_aux/reward/sweep/",
     "best": "2"
     },
    {"label": "FTA+SF",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g_v7/dissimilar/fix_rep/dqn_lta_aux/successor_as/sweep/",
     "best": "2"
     },

    {"label": "Random",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g_v7/dissimilar/fix_rep/random/sweep/",
     "best": "0"
     },
    {"label": "Input",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g_v7/dissimilar/fix_rep/input/sweep/",
     "best": "3"
     },
    {"label": "Scratch",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g_v7/dissimilar/scratch/dqn/sweep/",
     "best": "2"
     },
]
maze_checkpoint50000_same_sweep_v12 = [
    {"label": "ReLU",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g/checkpoints/50000/same/fix_rep/dqn/sweep/",
     },
    {"label": "ReLU+XY",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g/checkpoints/50000/same/fix_rep/dqn_aux/info/sweep/",
     },
    {"label": "ReLU+Decoder",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g/checkpoints/50000/same/fix_rep/dqn_aux/input_decoder/sweep/",
     },
    {"label": "ReLU+NAS",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g/checkpoints/50000/same/fix_rep/dqn_aux/nas_v2_delta/sweep/",
     },
    {"label": "ReLU+Reward",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g/checkpoints/50000/same/fix_rep/dqn_aux/reward/sweep/",
     },
    {"label": "ReLU+SF",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g/checkpoints/50000/same/fix_rep/dqn_aux/successor_as/sweep/",
     },
    {"label": "FTA eta=0.2",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g/checkpoints/50000/same/fix_rep/dqn_lta/eta_study_0.2_sweep/",
     },
    {"label": "FTA eta=0.4",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g/checkpoints/50000/same/fix_rep/dqn_lta/eta_study_0.4_sweep/",
     },
    {"label": "FTA eta=0.6",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g/checkpoints/50000/same/fix_rep/dqn_lta/eta_study_0.6_sweep/",
     },
    {"label": "FTA eta=0.8",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g/checkpoints/50000/same/fix_rep/dqn_lta/eta_study_0.8_sweep/",
     },
    {"label": "FTA+XY",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g/checkpoints/50000/same/fix_rep/dqn_lta_aux/info/sweep/",
     },
    {"label": "FTA+Decoder",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g/checkpoints/50000/same/fix_rep/dqn_lta_aux/input_decoder/sweep/",
     },
    {"label": "FTA+NAS",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g/checkpoints/50000/same/fix_rep/dqn_lta_aux/nas_v2_delta/sweep/",
     },
    {"label": "FTA+Reward",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g/checkpoints/50000/same/fix_rep/dqn_lta_aux/reward/sweep/",
     },
    {"label": "FTA+SF",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g/checkpoints/50000/same/fix_rep/dqn_lta_aux/successor_as/sweep/",
     },
    {"label": "Scratch",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/source_task/dqn/best/",
     },
]
maze_checkpoint50000_same_best_v12 = [
    {"label": "ReLU",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g/checkpoints/50000/same/fix_rep/dqn/best/",
     },
    {"label": "ReLU+XY",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g/checkpoints/50000/same/fix_rep/dqn_aux/info/best/",
     },
    {"label": "ReLU+Decoder",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g/checkpoints/50000/same/fix_rep/dqn_aux/input_decoder/best/",
     },
    {"label": "ReLU+NAS",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g/checkpoints/50000/same/fix_rep/dqn_aux/nas_v2_delta/best/",
     },
    {"label": "ReLU+Reward",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g/checkpoints/50000/same/fix_rep/dqn_aux/reward/best/",
     },
    {"label": "ReLU+SF",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g/checkpoints/50000/same/fix_rep/dqn_aux/successor_as/best/",
     },
    {"label": "FTA eta=0.2",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g/checkpoints/50000/same/fix_rep/dqn_lta/eta_study_0.2_best/",
     },
    {"label": "FTA eta=0.4",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g/checkpoints/50000/same/fix_rep/dqn_lta/eta_study_0.4_best/",
     },
    {"label": "FTA eta=0.6",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g/checkpoints/50000/same/fix_rep/dqn_lta/eta_study_0.6_best/",
     },
    {"label": "FTA eta=0.8",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g/checkpoints/50000/same/fix_rep/dqn_lta/eta_study_0.8_best/",
     },
    {"label": "FTA+XY",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g/checkpoints/50000/same/fix_rep/dqn_lta_aux/info/best/",
     },
    {"label": "FTA+Decoder",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g/checkpoints/50000/same/fix_rep/dqn_lta_aux/input_decoder/best/",
     },
    {"label": "FTA+NAS",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g/checkpoints/50000/same/fix_rep/dqn_lta_aux/nas_v2_delta/best/",
     },
    {"label": "FTA+Reward",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g/checkpoints/50000/same/fix_rep/dqn_lta_aux/reward/best/",
     },
    {"label": "FTA+SF",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g/checkpoints/50000/same/fix_rep/dqn_lta_aux/successor_as/best/",
     },
    {"label": "Scratch",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/source_task/dqn/best/",
     },
]

maze_checkpoint50000_dissimilar_sweep_v12 = [
    {"label": "ReLU",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g/checkpoints/50000/dissimilar/fix_rep/dqn/sweep/",
     "best": "2"
     },
    {"label": "ReLU+XY",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g/checkpoints/50000/dissimilar/fix_rep/dqn_aux/info/sweep/",
     "best": "2"
     },
    {"label": "ReLU+Decoder",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g/checkpoints/50000/dissimilar/fix_rep/dqn_aux/input_decoder/sweep/",
     "best": "2"
     },
    {"label": "ReLU+NAS",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g/checkpoints/50000/dissimilar/fix_rep/dqn_aux/nas_v2_delta/sweep/",
     "best": "2"
     },
    {"label": "ReLU+Reward",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g/checkpoints/50000/dissimilar/fix_rep/dqn_aux/reward/sweep/",
     "best": "2"
     },
    {"label": "ReLU+SF",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g/checkpoints/50000/dissimilar/fix_rep/dqn_aux/successor_as/sweep/",
     "best": "1"
     },
    {"label": "FTA eta=0.2",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g/checkpoints/50000/dissimilar/fix_rep/dqn_lta/eta_study_0.2_sweep/",
     "best": "2"
     },
    {"label": "FTA eta=0.4",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g/checkpoints/50000/dissimilar/fix_rep/dqn_lta/eta_study_0.4_sweep/",
     "best": "3"
     },
    {"label": "FTA eta=0.6",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g/checkpoints/50000/dissimilar/fix_rep/dqn_lta/eta_study_0.6_sweep/",
     "best": "3"
     },
    {"label": "FTA eta=0.8",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g/checkpoints/50000/dissimilar/fix_rep/dqn_lta/eta_study_0.8_sweep/",
     "best": "2"
     },
    {"label": "FTA+XY",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g/checkpoints/50000/dissimilar/fix_rep/dqn_lta_aux/info/sweep/",
     "best": "4"
     },
    {"label": "FTA+Decoder",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g/checkpoints/50000/dissimilar/fix_rep/dqn_lta_aux/input_decoder/sweep/",
     "best": "4"
     },
    {"label": "FTA+NAS",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g/checkpoints/50000/dissimilar/fix_rep/dqn_lta_aux/nas_v2_delta/sweep/",
     "best": "3"
     },
    {"label": "FTA+Reward",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g/checkpoints/50000/dissimilar/fix_rep/dqn_lta_aux/reward/sweep/",
     "best": "3"
     },
    {"label": "FTA+SF",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g/checkpoints/50000/dissimilar/fix_rep/dqn_lta_aux/successor_as/sweep/",
     "best": "4"
     }
]
maze_checkpoint50000_dissimilar_best_v12 = [
    {"label": "ReLU",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g/checkpoints/50000/dissimilar/fix_rep/dqn/best/",
     "best": "2"
     },
    {"label": "ReLU+XY",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g/checkpoints/50000/dissimilar/fix_rep/dqn_aux/info/best/",
     "best": "2"
     },
    {"label": "ReLU+Decoder",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g/checkpoints/50000/dissimilar/fix_rep/dqn_aux/input_decoder/best/",
     "best": "2"
     },
    {"label": "ReLU+NAS",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g/checkpoints/50000/dissimilar/fix_rep/dqn_aux/nas_v2_delta/best/",
     "best": "2"
     },
    {"label": "ReLU+Reward",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g/checkpoints/50000/dissimilar/fix_rep/dqn_aux/reward/best/",
     "best": "2"
     },
    {"label": "ReLU+SF",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g/checkpoints/50000/dissimilar/fix_rep/dqn_aux/successor_as/best/",
     "best": "1"
     },
    {"label": "FTA eta=0.2",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g/checkpoints/50000/dissimilar/fix_rep/dqn_lta/eta_study_0.2_best/",
     "best": "2"
     },
    {"label": "FTA eta=0.4",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g/checkpoints/50000/dissimilar/fix_rep/dqn_lta/eta_study_0.4_best/",
     "best": "3"
     },
    {"label": "FTA eta=0.6",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g/checkpoints/50000/dissimilar/fix_rep/dqn_lta/eta_study_0.6_best/",
     "best": "3"
     },
    {"label": "FTA eta=0.8",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g/checkpoints/50000/dissimilar/fix_rep/dqn_lta/eta_study_0.8_best/",
     "best": "2"
     },
    {"label": "FTA+XY",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g/checkpoints/50000/dissimilar/fix_rep/dqn_lta_aux/info/best/",
     "best": "4"
     },
    {"label": "FTA+Decoder",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g/checkpoints/50000/dissimilar/fix_rep/dqn_lta_aux/input_decoder/best/",
     "best": "4"
     },
    {"label": "FTA+NAS",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g/checkpoints/50000/dissimilar/fix_rep/dqn_lta_aux/nas_v2_delta/best/",
     "best": "3"
     },
    {"label": "FTA+Reward",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g/checkpoints/50000/dissimilar/fix_rep/dqn_lta_aux/reward/best/",
     "best": "3"
     },
    {"label": "FTA+SF",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g/checkpoints/50000/dissimilar/fix_rep/dqn_lta_aux/successor_as/best/",
     "best": "4"
     }
]


maze_checkpoint100000_same_sweep_v12 = [
    {"label": "FTA eta=0.2",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g/checkpoints/100000/same/fix_rep/dqn_lta/eta_study_0.2_sweep/",
     },
    {"label": "FTA eta=0.4",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g/checkpoints/100000/same/fix_rep/dqn_lta/eta_study_0.4_sweep/",
     },
    {"label": "FTA eta=0.6",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g/checkpoints/100000/same/fix_rep/dqn_lta/eta_study_0.6_sweep/",
     },
    {"label": "FTA eta=0.8",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g/checkpoints/100000/same/fix_rep/dqn_lta/eta_study_0.8_sweep/",
     },
    {"label": "FTA+XY",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g/checkpoints/100000/same/fix_rep/dqn_lta_aux/info/sweep/",
     },
    {"label": "FTA+NAS",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g/checkpoints/100000/same/fix_rep/dqn_lta_aux/nas_v2_delta/sweep/",
     },
    {"label": "FTA+Reward",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g/checkpoints/100000/same/fix_rep/dqn_lta_aux/reward/sweep/",
     },
    {"label": "FTA+SF",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g/checkpoints/100000/same/fix_rep/dqn_lta_aux/successor_as/sweep/",
     },
    {"label": "Scratch",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/source_task/dqn/best/",
     },
]
maze_checkpoint100000_same_best_v12 = [
    {"label": "FTA eta=0.2",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g/checkpoints/100000/same/fix_rep/dqn_lta/eta_study_0.2_best/",
     },
    {"label": "FTA eta=0.4",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g/checkpoints/100000/same/fix_rep/dqn_lta/eta_study_0.4_best/",
     },
    {"label": "FTA eta=0.6",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g/checkpoints/100000/same/fix_rep/dqn_lta/eta_study_0.6_best/",
     },
    {"label": "FTA eta=0.8",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g/checkpoints/100000/same/fix_rep/dqn_lta/eta_study_0.8_best/",
     },
    {"label": "FTA+XY",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g/checkpoints/100000/same/fix_rep/dqn_lta_aux/info/best/",
     },
    {"label": "FTA+NAS",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g/checkpoints/100000/same/fix_rep/dqn_lta_aux/nas_v2_delta/best/",
     },
    {"label": "FTA+Reward",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g/checkpoints/100000/same/fix_rep/dqn_lta_aux/reward/best/",
     },
    {"label": "FTA+SF",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g/checkpoints/100000/same/fix_rep/dqn_lta_aux/successor_as/best/",
     },
    {"label": "Scratch",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/source_task/dqn/best/",
     },
]

maze_checkpoint100000_dissimilar_sweep_v12 = [
    {"label": "FTA eta=0.2",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g/checkpoints/100000/dissimilar/fix_rep/dqn_lta/eta_study_0.2_sweep/",
     "best": "2"
     },
    {"label": "FTA eta=0.4",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g/checkpoints/100000/dissimilar/fix_rep/dqn_lta/eta_study_0.4_sweep/",
     "best": "3"
     },
    {"label": "FTA eta=0.6",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g/checkpoints/100000/dissimilar/fix_rep/dqn_lta/eta_study_0.6_sweep/",
     "best": "3"
     },
    {"label": "FTA eta=0.8",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g/checkpoints/100000/dissimilar/fix_rep/dqn_lta/eta_study_0.8_sweep/",
     "best": "2"
     },
    {"label": "FTA+XY",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g/checkpoints/100000/dissimilar/fix_rep/dqn_lta_aux/info/sweep/",
     "best": "4"
     },
    {"label": "FTA+NAS",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g/checkpoints/100000/dissimilar/fix_rep/dqn_lta_aux/nas_v2_delta/sweep/",
     "best": "3"
     },
    {"label": "FTA+Reward",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g/checkpoints/100000/dissimilar/fix_rep/dqn_lta_aux/reward/sweep/",
     "best": "3"
     },
    {"label": "FTA+SF",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g/checkpoints/100000/dissimilar/fix_rep/dqn_lta_aux/successor_as/sweep/",
     "best": "4"
     }
]
maze_checkpoint100000_dissimilar_best_v12 = [
    {"label": "FTA eta=0.2",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g/checkpoints/100000/dissimilar/fix_rep/dqn_lta/eta_study_0.2_best/",
     "best": "2"
     },
    {"label": "FTA eta=0.4",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g/checkpoints/100000/dissimilar/fix_rep/dqn_lta/eta_study_0.4_best/",
     "best": "3"
     },
    {"label": "FTA eta=0.6",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g/checkpoints/100000/dissimilar/fix_rep/dqn_lta/eta_study_0.6_best/",
     "best": "3"
     },
    {"label": "FTA eta=0.8",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g/checkpoints/100000/dissimilar/fix_rep/dqn_lta/eta_study_0.8_best/",
     "best": "2"
     },
    {"label": "FTA+XY",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g/checkpoints/100000/dissimilar/fix_rep/dqn_lta_aux/info/best/",
     "best": "4"
     },
    {"label": "FTA+NAS",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g/checkpoints/100000/dissimilar/fix_rep/dqn_lta_aux/nas_v2_delta/best/",
     "best": "3"
     },
    {"label": "FTA+Reward",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g/checkpoints/100000/dissimilar/fix_rep/dqn_lta_aux/reward/best/",
     "best": "3"
     },
    {"label": "FTA+SF",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g/checkpoints/100000/dissimilar/fix_rep/dqn_lta_aux/successor_as/best/",
     "best": "4"
     }
]


maze_checkpoint150000_same_sweep_v12 = [
    {"label": "ReLU",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g/checkpoints/150000/same/fix_rep/dqn/sweep/",
     "best": "2"
     },
    {"label": "ReLU+XY",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g/checkpoints/150000/same/fix_rep/dqn_aux/info/sweep/",
     "best": "2"
     },
    {"label": "ReLU+Decoder",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g/checkpoints/150000/same/fix_rep/dqn_aux/input_decoder/sweep/",
     "best": "2"
     },
    {"label": "ReLU+NAS",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g/checkpoints/150000/same/fix_rep/dqn_aux/nas_v2_delta/sweep/",
     "best": "2"
     },
    {"label": "ReLU+Reward",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g/checkpoints/150000/same/fix_rep/dqn_aux/reward/sweep/",
     "best": "2"
     },
    {"label": "ReLU+SF",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g/checkpoints/150000/same/fix_rep/dqn_aux/successor_as/sweep/",
     "best": "1"
     },
    {"label": "FTA eta=0.2",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g/checkpoints/150000/same/fix_rep/dqn_lta/eta_study_0.2_sweep/",
     "best": "2"
     },
    {"label": "FTA eta=0.4",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g/checkpoints/150000/same/fix_rep/dqn_lta/eta_study_0.4_sweep/",
     "best": "3"
     },
    {"label": "FTA eta=0.6",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g/checkpoints/150000/same/fix_rep/dqn_lta/eta_study_0.6_sweep/",
     "best": "3"
     },
    {"label": "FTA eta=0.8",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g/checkpoints/150000/same/fix_rep/dqn_lta/eta_study_0.8_sweep/",
     "best": "2"
     },
    {"label": "FTA+XY",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g/checkpoints/150000/same/fix_rep/dqn_lta_aux/info/sweep/",
     "best": "4"
     },
    {"label": "FTA+Decoder",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g/checkpoints/150000/same/fix_rep/dqn_lta_aux/input_decoder/sweep/",
     "best": "4"
     },
    {"label": "FTA+NAS",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g/checkpoints/150000/same/fix_rep/dqn_lta_aux/nas_v2_delta/sweep/",
     "best": "3"
     },
    {"label": "FTA+Reward",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g/checkpoints/150000/same/fix_rep/dqn_lta_aux/reward/sweep/",
     "best": "3"
     },
    {"label": "FTA+SF",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g/checkpoints/150000/same/fix_rep/dqn_lta_aux/successor_as/sweep/",
     "best": "4"
     },
    {"label": "Scratch",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/source_task/dqn/best/",
     "best": "1"
     },
]
maze_checkpoint150000_same_best_v12 = [
    {"label": "ReLU",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g/checkpoints/150000/same/fix_rep/dqn/best/",
     "best": "2"
     },
    {"label": "ReLU+XY",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g/checkpoints/150000/same/fix_rep/dqn_aux/info/best/",
     "best": "2"
     },
    {"label": "ReLU+Decoder",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g/checkpoints/150000/same/fix_rep/dqn_aux/input_decoder/best/",
     "best": "2"
     },
    {"label": "ReLU+NAS",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g/checkpoints/150000/same/fix_rep/dqn_aux/nas_v2_delta/best/",
     "best": "2"
     },
    {"label": "ReLU+Reward",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g/checkpoints/150000/same/fix_rep/dqn_aux/reward/best/",
     "best": "2"
     },
    {"label": "ReLU+SF",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g/checkpoints/150000/same/fix_rep/dqn_aux/successor_as/best/",
     "best": "1"
     },
    {"label": "FTA eta=0.2",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g/checkpoints/150000/same/fix_rep/dqn_lta/eta_study_0.2_best/",
     "best": "2"
     },
    {"label": "FTA eta=0.4",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g/checkpoints/150000/same/fix_rep/dqn_lta/eta_study_0.4_best/",
     "best": "3"
     },
    {"label": "FTA eta=0.6",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g/checkpoints/150000/same/fix_rep/dqn_lta/eta_study_0.6_best/",
     "best": "3"
     },
    {"label": "FTA eta=0.8",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g/checkpoints/150000/same/fix_rep/dqn_lta/eta_study_0.8_best/",
     "best": "2"
     },
    {"label": "FTA+XY",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g/checkpoints/150000/same/fix_rep/dqn_lta_aux/info/best/",
     "best": "4"
     },
    {"label": "FTA+Decoder",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g/checkpoints/150000/same/fix_rep/dqn_lta_aux/input_decoder/best/",
     "best": "4"
     },
    {"label": "FTA+NAS",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g/checkpoints/150000/same/fix_rep/dqn_lta_aux/nas_v2_delta/best/",
     "best": "3"
     },
    {"label": "FTA+Reward",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g/checkpoints/150000/same/fix_rep/dqn_lta_aux/reward/best/",
     "best": "3"
     },
    {"label": "FTA+SF",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g/checkpoints/150000/same/fix_rep/dqn_lta_aux/successor_as/best/",
     "best": "4"
     },
    {"label": "Scratch",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/source_task/dqn/best/",
     "best": "1"
     },
]

maze_checkpoint150000_dissimilar_sweep_v12 = [
    {"label": "ReLU",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g/checkpoints/150000/dissimilar/fix_rep/dqn/sweep/",
     "best": "2"
     },
    {"label": "ReLU+XY",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g/checkpoints/150000/dissimilar/fix_rep/dqn_aux/info/sweep/",
     "best": "2"
     },
    {"label": "ReLU+Decoder",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g/checkpoints/150000/dissimilar/fix_rep/dqn_aux/input_decoder/sweep/",
     "best": "2"
     },
    {"label": "ReLU+NAS",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g/checkpoints/150000/dissimilar/fix_rep/dqn_aux/nas_v2_delta/sweep/",
     "best": "2"
     },
    {"label": "ReLU+Reward",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g/checkpoints/150000/dissimilar/fix_rep/dqn_aux/reward/sweep/",
     "best": "2"
     },
    {"label": "ReLU+SF",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g/checkpoints/150000/dissimilar/fix_rep/dqn_aux/successor_as/sweep/",
     "best": "1"
     },
    {"label": "FTA eta=0.2",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g/checkpoints/150000/dissimilar/fix_rep/dqn_lta/eta_study_0.2_sweep/",
     "best": "2"
     },
    {"label": "FTA eta=0.4",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g/checkpoints/150000/dissimilar/fix_rep/dqn_lta/eta_study_0.4_sweep/",
     "best": "3"
     },
    {"label": "FTA eta=0.6",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g/checkpoints/150000/dissimilar/fix_rep/dqn_lta/eta_study_0.6_sweep/",
     "best": "3"
     },
    {"label": "FTA eta=0.8",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g/checkpoints/150000/dissimilar/fix_rep/dqn_lta/eta_study_0.8_sweep/",
     "best": "2"
     },
    {"label": "FTA+XY",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g/checkpoints/150000/dissimilar/fix_rep/dqn_lta_aux/info/sweep/",
     "best": "4"
     },
    {"label": "FTA+Decoder",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g/checkpoints/150000/dissimilar/fix_rep/dqn_lta_aux/input_decoder/sweep/",
     "best": "4"
     },
    {"label": "FTA+NAS",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g/checkpoints/150000/dissimilar/fix_rep/dqn_lta_aux/nas_v2_delta/sweep/",
     "best": "3"
     },
    {"label": "FTA+Reward",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g/checkpoints/150000/dissimilar/fix_rep/dqn_lta_aux/reward/sweep/",
     "best": "3"
     },
    {"label": "FTA+SF",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g/checkpoints/150000/dissimilar/fix_rep/dqn_lta_aux/successor_as/sweep/",
     "best": "4"
     }
]
maze_checkpoint150000_dissimilar_best_v12 = [
    {"label": "ReLU",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g/checkpoints/150000/dissimilar/fix_rep/dqn/best/",
     "best": "2"
     },
    {"label": "ReLU+XY",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g/checkpoints/150000/dissimilar/fix_rep/dqn_aux/info/best/",
     "best": "2"
     },
    {"label": "ReLU+Decoder",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g/checkpoints/150000/dissimilar/fix_rep/dqn_aux/input_decoder/best/",
     "best": "2"
     },
    {"label": "ReLU+NAS",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g/checkpoints/150000/dissimilar/fix_rep/dqn_aux/nas_v2_delta/best/",
     "best": "2"
     },
    {"label": "ReLU+Reward",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g/checkpoints/150000/dissimilar/fix_rep/dqn_aux/reward/best/",
     "best": "2"
     },
    {"label": "ReLU+SF",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g/checkpoints/150000/dissimilar/fix_rep/dqn_aux/successor_as/best/",
     "best": "1"
     },
    {"label": "FTA eta=0.2",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g/checkpoints/150000/dissimilar/fix_rep/dqn_lta/eta_study_0.2_best/",
     "best": "2"
     },
    {"label": "FTA eta=0.4",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g/checkpoints/150000/dissimilar/fix_rep/dqn_lta/eta_study_0.4_best/",
     "best": "3"
     },
    {"label": "FTA eta=0.6",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g/checkpoints/150000/dissimilar/fix_rep/dqn_lta/eta_study_0.6_best/",
     "best": "3"
     },
    {"label": "FTA eta=0.8",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g/checkpoints/150000/dissimilar/fix_rep/dqn_lta/eta_study_0.8_best/",
     "best": "2"
     },
    {"label": "FTA+XY",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g/checkpoints/150000/dissimilar/fix_rep/dqn_lta_aux/info/best/",
     "best": "4"
     },
    {"label": "FTA+Decoder",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g/checkpoints/150000/dissimilar/fix_rep/dqn_lta_aux/input_decoder/best/",
     "best": "4"
     },
    {"label": "FTA+NAS",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g/checkpoints/150000/dissimilar/fix_rep/dqn_lta_aux/nas_v2_delta/best/",
     "best": "3"
     },
    {"label": "FTA+Reward",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g/checkpoints/150000/dissimilar/fix_rep/dqn_lta_aux/reward/best/",
     "best": "3"
     },
    {"label": "FTA+SF",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g/checkpoints/150000/dissimilar/fix_rep/dqn_lta_aux/successor_as/best/",
     "best": "4"
     }
]


maze_checkpoint200000_same_sweep_v12 = [
    {"label": "FTA eta=0.2",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g/checkpoints/200000/same/fix_rep/dqn_lta/eta_study_0.2_sweep/",
     },
    {"label": "FTA eta=0.4",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g/checkpoints/200000/same/fix_rep/dqn_lta/eta_study_0.4_sweep/",
     },
    {"label": "FTA eta=0.6",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g/checkpoints/200000/same/fix_rep/dqn_lta/eta_study_0.6_sweep/",
     },
    {"label": "FTA eta=0.8",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g/checkpoints/200000/same/fix_rep/dqn_lta/eta_study_0.8_sweep/",
     },
    {"label": "FTA+XY",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g/checkpoints/200000/same/fix_rep/dqn_lta_aux/info/sweep/",
     },
    {"label": "FTA+NAS",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g/checkpoints/200000/same/fix_rep/dqn_lta_aux/nas_v2_delta/sweep/",
     },
    {"label": "FTA+Reward",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g/checkpoints/200000/same/fix_rep/dqn_lta_aux/reward/sweep/",
     },
    {"label": "FTA+SF",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g/checkpoints/200000/same/fix_rep/dqn_lta_aux/successor_as/sweep/",
     },
    {"label": "Scratch",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/source_task/dqn/best/",
     },
]
maze_checkpoint200000_same_best_v12 = [
    {"label": "FTA eta=0.2",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g/checkpoints/200000/same/fix_rep/dqn_lta/eta_study_0.2_best/",
     },
    {"label": "FTA eta=0.4",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g/checkpoints/200000/same/fix_rep/dqn_lta/eta_study_0.4_best/",
     },
    {"label": "FTA eta=0.6",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g/checkpoints/200000/same/fix_rep/dqn_lta/eta_study_0.6_best/",
     },
    {"label": "FTA eta=0.8",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g/checkpoints/200000/same/fix_rep/dqn_lta/eta_study_0.8_best/",
     },
    {"label": "FTA+XY",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g/checkpoints/200000/same/fix_rep/dqn_lta_aux/info/best/",
     },
    {"label": "FTA+NAS",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g/checkpoints/200000/same/fix_rep/dqn_lta_aux/nas_v2_delta/best/",
     },
    {"label": "FTA+Reward",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g/checkpoints/200000/same/fix_rep/dqn_lta_aux/reward/best/",
     },
    {"label": "FTA+SF",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g/checkpoints/200000/same/fix_rep/dqn_lta_aux/successor_as/best/",
     },
    {"label": "Scratch",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/source_task/dqn/best/",
     },
]

maze_checkpoint200000_dissimilar_sweep_v12 = [
    {"label": "FTA eta=0.2",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g/checkpoints/200000/dissimilar/fix_rep/dqn_lta/eta_study_0.2_sweep/",
     "best": "2"
     },
    {"label": "FTA eta=0.4",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g/checkpoints/200000/dissimilar/fix_rep/dqn_lta/eta_study_0.4_sweep/",
     "best": "3"
     },
    {"label": "FTA eta=0.6",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g/checkpoints/200000/dissimilar/fix_rep/dqn_lta/eta_study_0.6_sweep/",
     "best": "3"
     },
    {"label": "FTA eta=0.8",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g/checkpoints/200000/dissimilar/fix_rep/dqn_lta/eta_study_0.8_sweep/",
     "best": "2"
     },
    {"label": "FTA+XY",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g/checkpoints/200000/dissimilar/fix_rep/dqn_lta_aux/info/sweep/",
     "best": "4"
     },
    {"label": "FTA+NAS",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g/checkpoints/200000/dissimilar/fix_rep/dqn_lta_aux/nas_v2_delta/sweep/",
     "best": "3"
     },
    {"label": "FTA+Reward",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g/checkpoints/200000/dissimilar/fix_rep/dqn_lta_aux/reward/sweep/",
     "best": "3"
     },
    {"label": "FTA+SF",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g/checkpoints/200000/dissimilar/fix_rep/dqn_lta_aux/successor_as/sweep/",
     "best": "4"
     }
]
maze_checkpoint200000_dissimilar_best_v12 = [
    {"label": "FTA eta=0.2",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g/checkpoints/200000/dissimilar/fix_rep/dqn_lta/eta_study_0.2_best/",
     "best": "2"
     },
    {"label": "FTA eta=0.4",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g/checkpoints/200000/dissimilar/fix_rep/dqn_lta/eta_study_0.4_best/",
     "best": "3"
     },
    {"label": "FTA eta=0.6",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g/checkpoints/200000/dissimilar/fix_rep/dqn_lta/eta_study_0.6_best/",
     "best": "3"
     },
    {"label": "FTA eta=0.8",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g/checkpoints/200000/dissimilar/fix_rep/dqn_lta/eta_study_0.8_best/",
     "best": "2"
     },
    {"label": "FTA+XY",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g/checkpoints/200000/dissimilar/fix_rep/dqn_lta_aux/info/best/",
     "best": "4"
     },
    {"label": "FTA+NAS",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g/checkpoints/200000/dissimilar/fix_rep/dqn_lta_aux/nas_v2_delta/best/",
     "best": "3"
     },
    {"label": "FTA+Reward",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g/checkpoints/200000/dissimilar/fix_rep/dqn_lta_aux/reward/best/",
     "best": "3"
     },
    {"label": "FTA+SF",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g/checkpoints/200000/dissimilar/fix_rep/dqn_lta_aux/successor_as/best/",
     "best": "4"
     }
]


maze_checkpoint250000_same_sweep_v12 = [
    {"label": "FTA eta=0.2",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g/checkpoints/250000/same/fix_rep/dqn_lta/eta_study_0.2_sweep/",
     },
    {"label": "FTA eta=0.4",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g/checkpoints/250000/same/fix_rep/dqn_lta/eta_study_0.4_sweep/",
     },
    {"label": "FTA eta=0.6",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g/checkpoints/250000/same/fix_rep/dqn_lta/eta_study_0.6_sweep/",
     },
    {"label": "FTA eta=0.8",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g/checkpoints/250000/same/fix_rep/dqn_lta/eta_study_0.8_sweep/",
     },
    {"label": "FTA+XY",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g/checkpoints/250000/same/fix_rep/dqn_lta_aux/info/sweep/",
     },
    {"label": "FTA+NAS",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g/checkpoints/250000/same/fix_rep/dqn_lta_aux/nas_v2_delta/sweep/",
     },
    {"label": "FTA+Reward",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g/checkpoints/250000/same/fix_rep/dqn_lta_aux/reward/sweep/",
     },
    {"label": "FTA+SF",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g/checkpoints/250000/same/fix_rep/dqn_lta_aux/successor_as/sweep/",
     },
    {"label": "Scratch",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/source_task/dqn/best/",
     },
]
maze_checkpoint250000_same_best_v12 = [
    {"label": "FTA eta=0.2",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g/checkpoints/250000/same/fix_rep/dqn_lta/eta_study_0.2_best/",
     },
    {"label": "FTA eta=0.4",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g/checkpoints/250000/same/fix_rep/dqn_lta/eta_study_0.4_best/",
     },
    {"label": "FTA eta=0.6",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g/checkpoints/250000/same/fix_rep/dqn_lta/eta_study_0.6_best/",
     },
    {"label": "FTA eta=0.8",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g/checkpoints/250000/same/fix_rep/dqn_lta/eta_study_0.8_best/",
     },
    {"label": "FTA+XY",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g/checkpoints/250000/same/fix_rep/dqn_lta_aux/info/best/",
     },
    {"label": "FTA+NAS",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g/checkpoints/250000/same/fix_rep/dqn_lta_aux/nas_v2_delta/best/",
     },
    {"label": "FTA+Reward",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g/checkpoints/250000/same/fix_rep/dqn_lta_aux/reward/best/",
     },
    {"label": "FTA+SF",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g/checkpoints/250000/same/fix_rep/dqn_lta_aux/successor_as/best/",
     },
    {"label": "Scratch",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/source_task/dqn/best/",
     },
]

maze_checkpoint250000_dissimilar_sweep_v12 = [
    {"label": "FTA eta=0.2",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g/checkpoints/250000/dissimilar/fix_rep/dqn_lta/eta_study_0.2_sweep/",
     "best": "2"
     },
    {"label": "FTA eta=0.4",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g/checkpoints/250000/dissimilar/fix_rep/dqn_lta/eta_study_0.4_sweep/",
     "best": "3"
     },
    {"label": "FTA eta=0.6",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g/checkpoints/250000/dissimilar/fix_rep/dqn_lta/eta_study_0.6_sweep/",
     "best": "3"
     },
    {"label": "FTA eta=0.8",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g/checkpoints/250000/dissimilar/fix_rep/dqn_lta/eta_study_0.8_sweep/",
     "best": "2"
     },
    {"label": "FTA+XY",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g/checkpoints/250000/dissimilar/fix_rep/dqn_lta_aux/info/sweep/",
     "best": "4"
     },
    {"label": "FTA+NAS",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g/checkpoints/250000/dissimilar/fix_rep/dqn_lta_aux/nas_v2_delta/sweep/",
     "best": "3"
     },
    {"label": "FTA+Reward",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g/checkpoints/250000/dissimilar/fix_rep/dqn_lta_aux/reward/sweep/",
     "best": "3"
     },
    {"label": "FTA+SF",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g/checkpoints/250000/dissimilar/fix_rep/dqn_lta_aux/successor_as/sweep/",
     "best": "4"
     }
]
maze_checkpoint250000_dissimilar_best_v12 = [
    {"label": "FTA eta=0.2",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g/checkpoints/250000/dissimilar/fix_rep/dqn_lta/eta_study_0.2_best/",
     "best": "2"
     },
    {"label": "FTA eta=0.4",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g/checkpoints/250000/dissimilar/fix_rep/dqn_lta/eta_study_0.4_best/",
     "best": "3"
     },
    {"label": "FTA eta=0.6",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g/checkpoints/250000/dissimilar/fix_rep/dqn_lta/eta_study_0.6_best/",
     "best": "3"
     },
    {"label": "FTA eta=0.8",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g/checkpoints/250000/dissimilar/fix_rep/dqn_lta/eta_study_0.8_best/",
     "best": "2"
     },
    {"label": "FTA+XY",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g/checkpoints/250000/dissimilar/fix_rep/dqn_lta_aux/info/best/",
     "best": "4"
     },
    {"label": "FTA+NAS",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g/checkpoints/250000/dissimilar/fix_rep/dqn_lta_aux/nas_v2_delta/best/",
     "best": "3"
     },
    {"label": "FTA+Reward",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g/checkpoints/250000/dissimilar/fix_rep/dqn_lta_aux/reward/best/",
     "best": "3"
     },
    {"label": "FTA+SF",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g/checkpoints/250000/dissimilar/fix_rep/dqn_lta_aux/successor_as/best/",
     "best": "4"
     }
]


maze_checkpoint300000_same_sweep_v12 = [
    {"label": "FTA eta=0.2",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g/checkpoints/300000/same/fix_rep/dqn_lta/eta_study_0.2_sweep/",
     },
    {"label": "FTA eta=0.4",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g/checkpoints/300000/same/fix_rep/dqn_lta/eta_study_0.4_sweep/",
     },
    {"label": "FTA eta=0.6",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g/checkpoints/300000/same/fix_rep/dqn_lta/eta_study_0.6_sweep/",
     },
    {"label": "FTA eta=0.8",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g/checkpoints/300000/same/fix_rep/dqn_lta/eta_study_0.8_sweep/",
     },
    {"label": "FTA+XY",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g/checkpoints/300000/same/fix_rep/dqn_lta_aux/info/sweep/",
     },
    {"label": "FTA+NAS",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g/checkpoints/300000/same/fix_rep/dqn_lta_aux/nas_v2_delta/sweep/",
     },
    {"label": "FTA+Reward",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g/checkpoints/300000/same/fix_rep/dqn_lta_aux/reward/sweep/",
     },
    {"label": "FTA+SF",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g/checkpoints/300000/same/fix_rep/dqn_lta_aux/successor_as/sweep/",
     },
    {"label": "Scratch",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/source_task/dqn/best/",
     },
]
maze_checkpoint300000_same_best_v12 = [
    {"label": "FTA eta=0.2",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g/checkpoints/300000/same/fix_rep/dqn_lta/eta_study_0.2_best/",
     },
    {"label": "FTA eta=0.4",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g/checkpoints/300000/same/fix_rep/dqn_lta/eta_study_0.4_best/",
     },
    {"label": "FTA eta=0.6",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g/checkpoints/300000/same/fix_rep/dqn_lta/eta_study_0.6_best/",
     },
    {"label": "FTA eta=0.8",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g/checkpoints/300000/same/fix_rep/dqn_lta/eta_study_0.8_best/",
     },
    {"label": "FTA+XY",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g/checkpoints/300000/same/fix_rep/dqn_lta_aux/info/best/",
     },
    {"label": "FTA+NAS",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g/checkpoints/300000/same/fix_rep/dqn_lta_aux/nas_v2_delta/best/",
     },
    {"label": "FTA+Reward",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g/checkpoints/300000/same/fix_rep/dqn_lta_aux/reward/best/",
     },
    {"label": "FTA+SF",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g/checkpoints/300000/same/fix_rep/dqn_lta_aux/successor_as/best/",
     },
    {"label": "Scratch",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/source_task/dqn/best/",
     },
]

maze_checkpoint300000_dissimilar_sweep_v12 = [
    {"label": "FTA eta=0.2",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g/checkpoints/300000/dissimilar/fix_rep/dqn_lta/eta_study_0.2_sweep/",
     "best": "2"
     },
    {"label": "FTA eta=0.4",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g/checkpoints/300000/dissimilar/fix_rep/dqn_lta/eta_study_0.4_sweep/",
     "best": "3"
     },
    {"label": "FTA eta=0.6",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g/checkpoints/300000/dissimilar/fix_rep/dqn_lta/eta_study_0.6_sweep/",
     "best": "3"
     },
    {"label": "FTA eta=0.8",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g/checkpoints/300000/dissimilar/fix_rep/dqn_lta/eta_study_0.8_sweep/",
     "best": "2"
     },
    {"label": "FTA+XY",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g/checkpoints/300000/dissimilar/fix_rep/dqn_lta_aux/info/sweep/",
     "best": "4"
     },
    {"label": "FTA+NAS",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g/checkpoints/300000/dissimilar/fix_rep/dqn_lta_aux/nas_v2_delta/sweep/",
     "best": "3"
     },
    {"label": "FTA+Reward",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g/checkpoints/300000/dissimilar/fix_rep/dqn_lta_aux/reward/sweep/",
     "best": "3"
     },
    {"label": "FTA+SF",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g/checkpoints/300000/dissimilar/fix_rep/dqn_lta_aux/successor_as/sweep/",
     "best": "4"
     }
]
maze_checkpoint300000_dissimilar_best_v12 = [
    {"label": "FTA eta=0.2",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g/checkpoints/300000/dissimilar/fix_rep/dqn_lta/eta_study_0.2_best/",
     "best": "2"
     },
    {"label": "FTA eta=0.4",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g/checkpoints/300000/dissimilar/fix_rep/dqn_lta/eta_study_0.4_best/",
     "best": "3"
     },
    {"label": "FTA eta=0.6",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g/checkpoints/300000/dissimilar/fix_rep/dqn_lta/eta_study_0.6_best/",
     "best": "3"
     },
    {"label": "FTA eta=0.8",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g/checkpoints/300000/dissimilar/fix_rep/dqn_lta/eta_study_0.8_best/",
     "best": "2"
     },
    {"label": "FTA+XY",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g/checkpoints/300000/dissimilar/fix_rep/dqn_lta_aux/info/best/",
     "best": "4"
     },
    {"label": "FTA+NAS",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g/checkpoints/300000/dissimilar/fix_rep/dqn_lta_aux/nas_v2_delta/best/",
     "best": "3"
     },
    {"label": "FTA+Reward",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g/checkpoints/300000/dissimilar/fix_rep/dqn_lta_aux/reward/best/",
     "best": "3"
     },
    {"label": "FTA+SF",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g/checkpoints/300000/dissimilar/fix_rep/dqn_lta_aux/successor_as/best/",
     "best": "4"
     }
]


mazesimple_notarget_same_sweep_v12 = [
    {"label": "FTA eta=0.2",
     "control": "data/output/test_v12/maze/linear_vf/target_task_notarget/same/fix_rep/dqn_lta/eta_study_0.2_sweep/",
     },
    {"label": "FTA eta=0.4",
     "control": "data/output/test_v12/maze/linear_vf/target_task_notarget/same/fix_rep/dqn_lta/eta_study_0.4_sweep/",
     },
    {"label": "FTA eta=0.6",
     "control": "data/output/test_v12/maze/linear_vf/target_task_notarget/same/fix_rep/dqn_lta/eta_study_0.6_sweep/",
     },
    {"label": "FTA eta=0.8",
     "control": "data/output/test_v12/maze/linear_vf/target_task_notarget/same/fix_rep/dqn_lta/eta_study_0.8_sweep/",
     },
    {"label": "FTA+XY",
     "control": "data/output/test_v12/maze/linear_vf/target_task_notarget/same/fix_rep/dqn_lta_aux/info/sweep/",
     },
    {"label": "FTA+NAS",
     "control": "data/output/test_v12/maze/linear_vf/target_task_notarget/same/fix_rep/dqn_lta_aux/nas_v2_delta/sweep/",
     },
    {"label": "FTA+Reward",
     "control": "data/output/test_v12/maze/linear_vf/target_task_notarget/same/fix_rep/dqn_lta_aux/reward/sweep/",
     },
    {"label": "FTA+SF",
     "control": "data/output/test_v12/maze/linear_vf/target_task_notarget/same/fix_rep/dqn_lta_aux/successor_as/sweep/",
     },
    {"label": "Scratch ReLU",
     "control": "data/output/test_v12/maze/linear_vf/target_task_notarget/same/scratch/dqn/sweep/",
     },
    # {"label": "Scratch FTA",
    #  "control": "data/output/test_v12/maze/linear_vf/target_task_notarget/same/scratch/dqn_lta/eta_study_0.2_sweep/",
    #  }
]

mazesimple_qlearning_same_sweep_v12 = [
    {"label": "FTA eta=0.2",
     "control": "data/output/test_v12/maze/linear_vf/target_task_qlearning/same/fix_rep/dqn_lta/eta_study_0.2_sweep/",
     },
    {"label": "FTA eta=0.4",
     "control": "data/output/test_v12/maze/linear_vf/target_task_qlearning/same/fix_rep/dqn_lta/eta_study_0.4_sweep/",
     },
    {"label": "FTA eta=0.6",
     "control": "data/output/test_v12/maze/linear_vf/target_task_qlearning/same/fix_rep/dqn_lta/eta_study_0.6_sweep/",
     },
    {"label": "FTA eta=0.8",
     "control": "data/output/test_v12/maze/linear_vf/target_task_qlearning/same/fix_rep/dqn_lta/eta_study_0.8_sweep/",
     },
    {"label": "FTA+XY",
     "control": "data/output/test_v12/maze/linear_vf/target_task_qlearning/same/fix_rep/dqn_lta_aux/info/sweep/",
     },
    {"label": "FTA+NAS",
     "control": "data/output/test_v12/maze/linear_vf/target_task_qlearning/same/fix_rep/dqn_lta_aux/nas_v2_delta/sweep/",
     },
    {"label": "FTA+Reward",
     "control": "data/output/test_v12/maze/linear_vf/target_task_qlearning/same/fix_rep/dqn_lta_aux/reward/sweep/",
     },
    {"label": "FTA+SF",
     "control": "data/output/test_v12/maze/linear_vf/target_task_qlearning/same/fix_rep/dqn_lta_aux/successor_as/sweep/",
     },
    {"label": "Scratch ReLU",
     "control": "data/output/test_v12/maze/linear_vf/target_task_qlearning/same/scratch/qlearning/sweep/",
     },
    {"label": "Scratch FTA",
     "control": "data/output/test_v12/maze/linear_vf/target_task_qlearning/same/scratch/qlearning_lta/eta_study_0.2_sweep/",
     }
]
mazesimple_qlearning_same_best_v12 = [
    {"label": "FTA eta=0.2",
     "control": "data/output/test_v12/maze/linear_vf/target_task_qlearning/same/fix_rep/dqn_lta/eta_study_0.2_best/",
     },
    {"label": "FTA eta=0.4",
     "control": "data/output/test_v12/maze/linear_vf/target_task_qlearning/same/fix_rep/dqn_lta/eta_study_0.4_best/",
     },
    {"label": "FTA eta=0.6",
     "control": "data/output/test_v12/maze/linear_vf/target_task_qlearning/same/fix_rep/dqn_lta/eta_study_0.6_best/",
     },
    {"label": "FTA eta=0.8",
     "control": "data/output/test_v12/maze/linear_vf/target_task_qlearning/same/fix_rep/dqn_lta/eta_study_0.8_best/",
     },
    {"label": "FTA+XY",
     "control": "data/output/test_v12/maze/linear_vf/target_task_qlearning/same/fix_rep/dqn_lta_aux/info/best/",
     },
    {"label": "FTA+NAS",
     "control": "data/output/test_v12/maze/linear_vf/target_task_qlearning/same/fix_rep/dqn_lta_aux/nas_v2_delta/best/",
     },
    {"label": "FTA+Reward",
     "control": "data/output/test_v12/maze/linear_vf/target_task_qlearning/same/fix_rep/dqn_lta_aux/reward/best/",
     },
    {"label": "FTA+SF",
     "control": "data/output/test_v12/maze/linear_vf/target_task_qlearning/same/fix_rep/dqn_lta_aux/successor_as/best/",
     },
    {"label": "Scratch ReLU",
     "control": "data/output/test_v12/maze/linear_vf/target_task_qlearning/same/scratch/qlearning_lta/best/",
     },
    {"label": "Scratch FTA",
     "control": "data/output/test_v12/maze/linear_vf/target_task_qlearning/same/scratch/qlearning_lta/eta_study_0.2_best/",
     }
]


maze_multigoal_notarget_same_sweep_v12 = [
    {"label": "FTA eta=0.2",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g_notarget/same/fix_rep/dqn_lta/eta_study_0.2_sweep/",
     },
    {"label": "FTA eta=0.4",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g_notarget/same/fix_rep/dqn_lta/eta_study_0.4_sweep/",
     },
    {"label": "FTA eta=0.6",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g_notarget/same/fix_rep/dqn_lta/eta_study_0.6_sweep/",
     },
    {"label": "FTA eta=0.8",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g_notarget/same/fix_rep/dqn_lta/eta_study_0.8_sweep/",
     },
    {"label": "FTA+XY",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g_notarget/same/fix_rep/dqn_lta_aux/info/sweep/",
     },
    {"label": "FTA+NAS",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g_notarget/same/fix_rep/dqn_lta_aux/nas_v2_delta/sweep/",
     },
    {"label": "FTA+Reward",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g_notarget/same/fix_rep/dqn_lta_aux/reward/sweep/",
     },
    {"label": "FTA+SF",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g_notarget/same/fix_rep/dqn_lta_aux/successor_as/sweep/",
     },
    # {"label": "Scratch ReLU",
    #  "control": "data/output/test_v12/maze_multigoal/linear_vf/source_task/dqn/sweep/",
    #  },
    # {"label": "Scratch ReLU",
    #  "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g_notarget/same/scratch/dqn/sweep/",
    #  },
    # {"label": "Scratch FTA",
    #  "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g_notarget/same/scratch/dqn_lta/eta_study_0.2_sweep/",
    #  }
]
maze_multigoal_notarget_same_best_v12 = [
    {"label": "FTA eta=0.2",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g_notarget/same/fix_rep/dqn_lta/eta_study_0.2_best/",
     },
    {"label": "FTA eta=0.4",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g_notarget/same/fix_rep/dqn_lta/eta_study_0.4_best/",
     },
    {"label": "FTA eta=0.6",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g_notarget/same/fix_rep/dqn_lta/eta_study_0.6_best/",
     },
    {"label": "FTA eta=0.8",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g_notarget/same/fix_rep/dqn_lta/eta_study_0.8_best/",
     },
    {"label": "FTA+XY",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g_notarget/same/fix_rep/dqn_lta_aux/info/best/",
     },
    {"label": "FTA+NAS",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g_notarget/same/fix_rep/dqn_lta_aux/nas_v2_delta/best/",
     },
    {"label": "FTA+Reward",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g_notarget/same/fix_rep/dqn_lta_aux/reward/best/",
     },
    {"label": "FTA+SF",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g_notarget/same/fix_rep/dqn_lta_aux/successor_as/best/",
     },
    {"label": "Scratch",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/source_task/dqn/best/",
     },
    # {"label": "Scratch(ReLU)",
    #  "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g_notarget/same/scratch/dqn/best/",
    #  },
    # {"label": "Scratch(FTA)",
    #  "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g_notarget/same/scratch/dqn_lta/eta_study_0.2_best/",
    #  }
]

maze_multigoal_notarget_diff_sweep_v12 = [
    {"label": "FTA eta=0.2",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g_notarget/dissimilar/fix_rep/dqn_lta/eta_study_0.2_sweep/",
     },
    {"label": "FTA eta=0.4",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g_notarget/dissimilar/fix_rep/dqn_lta/eta_study_0.4_sweep/",
     },
    {"label": "FTA eta=0.6",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g_notarget/dissimilar/fix_rep/dqn_lta/eta_study_0.6_sweep/",
     },
    {"label": "FTA eta=0.8",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g_notarget/dissimilar/fix_rep/dqn_lta/eta_study_0.8_sweep/",
     },
    {"label": "FTA+XY",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g_notarget/dissimilar/fix_rep/dqn_lta_aux/info/sweep/",
     },
    {"label": "FTA+NAS",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g_notarget/dissimilar/fix_rep/dqn_lta_aux/nas_v2_delta/sweep/",
     },
    {"label": "FTA+Reward",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g_notarget/dissimilar/fix_rep/dqn_lta_aux/reward/sweep/",
     },
    {"label": "FTA+SF",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g_notarget/dissimilar/fix_rep/dqn_lta_aux/successor_as/sweep/",
     },
    {"label": "Scratch(ReLU)",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g/dissimilar/scratch/dqn/sweep/",
     },
    # {"label": "Scratch(FTA)",
    #  "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g/dissimilar/scratch/dqn_lta/eta_study_0.2_sweep/",
    #  }
]

maze_multigoal_qlearning_same_sweep_v12 = [
    {"label": "FTA eta=0.2",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g_qlearning/same/fix_rep/dqn_lta/eta_study_0.2_sweep/",
     },
    {"label": "FTA eta=0.4",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g_qlearning/same/fix_rep/dqn_lta/eta_study_0.4_sweep/",
     },
    {"label": "FTA eta=0.6",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g_qlearning/same/fix_rep/dqn_lta/eta_study_0.6_sweep/",
     },
    {"label": "FTA eta=0.8",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g_qlearning/same/fix_rep/dqn_lta/eta_study_0.8_sweep/",
     },
    {"label": "FTA+XY",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g_qlearning/same/fix_rep/dqn_lta_aux/info/sweep/",
     },
    {"label": "FTA+NAS",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g_qlearning/same/fix_rep/dqn_lta_aux/nas_v2_delta/sweep/",
     },
    {"label": "FTA+Reward",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g_qlearning/same/fix_rep/dqn_lta_aux/reward/sweep/",
     },
    {"label": "FTA+SF",
     "control": "data/output/test_v12/maze_multigoal/linear_vf/target_task_30g_qlearning/same/fix_rep/dqn_lta_aux/successor_as/sweep/",
     }
]


gh_original_sweep_v13 = [
    # {"label": "ReLU+divConstr w0.01",
    #  "control": "data/output/test_v13/gridhard/linear_vf/original_0909/online_property/dqn_div_constr/sweep_w0.01/",
    #  },
    # {"label": "ReLU+divConstr w0.001",
    #  "control": "data/output/test_v13/gridhard/linear_vf/original_0909/online_property/dqn_div_constr/sweep_w0.001/",
    #  },
    # {"label": "ReLU+divConstr w0.0001",
    #  "control": "data/output/test_v13/gridhard/linear_vf/original_0909/online_property/dqn_div_constr/sweep_w0.0001/",
    #  },
    # {"label": "ReLU+divConstr w0.00001",
    #  "control": "data/output/test_v13/gridhard/linear_vf/original_0909/online_property/dqn_div_constr/sweep_w0.00001/",
    #  },

    {"label": "ReLU",
     "control": "data/output/test_v13/gridhard/linear_vf/original_0909/online_property/dqn/sweep/",
     },
    {"label": "ReLU+VirtualVF1",
     "control": "data/output/test_v13/gridhard/linear_vf/original_0909/online_property/dqn_aux/aux_control/sweep_1g_gamma0.9_slow_sync/",
     },
    {"label": "ReLU+VirtualVF5",
     "control": "data/output/test_v13/gridhard/linear_vf/original_0909/online_property/dqn_aux/aux_control/sweep_5g_gamma0.9_slow_sync_smallw/",
     },
    {"label": "ReLU+Info",
     "control": "data/output/test_v13/gridhard/linear_vf/original_0909/online_property/dqn_aux/info/sweep/",
     },
    {"label": "ReLU+Decoder",
     "control": "data/output/test_v13/gridhard/linear_vf/original_0909/online_property/dqn_aux/input_decoder/sweep/",
     },
    {"label": "ReLU+NAS",
     "control": "data/output/test_v13/gridhard/linear_vf/original_0909/online_property/dqn_aux/nas_v2_delta/sweep/",
     },
    {"label": "ReLU+Reward",
     "control": "data/output/test_v13/gridhard/linear_vf/original_0909/online_property/dqn_aux/reward/sweep/",
     },
    {"label": "ReLU+SF",
     "control": "data/output/test_v13/gridhard/linear_vf/original_0909/online_property/dqn_aux/successor_as/sweep/",
     },
    {"label": "FTA eta=0.2",
     "control": "data/output/test_v13/gridhard/linear_vf/original_0909/online_property/dqn_lta/eta_study_0.2_sweep/",
     },
    {"label": "FTA eta=0.4",
     "control": "data/output/test_v13/gridhard/linear_vf/original_0909/online_property/dqn_lta/eta_study_0.4_sweep/",
     },
    {"label": "FTA eta=0.6",
     "control": "data/output/test_v13/gridhard/linear_vf/original_0909/online_property/dqn_lta/eta_study_0.6_sweep/",
     },
    {"label": "FTA eta=0.8",
     "control": "data/output/test_v13/gridhard/linear_vf/original_0909/online_property/dqn_lta/eta_study_0.8_sweep/",
     },
    {"label": "FTA+VirtualVF1",
     "control": "data/output/test_v13/gridhard/linear_vf/original_0909/online_property/dqn_lta_aux/aux_control/sweep_1g_gamma0.9_slow_sync/",
     },
    {"label": "FTA+VirtualVF5",
     "control": "data/output/test_v13/gridhard/linear_vf/original_0909/online_property/dqn_lta_aux/aux_control/sweep_5g_gamma0.9_slow_sync_smallw/",
     },
    {"label": "FTA+Info",
     "control": "data/output/test_v13/gridhard/linear_vf/original_0909/online_property/dqn_lta_aux/info/sweep/",
     },
    {"label": "FTA+Decoder",
     "control": "data/output/test_v13/gridhard/linear_vf/original_0909/online_property/dqn_lta_aux/input_decoder/sweep/",
     },
    {"label": "FTA+NAS",
     "control": "data/output/test_v13/gridhard/linear_vf/original_0909/online_property/dqn_lta_aux/nas_v2_delta/sweep/",
     },
    {"label": "FTA+Reward",
     "control": "data/output/test_v13/gridhard/linear_vf/original_0909/online_property/dqn_lta_aux/reward/sweep/",
     },
    {"label": "FTA+SF",
     "control": "data/output/test_v13/gridhard/linear_vf/original_0909/online_property/dqn_lta_aux/successor_as/sweep/",
     },
]

gh_transfer_samelr_v13 = [
    {"label": "ReLU",
     "control": "data/output/test_v13/gridhard/linear_vf/original_0909/transfer/goal_id_{}/dqn/same_lr/",
     },
    # {"label": "ReLU+VirtualVF1",
    #  "control": "data/output/test_v13/gridhard/linear_vf/original_0909/transfer/goal_id_{}/dqn_aux/aux_control/same_lr_1g/",
    #  },
    # {"label": "ReLU+VirtualVF5",
    #  "control": "data/output/test_v13/gridhard/linear_vf/original_0909/transfer/goal_id_{}/dqn_aux/aux_control/same_lr_5g/",
    #  },
    {"label": "ReLU+XY",
     "control": "data/output/test_v13/gridhard/linear_vf/original_0909/transfer/goal_id_{}/dqn_aux/info/same_lr/",
     },
    {"label": "ReLU+Decoder",
     "control": "data/output/test_v13/gridhard/linear_vf/original_0909/transfer/goal_id_{}/dqn_aux/input_decoder/same_lr/",
     },
    {"label": "ReLU+NAS",
     "control": "data/output/test_v13/gridhard/linear_vf/original_0909/transfer/goal_id_{}/dqn_aux/nas_v2_delta/same_lr/",
     },
    {"label": "ReLU+Reward",
     "control": "data/output/test_v13/gridhard/linear_vf/original_0909/transfer/goal_id_{}/dqn_aux/reward/same_lr/",
     },
    {"label": "ReLU+SF",
     "control": "data/output/test_v13/gridhard/linear_vf/original_0909/transfer/goal_id_{}/dqn_aux/successor_as/same_lr/",
     },
    # {"label": "FTA eta=0.2",
    #  "control": "data/output/test_v13/gridhard/linear_vf/original_0909/transfer/goal_id_{}/dqn_lta/eta_study_0.2_same_lr/",
    #  },
    # {"label": "FTA eta=0.4",
    #  "control": "data/output/test_v13/gridhard/linear_vf/original_0909/transfer/goal_id_{}/dqn_lta/eta_study_0.4_same_lr/",
    #  },
    # {"label": "FTA eta=0.6",
    #  "control": "data/output/test_v13/gridhard/linear_vf/original_0909/transfer/goal_id_{}/dqn_lta/eta_study_0.6_same_lr/",
    #  },
    {"label": "FTA eta=0.8",
     "control": "data/output/test_v13/gridhard/linear_vf/original_0909/transfer/goal_id_{}/dqn_lta/eta_study_0.8_same_lr/",
     },
    # {"label": "FTA+VirtualVF1",
    #  "control": "data/output/test_v13/gridhard/linear_vf/original_0909/transfer/goal_id_{}/dqn_lta_aux/aux_control/same_lr_1g/",
    #  },
    # {"label": "FTA+VirtualVF5",
    #  "control": "data/output/test_v13/gridhard/linear_vf/original_0909/transfer/goal_id_{}/dqn_lta_aux/aux_control/same_lr_5g/",
    #  },
    # {"label": "FTA+XY",
    #  "control": "data/output/test_v13/gridhard/linear_vf/original_0909/transfer/goal_id_{}/dqn_lta_aux/info/same_lr/",
    #  },
    {"label": "FTA+Decoder",
     "control": "data/output/test_v13/gridhard/linear_vf/original_0909/transfer/goal_id_{}/dqn_lta_aux/input_decoder/same_lr/",
     },
    {"label": "FTA+NAS",
     "control": "data/output/test_v13/gridhard/linear_vf/original_0909/transfer/goal_id_{}/dqn_lta_aux/nas_v2_delta/same_lr/",
     },
    # {"label": "FTA+Reward",
    #  "control": "data/output/test_v13/gridhard/linear_vf/original_0909/transfer/goal_id_{}/dqn_lta_aux/reward/same_lr/",
    #  },
    {"label": "FTA+SF",
     "control": "data/output/test_v13/gridhard/linear_vf/original_0909/transfer/goal_id_{}/dqn_lta_aux/successor_as/same_lr/",
     },
    {"label": "Scratch",
     "control": "data/output/test_v13/gridhard/linear_vf/learning_scratch/goal_id_{}/dqn/sweep/",
     },
]

gh_transfer_sweep_v13 = [
    {"label": "ReLU",
     "control": "data/output/test_v13/gridhard/linear_vf/original_0909/transfer/goal_id_{}/dqn/sweep/",
     "online_measure": ["data/output/test_v13/gridhard/linear_vf/original_0909/online_property/dqn/sweep/", 1]
     },
    {"label": "ReLU+VirtualVF1",
     "control": "data/output/test_v13/gridhard/linear_vf/original_0909/transfer/goal_id_{}/dqn_aux/aux_control/sweep_1g_slow_sync/",
     "online_measure": ["data/output/test_v13/gridhard/linear_vf/original_0909/online_property/dqn_aux/aux_control/sweep_1g_gamma0.9_slow_sync/", 1],
     },
    {"label": "ReLU+VirtualVF5",
     "control": "data/output/test_v13/gridhard/linear_vf/original_0909/transfer/goal_id_{}/dqn_aux/aux_control/sweep_5g_slow_sync_smallw/",
     "online_measure": ["data/output/test_v13/gridhard/linear_vf/original_0909/online_property/dqn_aux/aux_control/sweep_5g_gamma0.9_slow_sync_smallw/", 1],
     },
    {"label": "ReLU+XY",
     "control": "data/output/test_v13/gridhard/linear_vf/original_0909/transfer/goal_id_{}/dqn_aux/info/sweep/",
     "online_measure": ["data/output/test_v13/gridhard/linear_vf/original_0909/online_property/dqn_aux/info/sweep/", 1],
     },
    {"label": "ReLU+Decoder",
     "control": "data/output/test_v13/gridhard/linear_vf/original_0909/transfer/goal_id_{}/dqn_aux/input_decoder/sweep/",
     "online_measure": ["data/output/test_v13/gridhard/linear_vf/original_0909/online_property/dqn_aux/input_decoder/sweep/", 1],
     },
    {"label": "ReLU+NAS",
     "control": "data/output/test_v13/gridhard/linear_vf/original_0909/transfer/goal_id_{}/dqn_aux/nas_v2_delta/sweep/",
     "online_measure": ["data/output/test_v13/gridhard/linear_vf/original_0909/online_property/dqn_aux/nas_v2_delta/sweep/", 1],
     },
    {"label": "ReLU+Reward",
     "control": "data/output/test_v13/gridhard/linear_vf/original_0909/transfer/goal_id_{}/dqn_aux/reward/sweep/",
     "online_measure": ["data/output/test_v13/gridhard/linear_vf/original_0909/online_property/dqn_aux/reward/sweep/", 0],
     },
    {"label": "ReLU+SF",
     "control": "data/output/test_v13/gridhard/linear_vf/original_0909/transfer/goal_id_{}/dqn_aux/successor_as/sweep/",
     "online_measure": ["data/output/test_v13/gridhard/linear_vf/original_0909/online_property/dqn_aux/successor_as/sweep/", 1],
     },
    {"label": "ReLU+ATC",
     "control": "data/output/test_v13/gridhard/linear_vf/original_0909/transfer/goal_id_{}/dqn_cl/sweep/",
     "online_measure": ["data/output/test_v13/gridhard/linear_vf/original_0909/online_property/dqn_cl/best/", 0],
     },
    {"label": "FTA eta=0.2",
     "control": "data/output/test_v13/gridhard/linear_vf/original_0909/transfer/goal_id_{}/dqn_lta/eta_study_0.2_sweep/",
     "online_measure": ["data/output/test_v13/gridhard/linear_vf/original_0909/online_property/dqn_lta/eta_study_0.2_sweep/", 1],
     },
    {"label": "FTA eta=0.4",
     "control": "data/output/test_v13/gridhard/linear_vf/original_0909/transfer/goal_id_{}/dqn_lta/eta_study_0.4_sweep/",
     "online_measure": ["data/output/test_v13/gridhard/linear_vf/original_0909/online_property/dqn_lta/eta_study_0.4_sweep/", 1],
     },
    {"label": "FTA eta=0.6",
     "control": "data/output/test_v13/gridhard/linear_vf/original_0909/transfer/goal_id_{}/dqn_lta/eta_study_0.6_sweep/",
     "online_measure": ["data/output/test_v13/gridhard/linear_vf/original_0909/online_property/dqn_lta/eta_study_0.6_sweep/", 1],
     },
    {"label": "FTA eta=0.8",
     "control": "data/output/test_v13/gridhard/linear_vf/original_0909/transfer/goal_id_{}/dqn_lta/eta_study_0.8_sweep/",
     "online_measure": ["data/output/test_v13/gridhard/linear_vf/original_0909/online_property/dqn_lta/eta_study_0.8_sweep/", 2],
     },
    {"label": "FTA+VirtualVF1",
     "control": "data/output/test_v13/gridhard/linear_vf/original_0909/transfer/goal_id_{}/dqn_lta_aux/aux_control/sweep_1g_slow_sync/",
     "online_measure": ["data/output/test_v13/gridhard/linear_vf/original_0909/online_property/dqn_lta_aux/aux_control/sweep_1g_gamma0.9_slow_sync/", 2],
     },
    {"label": "FTA+VirtualVF5",
     "control": "data/output/test_v13/gridhard/linear_vf/original_0909/transfer/goal_id_{}/dqn_lta_aux/aux_control/sweep_5g_slow_sync_smallw/",
     "online_measure": ["data/output/test_v13/gridhard/linear_vf/original_0909/online_property/dqn_lta_aux/aux_control/sweep_5g_gamma0.9_slow_sync_smallw/", 2],
     },
    {"label": "FTA+XY",
     "control": "data/output/test_v13/gridhard/linear_vf/original_0909/transfer/goal_id_{}/dqn_lta_aux/info/sweep/",
     "online_measure": ["data/output/test_v13/gridhard/linear_vf/original_0909/online_property/dqn_lta_aux/info/sweep/", 1],
     },
    {"label": "FTA+Decoder",
     "control": "data/output/test_v13/gridhard/linear_vf/original_0909/transfer/goal_id_{}/dqn_lta_aux/input_decoder/sweep/",
     "online_measure": ["data/output/test_v13/gridhard/linear_vf/original_0909/online_property/dqn_lta_aux/input_decoder/sweep/", 2],
     },
    {"label": "FTA+NAS",
     "control": "data/output/test_v13/gridhard/linear_vf/original_0909/transfer/goal_id_{}/dqn_lta_aux/nas_v2_delta/sweep/",
     "online_measure": ["data/output/test_v13/gridhard/linear_vf/original_0909/online_property/dqn_lta_aux/nas_v2_delta/sweep/", 2],
     },
    {"label": "FTA+Reward",
     "control": "data/output/test_v13/gridhard/linear_vf/original_0909/transfer/goal_id_{}/dqn_lta_aux/reward/sweep/",
     "online_measure": ["data/output/test_v13/gridhard/linear_vf/original_0909/online_property/dqn_lta_aux/reward/sweep/", 1],
     },
    {"label": "FTA+SF",
     "control": "data/output/test_v13/gridhard/linear_vf/original_0909/transfer/goal_id_{}/dqn_lta_aux/successor_as/sweep/",
     "online_measure": ["data/output/test_v13/gridhard/linear_vf/original_0909/online_property/dqn_lta_aux/successor_as/sweep/", 2],
     },
    {"label": "FTA+ATC",
     "control": "data/output/test_v13/gridhard/linear_vf/original_0909/transfer/goal_id_{}/dqn_fta_cl/sweep/",
     "online_measure": ["data/output/test_v13/gridhard/linear_vf/original_0909/online_property/dqn_fta_cl/best/", 0],
     },
    {"label": "Scratch",
     # "control": "data/output/test_v13/gridhard/linear_vf/learning_scratch/goal_id_{}/dqn/fix_eps_sweep/",
     "control": "data/output/test_v13/gridhard/linear_vf/learning_scratch/goal_id_{}/dqn/sweep/",
     },
    {"label": "Random",
     "control": "data/output/test_v13/gridhard/linear_vf/learning_scratch/goal_id_{}/random/sweep/",
     "online_measure": ["data/output/test_v13/gridhard/linear_vf/fixrep_property/random/best/", 0],
     },
    {"label": "Input",
     "control": "data/output/test_v13/gridhard/linear_vf/learning_scratch/goal_id_{}/input/sweep/",
     "online_measure": ["data/output/test_v13/gridhard/linear_vf/fixrep_property/input/best/", 0],
     },
    {"label": "Scratch(FTA)",
     "control": "data/output/test_v13/gridhard/linear_vf/learning_scratch/goal_id_{}/dqn_lta/eta_study_0.2_sweep/",
     },
    # {"label": "ReLU+divConstr",
    #  "control": "data/output/test_v13/gridhard/linear_vf/original_0909/transfer/goal_id_{}/dqn_div_constr/sweep_w0.0001_last/",
    #  "online_measure": ["data/output/test_v13/gridhard/linear_vf/original_0909/online_property/dqn_div_constr/sweep_w0.0001/", 1],
    #  },
]

gh_nonlinear_original_sweep_v13 = [
    {"label": "ReLU",
     "control": "data/output/test_v13/gridhard/nonlinear_vf/original_0909/online_property/dqn/sweep/",
     },
    {"label": "ReLU+VirtualVF1",
     "control": "data/output/test_v13/gridhard/nonlinear_vf/original_0909/online_property/dqn_aux/aux_control/sweep_1g/",
     },
    {"label": "ReLU+VirtualVF5",
     "control": "data/output/test_v13/gridhard/nonlinear_vf/original_0909/online_property/dqn_aux/aux_control/sweep_5g/",
     },
    {"label": "ReLU+Info",
     "control": "data/output/test_v13/gridhard/nonlinear_vf/original_0909/online_property/dqn_aux/info/sweep/",
     },
    {"label": "ReLU+Decoder",
     "control": "data/output/test_v13/gridhard/nonlinear_vf/original_0909/online_property/dqn_aux/input_decoder/sweep/",
     },
    {"label": "ReLU+NAS",
     "control": "data/output/test_v13/gridhard/nonlinear_vf/original_0909/online_property/dqn_aux/nas_v2_delta/sweep/",
     },
    {"label": "ReLU+Reward",
     "control": "data/output/test_v13/gridhard/nonlinear_vf/original_0909/online_property/dqn_aux/reward/sweep/",
     },
    {"label": "ReLU+SF",
     "control": "data/output/test_v13/gridhard/nonlinear_vf/original_0909/online_property/dqn_aux/successor_as/sweep/",
     },
    {"label": "FTA eta=0.2",
     "control": "data/output/test_v13/gridhard/nonlinear_vf/original_0909/online_property/dqn_lta/eta_study_0.2_sweep/",
     },
    {"label": "FTA eta=0.4",
     "control": "data/output/test_v13/gridhard/nonlinear_vf/original_0909/online_property/dqn_lta/eta_study_0.4_sweep/",
     },
    {"label": "FTA eta=0.6",
     "control": "data/output/test_v13/gridhard/nonlinear_vf/original_0909/online_property/dqn_lta/eta_study_0.6_sweep/",
     },
    {"label": "FTA eta=0.8",
     "control": "data/output/test_v13/gridhard/nonlinear_vf/original_0909/online_property/dqn_lta/eta_study_0.8_sweep/",
     },
    {"label": "FTA+VirtualVF1",
     "control": "data/output/test_v13/gridhard/nonlinear_vf/original_0909/online_property/dqn_lta_aux/aux_control/sweep_1g/",
     },
    {"label": "FTA+VirtualVF5",
     "control": "data/output/test_v13/gridhard/nonlinear_vf/original_0909/online_property/dqn_lta_aux/aux_control/sweep_5g/",
     },
    {"label": "FTA+Info",
     "control": "data/output/test_v13/gridhard/nonlinear_vf/original_0909/online_property/dqn_lta_aux/info/sweep/",
     },
    {"label": "FTA+Decoder",
     "control": "data/output/test_v13/gridhard/nonlinear_vf/original_0909/online_property/dqn_lta_aux/input_decoder/sweep/",
     },
    {"label": "FTA+NAS",
     "control": "data/output/test_v13/gridhard/nonlinear_vf/original_0909/online_property/dqn_lta_aux/nas_v2_delta/sweep/",
     },
    {"label": "FTA+Reward",
     "control": "data/output/test_v13/gridhard/nonlinear_vf/original_0909/online_property/dqn_lta_aux/reward/sweep/",
     },
    {"label": "FTA+SF",
     "control": "data/output/test_v13/gridhard/nonlinear_vf/original_0909/online_property/dqn_lta_aux/successor_as/sweep/",
     },
]

gh_nonlinear_transfer_sweep_v13 = [
    {"label": "ReLU",
     "control": "data/output/test_v13/gridhard/nonlinear_vf/original_0909/transfer/goal_id_{}/dqn/sweep/",
     "online_measure": ["data/output/test_v13/gridhard/nonlinear_vf/original_0909/online_property/dqn/sweep/", 3]
     },
    {"label": "ReLU+VirtualVF1",
     "control": "data/output/test_v13/gridhard/nonlinear_vf/original_0909/transfer/goal_id_{}/dqn_aux/aux_control/sweep_1g/",
     "online_measure": ["data/output/test_v13/gridhard/nonlinear_vf/original_0909/online_property/dqn_aux/aux_control/sweep_1g/", 2],
     },
    {"label": "ReLU+VirtualVF5",
     "control": "data/output/test_v13/gridhard/nonlinear_vf/original_0909/transfer/goal_id_{}/dqn_aux/aux_control/sweep_5g/",
     "online_measure": ["data/output/test_v13/gridhard/nonlinear_vf/original_0909/online_property/dqn_aux/aux_control/sweep_5g/", 2],
     },
    {"label": "ReLU+XY",
     "control": "data/output/test_v13/gridhard/nonlinear_vf/original_0909/transfer/goal_id_{}/dqn_aux/info/sweep/",
     "online_measure": ["data/output/test_v13/gridhard/nonlinear_vf/original_0909/online_property/dqn_aux/info/sweep/", 2],
     },
    {"label": "ReLU+Decoder",
     "control": "data/output/test_v13/gridhard/nonlinear_vf/original_0909/transfer/goal_id_{}/dqn_aux/input_decoder/sweep/",
     "online_measure": ["data/output/test_v13/gridhard/nonlinear_vf/original_0909/online_property/dqn_aux/input_decoder/sweep/", 3],
     },
    {"label": "ReLU+NAS",
     "control": "data/output/test_v13/gridhard/nonlinear_vf/original_0909/transfer/goal_id_{}/dqn_aux/nas_v2_delta/sweep/",
     "online_measure": ["data/output/test_v13/gridhard/nonlinear_vf/original_0909/online_property/dqn_aux/nas_v2_delta/sweep/", 3],
     },
    {"label": "ReLU+Reward",
     "control": "data/output/test_v13/gridhard/nonlinear_vf/original_0909/transfer/goal_id_{}/dqn_aux/reward/sweep/",
     "online_measure": ["data/output/test_v13/gridhard/nonlinear_vf/original_0909/online_property/dqn_aux/reward/sweep/", 2],
     },
    {"label": "ReLU+SF",
     "control": "data/output/test_v13/gridhard/nonlinear_vf/original_0909/transfer/goal_id_{}/dqn_aux/successor_as/sweep/",
     "online_measure": ["data/output/test_v13/gridhard/nonlinear_vf/original_0909/online_property/dqn_aux/successor_as/sweep/", 3],
     },
    {"label": "FTA eta=0.2",
     "control": "data/output/test_v13/gridhard/nonlinear_vf/original_0909/transfer/goal_id_{}/dqn_lta/eta_study_0.2_sweep/",
     "online_measure": ["data/output/test_v13/gridhard/nonlinear_vf/original_0909/online_property/dqn_lta/eta_study_0.2_sweep/", 2],
     },
    {"label": "FTA eta=0.4",
     "control": "data/output/test_v13/gridhard/nonlinear_vf/original_0909/transfer/goal_id_{}/dqn_lta/eta_study_0.4_sweep/",
     "online_measure": ["data/output/test_v13/gridhard/nonlinear_vf/original_0909/online_property/dqn_lta/eta_study_0.4_sweep/", 1],
     },
    {"label": "FTA eta=0.6",
     "control": "data/output/test_v13/gridhard/nonlinear_vf/original_0909/transfer/goal_id_{}/dqn_lta/eta_study_0.6_sweep/",
     "online_measure": ["data/output/test_v13/gridhard/nonlinear_vf/original_0909/online_property/dqn_lta/eta_study_0.6_sweep/", 2],
     },
    {"label": "FTA eta=0.8",
     "control": "data/output/test_v13/gridhard/nonlinear_vf/original_0909/transfer/goal_id_{}/dqn_lta/eta_study_0.8_sweep/",
     "online_measure": ["data/output/test_v13/gridhard/nonlinear_vf/original_0909/online_property/dqn_lta/eta_study_0.8_sweep/", 2],
     },
    {"label": "FTA+VirtualVF1",
     "control": "data/output/test_v13/gridhard/nonlinear_vf/original_0909/transfer/goal_id_{}/dqn_lta_aux/aux_control/sweep_1g/",
     "online_measure": ["data/output/test_v13/gridhard/nonlinear_vf/original_0909/online_property/dqn_lta_aux/aux_control/sweep_1g/", 1],
     },
    {"label": "FTA+VirtualVF5",
     "control": "data/output/test_v13/gridhard/nonlinear_vf/original_0909/transfer/goal_id_{}/dqn_lta_aux/aux_control/sweep_5g/",
     "online_measure": ["data/output/test_v13/gridhard/nonlinear_vf/original_0909/online_property/dqn_lta_aux/aux_control/sweep_5g/", 1],
     },
    {"label": "FTA+XY",
     "control": "data/output/test_v13/gridhard/nonlinear_vf/original_0909/transfer/goal_id_{}/dqn_lta_aux/info/sweep/",
     "online_measure": ["data/output/test_v13/gridhard/nonlinear_vf/original_0909/online_property/dqn_lta_aux/info/sweep/", 1],
     },
    {"label": "FTA+Decoder",
     "control": "data/output/test_v13/gridhard/nonlinear_vf/original_0909/transfer/goal_id_{}/dqn_lta_aux/input_decoder/sweep/",
     "online_measure": ["data/output/test_v13/gridhard/nonlinear_vf/original_0909/online_property/dqn_lta_aux/input_decoder/sweep/", 1],
     },
    {"label": "FTA+NAS",
     "control": "data/output/test_v13/gridhard/nonlinear_vf/original_0909/transfer/goal_id_{}/dqn_lta_aux/nas_v2_delta/sweep/",
     "online_measure": ["data/output/test_v13/gridhard/nonlinear_vf/original_0909/online_property/dqn_lta_aux/nas_v2_delta/sweep/", 2],
     },
    {"label": "FTA+Reward",
     "control": "data/output/test_v13/gridhard/nonlinear_vf/original_0909/transfer/goal_id_{}/dqn_lta_aux/reward/sweep/",
     "online_measure": ["data/output/test_v13/gridhard/nonlinear_vf/original_0909/online_property/dqn_lta_aux/reward/sweep/", 2],
     },
    {"label": "FTA+SF",
     "control": "data/output/test_v13/gridhard/nonlinear_vf/original_0909/transfer/goal_id_{}/dqn_lta_aux/successor_as/sweep/",
     "online_measure": ["data/output/test_v13/gridhard/nonlinear_vf/original_0909/online_property/dqn_lta_aux/successor_as/sweep/", 2],
     },
    {"label": "Scratch",
     "control": "data/output/test_v13/gridhard/nonlinear_vf/learning_scratch/goal_id_{}/dqn/sweep/",
     },
    {"label": "Random",
     "control": "data/output/test_v13/gridhard/nonlinear_vf/learning_scratch/goal_id_{}/random/sweep/",
     "online_measure": ["data/output/test_v13/gridhard/nonlinear_vf/fixrep_property/random/best/", 0],
     },
    {"label": "Input",
     "control": "data/output/test_v13/gridhard/nonlinear_vf/learning_scratch/goal_id_{}/input/sweep/",
     "online_measure": ["data/output/test_v13/gridhard/nonlinear_vf/fixrep_property/input/best/", 0],
     },
]

gh_nonlinear_original_sweep_v13_largeReLU = [
    {"label": "ReLU",
     "control": "data/output/test_v13/gridhard/nonlinear_vf/original_0909/online_property/dqn_large/sweep/",
     },
    {"label": "ReLU+VirtualVF1",
     "control": "data/output/test_v13/gridhard/nonlinear_vf/original_0909/online_property/dqn_large_aux/aux_control/sweep_1g/",
     },
    {"label": "ReLU+VirtualVF5",
     "control": "data/output/test_v13/gridhard/nonlinear_vf/original_0909/online_property/dqn_large_aux/aux_control/sweep_5g/",
     },
    {"label": "ReLU+Info",
     "control": "data/output/test_v13/gridhard/nonlinear_vf/original_0909/online_property/dqn_large_aux/info/sweep/",
     },
    {"label": "ReLU+Decoder",
     "control": "data/output/test_v13/gridhard/nonlinear_vf/original_0909/online_property/dqn_large_aux/input_decoder/sweep/",
     },
    {"label": "ReLU+NAS",
     "control": "data/output/test_v13/gridhard/nonlinear_vf/original_0909/online_property/dqn_large_aux/nas_v2_delta/sweep/",
     },
    {"label": "ReLU+Reward",
     "control": "data/output/test_v13/gridhard/nonlinear_vf/original_0909/online_property/dqn_large_aux/reward/sweep/",
     },
    {"label": "ReLU+SF",
     "control": "data/output/test_v13/gridhard/nonlinear_vf/original_0909/online_property/dqn_large_aux/successor_as/sweep/",
     }
]

gh_nonlinear_transfer_sweep_v13_largeReLU_wrongInterf = [
    {"label": "ReLU",
     "control": "data/output/test_v13/gridhard/nonlinear_vf/original_0909/transfer/goal_id_{}/dqn/sweep/",
     "online_measure": ["data/output/test_v13/gridhard/nonlinear_vf/original_0909/online_property/dqn/sweep/", 3],
     "fixrep_measure": ["data/output/test_v13/gridhard/nonlinear_vf/original_0909/fixrep_property/dqn/sweep/", 0]
     },
    {"label": "ReLU+VirtualVF1",
     "control": "data/output/test_v13/gridhard/nonlinear_vf/original_0909/transfer/goal_id_{}/dqn_aux/aux_control/sweep_1g/",
     "online_measure": ["data/output/test_v13/gridhard/nonlinear_vf/original_0909/online_property/dqn_aux/aux_control/sweep_1g/", 2],
     "fixrep_measure": ["data/output/test_v13/gridhard/nonlinear_vf/original_0909/fixrep_property/dqn_aux/aux_control/sweep_1g/", 0],
     },
    {"label": "ReLU+VirtualVF5",
     "control": "data/output/test_v13/gridhard/nonlinear_vf/original_0909/transfer/goal_id_{}/dqn_aux/aux_control/sweep_5g/",
     "online_measure": ["data/output/test_v13/gridhard/nonlinear_vf/original_0909/online_property/dqn_aux/aux_control/sweep_5g/", 2],
     "fixrep_measure": ["data/output/test_v13/gridhard/nonlinear_vf/original_0909/fixrep_property/dqn_aux/aux_control/sweep_5g/", 0],
     },
    {"label": "ReLU+XY",
     "control": "data/output/test_v13/gridhard/nonlinear_vf/original_0909/transfer/goal_id_{}/dqn_aux/info/sweep/",
     "online_measure": ["data/output/test_v13/gridhard/nonlinear_vf/original_0909/online_property/dqn_aux/info/sweep/", 2],
     "fixrep_measure": ["data/output/test_v13/gridhard/nonlinear_vf/original_0909/fixrep_property/dqn_aux/info/sweep/", 0],
     },
    {"label": "ReLU+Decoder",
     "control": "data/output/test_v13/gridhard/nonlinear_vf/original_0909/transfer/goal_id_{}/dqn_aux/input_decoder/sweep/",
     "online_measure": ["data/output/test_v13/gridhard/nonlinear_vf/original_0909/online_property/dqn_aux/input_decoder/sweep/", 3],
     "fixrep_measure": ["data/output/test_v13/gridhard/nonlinear_vf/original_0909/fixrep_property/dqn_aux/input_decoder/sweep/", 0],
     },
    {"label": "ReLU+NAS",
     "control": "data/output/test_v13/gridhard/nonlinear_vf/original_0909/transfer/goal_id_{}/dqn_aux/nas_v2_delta/sweep/",
     "online_measure": ["data/output/test_v13/gridhard/nonlinear_vf/original_0909/online_property/dqn_aux/nas_v2_delta/sweep/", 3],
     "fixrep_measure": ["data/output/test_v13/gridhard/nonlinear_vf/original_0909/fixrep_property/dqn_aux/nas_v2_delta/sweep/", 0],
     },
    {"label": "ReLU+Reward",
     "control": "data/output/test_v13/gridhard/nonlinear_vf/original_0909/transfer/goal_id_{}/dqn_aux/reward/sweep/",
     "online_measure": ["data/output/test_v13/gridhard/nonlinear_vf/original_0909/online_property/dqn_aux/reward/sweep/", 2],
     "fixrep_measure": ["data/output/test_v13/gridhard/nonlinear_vf/original_0909/fixrep_property/dqn_aux/reward/sweep/", 0],
     },
    {"label": "ReLU+SF",
     "control": "data/output/test_v13/gridhard/nonlinear_vf/original_0909/transfer/goal_id_{}/dqn_aux/successor_as/sweep/",
     "online_measure": ["data/output/test_v13/gridhard/nonlinear_vf/original_0909/online_property/dqn_aux/successor_as/sweep/", 3],
     "fixrep_measure": ["data/output/test_v13/gridhard/nonlinear_vf/original_0909/fixrep_property/dqn_aux/successor_as/sweep/", 0],
     },
    {"label": "ReLU(L)",
     "control": "data/output/test_v13/gridhard/nonlinear_vf/original_0909/transfer/goal_id_{}/dqn_large/sweep/",
     "online_measure": ["data/output/test_v13/gridhard/nonlinear_vf/original_0909/online_property/dqn_large/sweep/", 3],
     "fixrep_measure": ["data/output/test_v13/gridhard/nonlinear_vf/original_0909/fixrep_property/dqn_large/sweep/", 0]
     },
    {"label": "ReLU(L)+VirtualVF1",
     "control": "data/output/test_v13/gridhard/nonlinear_vf/original_0909/transfer/goal_id_{}/dqn_large_aux/aux_control/sweep_1g/",
     "online_measure": ["data/output/test_v13/gridhard/nonlinear_vf/original_0909/online_property/dqn_large_aux/aux_control/sweep_1g/", 3],
     "fixrep_measure": ["data/output/test_v13/gridhard/nonlinear_vf/original_0909/fixrep_property/dqn_large_aux/aux_control/sweep_1g/", 0],
     },
    {"label": "ReLU(L)+VirtualVF5",
     "control": "data/output/test_v13/gridhard/nonlinear_vf/original_0909/transfer/goal_id_{}/dqn_large_aux/aux_control/sweep_5g/",
     "online_measure": ["data/output/test_v13/gridhard/nonlinear_vf/original_0909/online_property/dqn_large_aux/aux_control/sweep_5g/", 2],
     "fixrep_measure": ["data/output/test_v13/gridhard/nonlinear_vf/original_0909/fixrep_property/dqn_large_aux/aux_control/sweep_5g/", 0],
     },
    {"label": "ReLU(L)+XY",
     "control": "data/output/test_v13/gridhard/nonlinear_vf/original_0909/transfer/goal_id_{}/dqn_large_aux/info/sweep/",
     "online_measure": ["data/output/test_v13/gridhard/nonlinear_vf/original_0909/online_property/dqn_large_aux/info/sweep/", 2],
     "fixrep_measure": ["data/output/test_v13/gridhard/nonlinear_vf/original_0909/fixrep_property/dqn_large_aux/info/sweep/", 0],
     },
    {"label": "ReLU(L)+Decoder",
     "control": "data/output/test_v13/gridhard/nonlinear_vf/original_0909/transfer/goal_id_{}/dqn_large_aux/input_decoder/sweep/",
     "online_measure": ["data/output/test_v13/gridhard/nonlinear_vf/original_0909/online_property/dqn_large_aux/input_decoder/sweep/", 3],
     "fixrep_measure": ["data/output/test_v13/gridhard/nonlinear_vf/original_0909/fixrep_property/dqn_large_aux/input_decoder/sweep/", 0],
     },
    {"label": "ReLU(L)+NAS",
     "control": "data/output/test_v13/gridhard/nonlinear_vf/original_0909/transfer/goal_id_{}/dqn_large_aux/nas_v2_delta/sweep/",
     "online_measure": ["data/output/test_v13/gridhard/nonlinear_vf/original_0909/online_property/dqn_large_aux/nas_v2_delta/sweep/", 4],
     "fixrep_measure": ["data/output/test_v13/gridhard/nonlinear_vf/original_0909/fixrep_property/dqn_large_aux/nas_v2_delta/sweep/", 0],
     },
    {"label": "ReLU(L)+Reward",
     "control": "data/output/test_v13/gridhard/nonlinear_vf/original_0909/transfer/goal_id_{}/dqn_large_aux/reward/sweep/",
     "online_measure": ["data/output/test_v13/gridhard/nonlinear_vf/original_0909/online_property/dqn_large_aux/reward/sweep/", 3],
     "fixrep_measure": ["data/output/test_v13/gridhard/nonlinear_vf/original_0909/fixrep_property/dqn_large_aux/reward/sweep/", 0],
     },
    {"label": "ReLU(L)+SF",
     "control": "data/output/test_v13/gridhard/nonlinear_vf/original_0909/transfer/goal_id_{}/dqn_large_aux/successor_as/sweep/",
     "online_measure": ["data/output/test_v13/gridhard/nonlinear_vf/original_0909/online_property/dqn_large_aux/successor_as/sweep/", 3],
     "fixrep_measure": ["data/output/test_v13/gridhard/nonlinear_vf/original_0909/fixrep_property/dqn_large_aux/successor_as/sweep/", 0],
     },
    {"label": "FTA eta=0.2",
     "control": "data/output/test_v13/gridhard/nonlinear_vf/original_0909/transfer/goal_id_{}/dqn_lta/eta_study_0.2_sweep/",
     "online_measure": ["data/output/test_v13/gridhard/nonlinear_vf/original_0909/online_property/dqn_lta/eta_study_0.2_sweep/", 2],
     "fixrep_measure": ["data/output/test_v13/gridhard/nonlinear_vf/original_0909/fixrep_property/dqn_lta/eta_study_0.2_sweep/", 0],
     },
    {"label": "FTA eta=0.4",
     "control": "data/output/test_v13/gridhard/nonlinear_vf/original_0909/transfer/goal_id_{}/dqn_lta/eta_study_0.4_sweep/",
     "online_measure": ["data/output/test_v13/gridhard/nonlinear_vf/original_0909/online_property/dqn_lta/eta_study_0.4_sweep/", 1],
     "fixrep_measure": ["data/output/test_v13/gridhard/nonlinear_vf/original_0909/fixrep_property/dqn_lta/eta_study_0.4_sweep/", 0],
     },
    {"label": "FTA eta=0.6",
     "control": "data/output/test_v13/gridhard/nonlinear_vf/original_0909/transfer/goal_id_{}/dqn_lta/eta_study_0.6_sweep/",
     "online_measure": ["data/output/test_v13/gridhard/nonlinear_vf/original_0909/online_property/dqn_lta/eta_study_0.6_sweep/", 2],
     "fixrep_measure": ["data/output/test_v13/gridhard/nonlinear_vf/original_0909/fixrep_property/dqn_lta/eta_study_0.6_sweep/", 0],
     },
    {"label": "FTA eta=0.8",
     "control": "data/output/test_v13/gridhard/nonlinear_vf/original_0909/transfer/goal_id_{}/dqn_lta/eta_study_0.8_sweep/",
     "online_measure": ["data/output/test_v13/gridhard/nonlinear_vf/original_0909/online_property/dqn_lta/eta_study_0.8_sweep/", 2],
     "fixrep_measure": ["data/output/test_v13/gridhard/nonlinear_vf/original_0909/fixrep_property/dqn_lta/eta_study_0.8_sweep/", 0],
     },
    {"label": "FTA+VirtualVF1",
     "control": "data/output/test_v13/gridhard/nonlinear_vf/original_0909/transfer/goal_id_{}/dqn_lta_aux/aux_control/sweep_1g/",
     "online_measure": ["data/output/test_v13/gridhard/nonlinear_vf/original_0909/online_property/dqn_lta_aux/aux_control/sweep_1g/", 1],
     "fixrep_measure": ["data/output/test_v13/gridhard/nonlinear_vf/original_0909/fixrep_property/dqn_lta_aux/aux_control/sweep_1g/", 0],
     },
    {"label": "FTA+VirtualVF5",
     "control": "data/output/test_v13/gridhard/nonlinear_vf/original_0909/transfer/goal_id_{}/dqn_lta_aux/aux_control/sweep_5g/",
     "online_measure": ["data/output/test_v13/gridhard/nonlinear_vf/original_0909/online_property/dqn_lta_aux/aux_control/sweep_5g/", 1],
     "fixrep_measure": ["data/output/test_v13/gridhard/nonlinear_vf/original_0909/fixrep_property/dqn_lta_aux/aux_control/sweep_5g/", 0],
     },
    {"label": "FTA+XY",
     "control": "data/output/test_v13/gridhard/nonlinear_vf/original_0909/transfer/goal_id_{}/dqn_lta_aux/info/sweep/",
     "online_measure": ["data/output/test_v13/gridhard/nonlinear_vf/original_0909/online_property/dqn_lta_aux/info/sweep/", 1],
     "fixrep_measure": ["data/output/test_v13/gridhard/nonlinear_vf/original_0909/fixrep_property/dqn_lta_aux/info/sweep/", 0],
     },
    {"label": "FTA+Decoder",
     "control": "data/output/test_v13/gridhard/nonlinear_vf/original_0909/transfer/goal_id_{}/dqn_lta_aux/input_decoder/sweep/",
     "online_measure": ["data/output/test_v13/gridhard/nonlinear_vf/original_0909/online_property/dqn_lta_aux/input_decoder/sweep/", 1],
     "fixrep_measure": ["data/output/test_v13/gridhard/nonlinear_vf/original_0909/fixrep_property/dqn_lta_aux/input_decoder/sweep/", 0],
     },
    {"label": "FTA+NAS",
     "control": "data/output/test_v13/gridhard/nonlinear_vf/original_0909/transfer/goal_id_{}/dqn_lta_aux/nas_v2_delta/sweep/",
     "online_measure": ["data/output/test_v13/gridhard/nonlinear_vf/original_0909/online_property/dqn_lta_aux/nas_v2_delta/sweep/", 2],
     "fixrep_measure": ["data/output/test_v13/gridhard/nonlinear_vf/original_0909/fixrep_property/dqn_lta_aux/nas_v2_delta/sweep/", 0],
     },
    {"label": "FTA+Reward",
     "control": "data/output/test_v13/gridhard/nonlinear_vf/original_0909/transfer/goal_id_{}/dqn_lta_aux/reward/sweep/",
     "online_measure": ["data/output/test_v13/gridhard/nonlinear_vf/original_0909/online_property/dqn_lta_aux/reward/sweep/", 2],
     "fixrep_measure": ["data/output/test_v13/gridhard/nonlinear_vf/original_0909/fixrep_property/dqn_lta_aux/reward/sweep/", 0],
     },
    {"label": "FTA+SF",
     "control": "data/output/test_v13/gridhard/nonlinear_vf/original_0909/transfer/goal_id_{}/dqn_lta_aux/successor_as/sweep/",
     "online_measure": ["data/output/test_v13/gridhard/nonlinear_vf/original_0909/online_property/dqn_lta_aux/successor_as/sweep/", 2],
     "fixrep_measure": ["data/output/test_v13/gridhard/nonlinear_vf/original_0909/fixrep_property/dqn_lta_aux/successor_as/sweep/", 0],
     },

    {"label": "ReLU+ATC",
    #  "control": "data/output/test_cl/gridhard/nonlinear_vf/transfer_new/goal_id_{}/dqn_cl/sweep/",
    #  "online_measure": ["data/output/test_cl/gridhard/nonlinear_vf/online_property/dqn_cl/best/", 0],
    #  },
    # {"label": "ReLU(L)+ATC",
    #  "control": "data/output/test_cl/gridhard/nonlinear_vf/transfer_new/goal_id_{}/dqn_cl/sweep_large/",
    #  "online_measure": ["data/output/test_cl/gridhard/nonlinear_vf/online_property/dqn_cl/best_large/", 0],
    #  },
    # {"label": "FTA+ATC",
    #  "control": "data/output/test_cl/gridhard/nonlinear_vf/transfer_new/goal_id_{}/dqn_fta_cl/sweep/",
    #  "online_measure": ["data/output/test_cl/gridhard/nonlinear_vf/online_property/dqn_fta_cl/best/", 0],
     "control": "data/output/test_v13/gridhard/nonlinear_vf/original_0909/transfer/goal_id_{}/dqn_cl/sweep/",
     "online_measure": ["data/output/test_v13/gridhard/nonlinear_vf/original_0909/online_property/dqn_cl/best/", 0],
     "fixrep_measure": ["data/output/test_v13/gridhard/nonlinear_vf/original_0909/fixrep_property/dqn_cl/best/", 0],
     },
    {"label": "ReLU(L)+ATC",
     "control": "data/output/test_v13/gridhard/nonlinear_vf/original_0909/transfer/goal_id_{}/dqn_cl/sweep_large/",
     "online_measure": ["data/output/test_v13/gridhard/nonlinear_vf/original_0909/online_property/dqn_large_cl/best/", 0],
     "fixrep_measure": ["data/output/test_v13/gridhard/nonlinear_vf/original_0909/fixrep_property/dqn_large_cl/best/", 0],
     },
    {"label": "FTA+ATC",
     "control": "data/output/test_v13/gridhard/nonlinear_vf/original_0909/transfer/goal_id_{}/dqn_fta_cl/sweep/",
     "online_measure": ["data/output/test_v13/gridhard/nonlinear_vf/original_0909/online_property/dqn_fta_cl/best/", 0],
     "fixrep_measure": ["data/output/test_v13/gridhard/nonlinear_vf/original_0909/fixrep_property/dqn_lta_cl/best/", 0],
     },

    {"label": "Scratch",
     "control": "data/output/test_v13/gridhard/nonlinear_vf/learning_scratch/goal_id_{}/dqn/sweep/",
     },
    {"label": "Random",
     "control": "data/output/test_v13/gridhard/nonlinear_vf/learning_scratch/goal_id_{}/random/sweep/",
     "online_measure": ["data/output/test_v13/gridhard/nonlinear_vf/fixrep_property/random/best/", 0],
     },
    {"label": "Input",
     "control": "data/output/test_v13/gridhard/nonlinear_vf/learning_scratch/goal_id_{}/input/sweep/",
     "online_measure": ["data/output/test_v13/gridhard/nonlinear_vf/fixrep_property/input/best/", 0],
     },
    {"label": "Scratch(FTA)",
     "control": "data/output/test_v13/gridhard/nonlinear_vf/learning_scratch/goal_id_{}/dqn_lta/eta_study_0.2_sweep/",
     },
    {"label": "Scratch(L)",
     "control": "data/output/test_v13/gridhard/nonlinear_vf/learning_scratch/goal_id_{}/dqn_large/sweep/",
     },
    {"label": "Random(L)",
     "control": "data/output/test_v13/gridhard/nonlinear_vf/learning_scratch/goal_id_{}/random_large/sweep/",
     "online_measure": ["data/output/test_v13/gridhard/nonlinear_vf/fixrep_property/random_large/best/", 0],
     },
]

gh_nonlinear_transfer_sweep_v13_largeReLU = [
    {"label": "ReLU",
     "control": "data/output/test_v13/gridhard/nonlinear_vf/original_0909/transfer/goal_id_{}/dqn/sweep/",
     "online_measure": ["data/output/test_v13/gridhard/nonlinear_vf/original_0909/online_property/dqn/best/", 0],
     "fixrep_measure": ["data/output/test_v13/gridhard/nonlinear_vf/original_0909/fixrep_property/dqn/sweep/", 0]
     },
    {"label": "ReLU+VirtualVF1",
     "control": "data/output/test_v13/gridhard/nonlinear_vf/original_0909/transfer/goal_id_{}/dqn_aux/aux_control/sweep_1g/",
     "online_measure": ["data/output/test_v13/gridhard/nonlinear_vf/original_0909/online_property/dqn_aux/aux_control/best_1g/", 0],
     "fixrep_measure": ["data/output/test_v13/gridhard/nonlinear_vf/original_0909/fixrep_property/dqn_aux/aux_control/sweep_1g/", 0],
     },
    {"label": "ReLU+VirtualVF5",
     "control": "data/output/test_v13/gridhard/nonlinear_vf/original_0909/transfer/goal_id_{}/dqn_aux/aux_control/sweep_5g/",
     "online_measure": ["data/output/test_v13/gridhard/nonlinear_vf/original_0909/online_property/dqn_aux/aux_control/best_5g/", 0],
     "fixrep_measure": ["data/output/test_v13/gridhard/nonlinear_vf/original_0909/fixrep_property/dqn_aux/aux_control/sweep_5g/", 0],
     },
    {"label": "ReLU+XY",
     "control": "data/output/test_v13/gridhard/nonlinear_vf/original_0909/transfer/goal_id_{}/dqn_aux/info/sweep/",
     "online_measure": ["data/output/test_v13/gridhard/nonlinear_vf/original_0909/online_property/dqn_aux/info/best/", 0],
     "fixrep_measure": ["data/output/test_v13/gridhard/nonlinear_vf/original_0909/fixrep_property/dqn_aux/info/sweep/", 0],
     },
    {"label": "ReLU+Decoder",
     "control": "data/output/test_v13/gridhard/nonlinear_vf/original_0909/transfer/goal_id_{}/dqn_aux/input_decoder/sweep/",
     "online_measure": ["data/output/test_v13/gridhard/nonlinear_vf/original_0909/online_property/dqn_aux/input_decoder/best/", 0],
     "fixrep_measure": ["data/output/test_v13/gridhard/nonlinear_vf/original_0909/fixrep_property/dqn_aux/input_decoder/sweep/", 0],
     },
    {"label": "ReLU+NAS",
     "control": "data/output/test_v13/gridhard/nonlinear_vf/original_0909/transfer/goal_id_{}/dqn_aux/nas_v2_delta/sweep/",
     "online_measure": ["data/output/test_v13/gridhard/nonlinear_vf/original_0909/online_property/dqn_aux/nas_v2_delta/best/", 0],
     "fixrep_measure": ["data/output/test_v13/gridhard/nonlinear_vf/original_0909/fixrep_property/dqn_aux/nas_v2_delta/sweep/", 0],
     },
    {"label": "ReLU+Reward",
     "control": "data/output/test_v13/gridhard/nonlinear_vf/original_0909/transfer/goal_id_{}/dqn_aux/reward/sweep/",
     "online_measure": ["data/output/test_v13/gridhard/nonlinear_vf/original_0909/online_property/dqn_aux/reward/best/", 0],
     "fixrep_measure": ["data/output/test_v13/gridhard/nonlinear_vf/original_0909/fixrep_property/dqn_aux/reward/sweep/", 0],
     },
    {"label": "ReLU+SF",
     "control": "data/output/test_v13/gridhard/nonlinear_vf/original_0909/transfer/goal_id_{}/dqn_aux/successor_as/sweep/",
     "online_measure": ["data/output/test_v13/gridhard/nonlinear_vf/original_0909/online_property/dqn_aux/successor_as/best/", 0],
     "fixrep_measure": ["data/output/test_v13/gridhard/nonlinear_vf/original_0909/fixrep_property/dqn_aux/successor_as/sweep/", 0],
     },
    {"label": "ReLU(L)",
     "control": "data/output/test_v13/gridhard/nonlinear_vf/original_0909/transfer/goal_id_{}/dqn_large/sweep/",
     "online_measure": ["data/output/test_v13/gridhard/nonlinear_vf/original_0909/online_property/dqn_large/best/", 0],
     "fixrep_measure": ["data/output/test_v13/gridhard/nonlinear_vf/original_0909/fixrep_property/dqn_large/sweep/", 0]
     },
    {"label": "ReLU(L)+VirtualVF1",
     "control": "data/output/test_v13/gridhard/nonlinear_vf/original_0909/transfer/goal_id_{}/dqn_large_aux/aux_control/sweep_1g/",
     "online_measure": ["data/output/test_v13/gridhard/nonlinear_vf/original_0909/online_property/dqn_large_aux/aux_control/best_1g/", 0],
     "fixrep_measure": ["data/output/test_v13/gridhard/nonlinear_vf/original_0909/fixrep_property/dqn_large_aux/aux_control/sweep_1g/", 0],
     },
    {"label": "ReLU(L)+VirtualVF5",
     "control": "data/output/test_v13/gridhard/nonlinear_vf/original_0909/transfer/goal_id_{}/dqn_large_aux/aux_control/sweep_5g/",
     "online_measure": ["data/output/test_v13/gridhard/nonlinear_vf/original_0909/online_property/dqn_large_aux/aux_control/best_5g/", 0],
     "fixrep_measure": ["data/output/test_v13/gridhard/nonlinear_vf/original_0909/fixrep_property/dqn_large_aux/aux_control/sweep_5g/", 0],
     },
    {"label": "ReLU(L)+XY",
     "control": "data/output/test_v13/gridhard/nonlinear_vf/original_0909/transfer/goal_id_{}/dqn_large_aux/info/sweep/",
     "online_measure": ["data/output/test_v13/gridhard/nonlinear_vf/original_0909/online_property/dqn_large_aux/info/best/", 0],
     "fixrep_measure": ["data/output/test_v13/gridhard/nonlinear_vf/original_0909/fixrep_property/dqn_large_aux/info/sweep/", 0],
     },
    {"label": "ReLU(L)+Decoder",
     "control": "data/output/test_v13/gridhard/nonlinear_vf/original_0909/transfer/goal_id_{}/dqn_large_aux/input_decoder/sweep/",
     "online_measure": ["data/output/test_v13/gridhard/nonlinear_vf/original_0909/online_property/dqn_large_aux/input_decoder/best/", 0],
     "fixrep_measure": ["data/output/test_v13/gridhard/nonlinear_vf/original_0909/fixrep_property/dqn_large_aux/input_decoder/sweep/", 0],
     },
    {"label": "ReLU(L)+NAS",
     "control": "data/output/test_v13/gridhard/nonlinear_vf/original_0909/transfer/goal_id_{}/dqn_large_aux/nas_v2_delta/sweep/",
     "online_measure": ["data/output/test_v13/gridhard/nonlinear_vf/original_0909/online_property/dqn_large_aux/nas_v2_delta/best/", 0],
     "fixrep_measure": ["data/output/test_v13/gridhard/nonlinear_vf/original_0909/fixrep_property/dqn_large_aux/nas_v2_delta/sweep/", 0],
     },
    {"label": "ReLU(L)+Reward",
     "control": "data/output/test_v13/gridhard/nonlinear_vf/original_0909/transfer/goal_id_{}/dqn_large_aux/reward/sweep/",
     "online_measure": ["data/output/test_v13/gridhard/nonlinear_vf/original_0909/online_property/dqn_large_aux/reward/best/", 0],
     "fixrep_measure": ["data/output/test_v13/gridhard/nonlinear_vf/original_0909/fixrep_property/dqn_large_aux/reward/sweep/", 0],
     },
    {"label": "ReLU(L)+SF",
     "control": "data/output/test_v13/gridhard/nonlinear_vf/original_0909/transfer/goal_id_{}/dqn_large_aux/successor_as/sweep/",
     "online_measure": ["data/output/test_v13/gridhard/nonlinear_vf/original_0909/online_property/dqn_large_aux/successor_as/best/", 0],
     "fixrep_measure": ["data/output/test_v13/gridhard/nonlinear_vf/original_0909/fixrep_property/dqn_large_aux/successor_as/sweep/", 0],
     },
    {"label": "FTA eta=0.2",
     "control": "data/output/test_v13/gridhard/nonlinear_vf/original_0909/transfer/goal_id_{}/dqn_lta/eta_study_0.2_sweep/",
     "online_measure": ["data/output/test_v13/gridhard/nonlinear_vf/original_0909/online_property/dqn_lta/eta_study_0.2_best/", 0],
     "fixrep_measure": ["data/output/test_v13/gridhard/nonlinear_vf/original_0909/fixrep_property/dqn_lta/eta_study_0.2_sweep/", 0],
     },
    {"label": "FTA eta=0.4",
     "control": "data/output/test_v13/gridhard/nonlinear_vf/original_0909/transfer/goal_id_{}/dqn_lta/eta_study_0.4_sweep/",
     "online_measure": ["data/output/test_v13/gridhard/nonlinear_vf/original_0909/online_property/dqn_lta/eta_study_0.4_best/", 0],
     "fixrep_measure": ["data/output/test_v13/gridhard/nonlinear_vf/original_0909/fixrep_property/dqn_lta/eta_study_0.4_sweep/", 0],
     },
    {"label": "FTA eta=0.6",
     "control": "data/output/test_v13/gridhard/nonlinear_vf/original_0909/transfer/goal_id_{}/dqn_lta/eta_study_0.6_sweep/",
     "online_measure": ["data/output/test_v13/gridhard/nonlinear_vf/original_0909/online_property/dqn_lta/eta_study_0.6_best/", 0],
     "fixrep_measure": ["data/output/test_v13/gridhard/nonlinear_vf/original_0909/fixrep_property/dqn_lta/eta_study_0.6_sweep/", 0],
     },
    {"label": "FTA eta=0.8",
     "control": "data/output/test_v13/gridhard/nonlinear_vf/original_0909/transfer/goal_id_{}/dqn_lta/eta_study_0.8_sweep/",
     "online_measure": ["data/output/test_v13/gridhard/nonlinear_vf/original_0909/online_property/dqn_lta/eta_study_0.8_best/", 0],
     "fixrep_measure": ["data/output/test_v13/gridhard/nonlinear_vf/original_0909/fixrep_property/dqn_lta/eta_study_0.8_sweep/", 0],
     },
    {"label": "FTA+VirtualVF1",
     "control": "data/output/test_v13/gridhard/nonlinear_vf/original_0909/transfer/goal_id_{}/dqn_lta_aux/aux_control/sweep_1g/",
     "online_measure": ["data/output/test_v13/gridhard/nonlinear_vf/original_0909/online_property/dqn_lta_aux/aux_control/best_1g/", 0],
     "fixrep_measure": ["data/output/test_v13/gridhard/nonlinear_vf/original_0909/fixrep_property/dqn_lta_aux/aux_control/sweep_1g/", 0],
     },
    {"label": "FTA+VirtualVF5",
     "control": "data/output/test_v13/gridhard/nonlinear_vf/original_0909/transfer/goal_id_{}/dqn_lta_aux/aux_control/sweep_5g/",
     "online_measure": ["data/output/test_v13/gridhard/nonlinear_vf/original_0909/online_property/dqn_lta_aux/aux_control/best_5g/", 0],
     "fixrep_measure": ["data/output/test_v13/gridhard/nonlinear_vf/original_0909/fixrep_property/dqn_lta_aux/aux_control/sweep_5g/", 0],
     },
    {"label": "FTA+XY",
     "control": "data/output/test_v13/gridhard/nonlinear_vf/original_0909/transfer/goal_id_{}/dqn_lta_aux/info/sweep/",
     "online_measure": ["data/output/test_v13/gridhard/nonlinear_vf/original_0909/online_property/dqn_lta_aux/info/best/", 0],
     "fixrep_measure": ["data/output/test_v13/gridhard/nonlinear_vf/original_0909/fixrep_property/dqn_lta_aux/info/sweep/", 0],
     },
    {"label": "FTA+Decoder",
     "control": "data/output/test_v13/gridhard/nonlinear_vf/original_0909/transfer/goal_id_{}/dqn_lta_aux/input_decoder/sweep/",
     "online_measure": ["data/output/test_v13/gridhard/nonlinear_vf/original_0909/online_property/dqn_lta_aux/input_decoder/best/", 0],
     "fixrep_measure": ["data/output/test_v13/gridhard/nonlinear_vf/original_0909/fixrep_property/dqn_lta_aux/input_decoder/sweep/", 0],
     },
    {"label": "FTA+NAS",
     "control": "data/output/test_v13/gridhard/nonlinear_vf/original_0909/transfer/goal_id_{}/dqn_lta_aux/nas_v2_delta/sweep/",
     "online_measure": ["data/output/test_v13/gridhard/nonlinear_vf/original_0909/online_property/dqn_lta_aux/nas_v2_delta/best/", 0],
     "fixrep_measure": ["data/output/test_v13/gridhard/nonlinear_vf/original_0909/fixrep_property/dqn_lta_aux/nas_v2_delta/sweep/", 0],
     },
    {"label": "FTA+Reward",
     "control": "data/output/test_v13/gridhard/nonlinear_vf/original_0909/transfer/goal_id_{}/dqn_lta_aux/reward/sweep/",
     "online_measure": ["data/output/test_v13/gridhard/nonlinear_vf/original_0909/online_property/dqn_lta_aux/reward/best/", 0],
     "fixrep_measure": ["data/output/test_v13/gridhard/nonlinear_vf/original_0909/fixrep_property/dqn_lta_aux/reward/sweep/", 0],
     },
    {"label": "FTA+SF",
     "control": "data/output/test_v13/gridhard/nonlinear_vf/original_0909/transfer/goal_id_{}/dqn_lta_aux/successor_as/sweep/",
     "online_measure": ["data/output/test_v13/gridhard/nonlinear_vf/original_0909/online_property/dqn_lta_aux/successor_as/best/", 0],
     "fixrep_measure": ["data/output/test_v13/gridhard/nonlinear_vf/original_0909/fixrep_property/dqn_lta_aux/successor_as/sweep/", 0],
     },

    {"label": "ReLU+ATC",
    #  "control": "data/output/test_cl/gridhard/nonlinear_vf/transfer_new/goal_id_{}/dqn_cl/sweep/",
    #  "online_measure": ["data/output/test_cl/gridhard/nonlinear_vf/online_property/dqn_cl/best/", 0],
    #  },
    # {"label": "ReLU(L)+ATC",
    #  "control": "data/output/test_cl/gridhard/nonlinear_vf/transfer_new/goal_id_{}/dqn_cl/sweep_large/",
    #  "online_measure": ["data/output/test_cl/gridhard/nonlinear_vf/online_property/dqn_cl/best_large/", 0],
    #  },
    # {"label": "FTA+ATC",
    #  "control": "data/output/test_cl/gridhard/nonlinear_vf/transfer_new/goal_id_{}/dqn_fta_cl/sweep/",
    #  "online_measure": ["data/output/test_cl/gridhard/nonlinear_vf/online_property/dqn_fta_cl/best/", 0],
     "control": "data/output/test_v13/gridhard/nonlinear_vf/original_0909/transfer/goal_id_{}/dqn_cl/sweep/",
     "online_measure": ["data/output/test_v13/gridhard/nonlinear_vf/original_0909/online_property/dqn_cl/best/", 0],
     "fixrep_measure": ["data/output/test_v13/gridhard/nonlinear_vf/original_0909/fixrep_property/dqn_cl/best/", 0],
     },
    {"label": "ReLU(L)+ATC",
     "control": "data/output/test_v13/gridhard/nonlinear_vf/original_0909/transfer/goal_id_{}/dqn_cl/sweep_large/",
     "online_measure": ["data/output/test_v13/gridhard/nonlinear_vf/original_0909/online_property/dqn_large_cl/best/", 0],
     "fixrep_measure": ["data/output/test_v13/gridhard/nonlinear_vf/original_0909/fixrep_property/dqn_large_cl/best/", 0],
     },
    {"label": "FTA+ATC",
     "control": "data/output/test_v13/gridhard/nonlinear_vf/original_0909/transfer/goal_id_{}/dqn_fta_cl/sweep/",
     "online_measure": ["data/output/test_v13/gridhard/nonlinear_vf/original_0909/online_property/dqn_fta_cl/best/", 0],
     "fixrep_measure": ["data/output/test_v13/gridhard/nonlinear_vf/original_0909/fixrep_property/dqn_lta_cl/best/", 0],
     },

    {"label": "Scratch",
     "control": "data/output/test_v13/gridhard/nonlinear_vf/learning_scratch/goal_id_{}/dqn/sweep/",
     },
    {"label": "Random",
     "control": "data/output/test_v13/gridhard/nonlinear_vf/learning_scratch/goal_id_{}/random/sweep/",
     "online_measure": ["data/output/test_v13/gridhard/nonlinear_vf/fixrep_property/random/best/", 0],
     },
    {"label": "Input",
     "control": "data/output/test_v13/gridhard/nonlinear_vf/learning_scratch/goal_id_{}/input/sweep/",
     "online_measure": ["data/output/test_v13/gridhard/nonlinear_vf/fixrep_property/input/best/", 0],
     },
    {"label": "Scratch(FTA)",
     "control": "data/output/test_v13/gridhard/nonlinear_vf/learning_scratch/goal_id_{}/dqn_lta/eta_study_0.2_sweep/",
     },
    {"label": "Scratch(L)",
     "control": "data/output/test_v13/gridhard/nonlinear_vf/learning_scratch/goal_id_{}/dqn_large/sweep/",
     },
    {"label": "Random(L)",
     "control": "data/output/test_v13/gridhard/nonlinear_vf/learning_scratch/goal_id_{}/random_large/sweep/",
     "online_measure": ["data/output/test_v13/gridhard/nonlinear_vf/fixrep_property/random_large/best/", 0],
     },
]

ghmg_original_sweep_v13 = [
    {"label": "ReLU close10",
     "control": "data/output/test_v13/gridhard_multigoal/linear_vf/close_10perc/online_property/dqn/sweep/",
     },
    {"label": "ReLU close25",
     "control": "data/output/test_v13/gridhard_multigoal/linear_vf/close_25perc/online_property/dqn/sweep/",
     },
    {"label": "ReLU close50",
     "control": "data/output/test_v13/gridhard_multigoal/linear_vf/close_50perc/online_property/dqn/sweep/",
     },
    {"label": "ReLU close75",
     "control": "data/output/test_v13/gridhard_multigoal/linear_vf/close_75perc/online_property/dqn/sweep/",
     },
]

ghmg_transfer_sweep_v13 = [
    {"label": "ReLU",
     "control": "data/output/test_v13/gridhard/linear_vf/original_0909/transfer/goal_id_{}/dqn/sweep/",
     "online_measure": ["data/output/test_v13/gridhard/linear_vf/original_0909/online_property/dqn/sweep/", 1]
     },
    {"label": "ReLU close10",
     "control": "data/output/test_v13/gridhard_multigoal/linear_vf/close_10perc/transfer/goal_id_{}/dqn/sweep/",
     "online_measure": ["data/output/test_v13/gridhard_multigoal/linear_vf/close_10perc/online_property/dqn/sweep/", 1]
     },
    {"label": "ReLU close25",
     "control": "data/output/test_v13/gridhard_multigoal/linear_vf/close_25perc/transfer/goal_id_{}/dqn/sweep/",
     "online_measure": ["data/output/test_v13/gridhard_multigoal/linear_vf/close_25perc/online_property/dqn/sweep/", 1]
     },
    {"label": "ReLU close50",
     "control": "data/output/test_v13/gridhard_multigoal/linear_vf/close_50perc/transfer/goal_id_{}/dqn/sweep/",
     "online_measure": ["data/output/test_v13/gridhard_multigoal/linear_vf/close_50perc/online_property/dqn/sweep/", 2]
     },

    {"label": "Scratch",
     # "control": "data/output/test_v13/gridhard/linear_vf/learning_scratch/goal_id_{}/dqn/fix_eps_sweep/",
     "control": "data/output/test_v13/gridhard/linear_vf/learning_scratch/goal_id_{}/dqn/sweep/",
     },
    {"label": "Random",
     "control": "data/output/test_v13/gridhard/linear_vf/learning_scratch/goal_id_{}/random/sweep/",
     },
    {"label": "Input",
     "control": "data/output/test_v13/gridhard/linear_vf/learning_scratch/goal_id_{}/input/sweep/",
     },
]

ghmg_transfer_last_sweep_v13 = [
    {"label": "ReLU",
     "control": "data/output/test_v13/gridhard/linear_vf/original_0909/transfer/goal_id_{}/dqn/sweep/",
     "online_measure": ["data/output/test_v13/gridhard/linear_vf/original_0909/online_property/dqn/sweep/", 1]
     },
    {"label": "ReLU close10",
     "control": "data/output/test_v13/gridhard_multigoal/linear_vf/close_10perc/transfer/goal_id_{}/dqn/last_sweep/",
     "online_measure": ["data/output/test_v13/gridhard_multigoal/linear_vf/close_10perc/online_property/dqn/sweep/", 1]
     },
    {"label": "ReLU close25",
     "control": "data/output/test_v13/gridhard_multigoal/linear_vf/close_25perc/transfer/goal_id_{}/dqn/last_sweep/",
     "online_measure": ["data/output/test_v13/gridhard_multigoal/linear_vf/close_25perc/online_property/dqn/sweep/", 1]
     },
    {"label": "ReLU close50",
     "control": "data/output/test_v13/gridhard_multigoal/linear_vf/close_50perc/transfer/goal_id_{}/dqn/last_sweep/",
     "online_measure": ["data/output/test_v13/gridhard_multigoal/linear_vf/close_50perc/online_property/dqn/sweep/", 2]
     },

    {"label": "Scratch",
     # "control": "data/output/test_v13/gridhard/linear_vf/learning_scratch/goal_id_{}/dqn/fix_eps_sweep/",
     "control": "data/output/test_v13/gridhard/linear_vf/learning_scratch/goal_id_{}/dqn/sweep/",
     },
    {"label": "Random",
     "control": "data/output/test_v13/gridhard/linear_vf/learning_scratch/goal_id_{}/random/sweep/",
     },
    {"label": "Input",
     "control": "data/output/test_v13/gridhard/linear_vf/learning_scratch/goal_id_{}/input/sweep/",
     },
]
gh_nonlinear_fta_original_sweep_v13 = [
    {"label": "FTA eta=0.8",
     "control": "data/output/test_v13/gridhard/nonlinear_vf/original_0909/online_property/dqn_lta/eta_study_0.8_sweep/",
     "best": '2'
    },
    {"label": "FTA+Control1g",
     "control": "data/output/test_v13/gridhard/nonlinear_vf/original_0909/online_property/dqn_lta_aux/aux_control/sweep_1g/",
     "best": '1'
     },
    {"label": "FTA+Control5g",
     "control": "data/output/test_v13/gridhard/nonlinear_vf/original_0909/online_property/dqn_lta_aux/aux_control/sweep_5g/",
     "best": '1'
     },
    {"label": "FTA+XY",
     "control": "data/output/test_v13/gridhard/nonlinear_vf/original_0909/online_property/dqn_lta_aux/info/sweep/",
     "best": '1'
     },
    {"label": "FTA+Decoder",
     "control": "data/output/test_v13/gridhard/nonlinear_vf/original_0909/online_property/dqn_lta_aux/input_decoder/sweep/",
     "best": '1'
     },
    {"label": "FTA+NAS",
     "control": "data/output/test_v13/gridhard/nonlinear_vf/original_0909/online_property/dqn_lta_aux/nas_v2_delta/sweep/",
     "best": '2'
     },
    {"label": "FTA+Reward",
     "control": "data/output/test_v13/gridhard/nonlinear_vf/original_0909/online_property/dqn_lta_aux/reward/sweep/",
     "best": '2'
     },
    {"label": "FTA+SF",
     "control": "data/output/test_v13/gridhard/nonlinear_vf/original_0909/online_property/dqn_lta_aux/successor_as/sweep/",
     "best": '2'
     },
]

dqn_cl_maze_sweep = [
#    {"label": "ReLU",
#     "control": "data/output/test_cl/gridhard/nonlinear_vf/online_property/dqn_cl/best"
#     },
    # {"label": "FTA eta=0.8",
    #  "control": "data/output/test_cl/gridhard/nonlinear_vf/online_property/dqn_fta_cl/sweep",
    #  "best": "4"
    #  },
    {"label": "ReLU(L)+ATC",
     "control": "data/output/test_cl/gridhard/nonlinear_vf/online_property/dqn_cl/best_large",
     "best": "28"
     },
    ]

dqn_cl_maze_sweep_es_sweep = [

    {"label": "ReLU+ATC encoder-size=8",
     "control": "data/output/test_cl/gridhard/nonlinear_vf/online_property/dqn_cl/sweep_hs_8",
          "best": "27"

     },
    {"label": "ReLU+ATC encoder-size=16",
     "control": "data/output/test_cl/gridhard/nonlinear_vf/online_property/dqn_cl/sweep_hs_16",
          "best": "1"
     },
    {"label": "ReLU+ATC encoder-size=32",
     "control": "data/output/test_cl/gridhard/nonlinear_vf/online_property/dqn_cl/sweep_hs_32",
          "best": "1"
},
    # {"label": "ReLU+ATC encoder-size=64",
    #  "control": "data/output/test_cl/gridhard/nonlinear_vf/online_property/dqn_cl/sweep",
    #  },
    {"label": "ReLU+ATC encoder-size=128",
     "control": "data/output/test_cl/gridhard/nonlinear_vf/online_property/dqn_cl/sweep_hs_128",
          "best": "22"
},     
]

dqn_cl_maze_best_es_sweep = [

    {"label": "ReLU+ATC encoder-size=8",
     "control": "data/output/test_cl/gridhard/nonlinear_vf/online_property/dqn_cl/best_hs_8",
     "best": "27"
 
     },
    {"label": "ReLU+ATC encoder-size=16",
     "control": "data/output/test_cl/gridhard/nonlinear_vf/online_property/dqn_cl/best_hs_16",
          "best": "1"

     },

    {"label": "ReLU+ATC encoder-size=32",
     "control": "data/output/test_cl/gridhard/nonlinear_vf/online_property/dqn_cl/best_hs_32",
     "best": "1"
     },
     

    {"label": "ReLU+ATC encoder-size=64",
     "control": "data/output/test_cl/gridhard/nonlinear_vf/online_property/dqn_cl/best",
     },
    {"label": "ReLU+ATC encoder-size=128",
     "control": "data/output/test_cl/gridhard/nonlinear_vf/online_property/dqn_cl/best_hs_128",
     },     
]

dqn_cl_maze_sweep_sp_sweep = [

    {"label": "ReLU+ATC shift-prob=0",
     "control": "data/output/test_cl/gridhard/nonlinear_vf/online_property/dqn_cl/sweep_sp_0",
     "best": "22"

     },
    {"label": "ReLU+ATC shift-prob=0.01",
     "control": "data/output/test_cl/gridhard/nonlinear_vf/online_property/dqn_cl/sweep_sp_0.01",
    "best": "22"

     },
    # {"label": "ReLU+ATC shift-prob=0.1",
    #  "control": "data/output/test_cl/gridhard/nonlinear_vf/online_property/dqn_cl/sweep",
    #  },
    {"label": "ReLU+ATC shift-prob=0.2",
     "control": "data/output/test_cl/gridhard/nonlinear_vf/online_property/dqn_cl/sweep_sp_0.2",
                    "best": "22"

     },
    {"label": "ReLU+ATC shift-prob=0.3",
     "control": "data/output/test_cl/gridhard/nonlinear_vf/online_property/dqn_cl/sweep_sp_0.3",
               "best": "22"

     },     
]


dqn_cl_maze_best_sp_sweep = [
    {"label": "ReLU+ATC shift-prob=0",
     "control": "data/output/test_cl/gridhard/nonlinear_vf/online_property/dqn_cl/best_sp_0",
     },
    {"label": "ReLU+ATC shift-prob=0.01",
     "control": "data/output/test_cl/gridhard/nonlinear_vf/online_property/dqn_cl/best_sp_0.01",
     },
    {"label": "ReLU+ATC shift-prob=0.1",
     "control": "data/output/test_cl/gridhard/nonlinear_vf/online_property/dqn_cl/best",
     },
    {"label": "ReLU+ATC shift-prob=0.2",
     "control": "data/output/test_cl/gridhard/nonlinear_vf/online_property/dqn_cl/best_sp_0.2",
     },
    {"label": "ReLU+ATC shift-prob=0.3",
     "control": "data/output/test_cl/gridhard/nonlinear_vf/online_property/dqn_cl/best_sp_0.3",
     },     
]



dqn_cl_maze_best = [
#    {"label": "ReLU",
#     "control": "data/output/test_cl/gridhard/nonlinear_vf/online_property/dqn_cl/best"
#     },
    # {"label": "FTA eta=0.8",
    #  "control": "data/output/test_cl/gridhard/nonlinear_vf/online_property/dqn_fta_cl/sweep",
    #  "best": "4"
    #  },
    {"label": "FTA eta=0.8",
     "control": "data/output/test_cl/gridhard/nonlinear_vf/online_property/dqn_cl/best_delta_2",
     "best": "2"
     },
    {"label": "FTA eta=0.6",
     "control": "data/output/test_cl/gridhard/nonlinear_vf/online_property/dqn_cl/best_delta_4",
     "best": "22"
     },
    {"label": "FTA eta=0.4",
     "control": "data/output/test_cl/gridhard/nonlinear_vf/online_property/dqn_cl/best_delta_5",
     "best": "27"
     },
    ]


dqn_cl_linear_maze_sweep = [
    {"label": "ReLU",
     "control": "data/output/test_cl/gridhard/linear_vf/online_property/dqn_cl/sweep",
     "best": "7"
     },
    {"label": "FTA eta=0.8",
     "control": "data/output/test_cl/gridhard/linear_vf/online_property/dqn_fta_cl/sweep",
     "best": "10"
     },

    ]

dqn_cl_linear_maze_best = [
    {"label": "ReLU",
     "control": "data/output/test_cl/gridhard/linear_vf/online_property/dqn_cl/best",
     "best": "7"
     },
    {"label": "FTA eta=0.8",
     "control": "data/output/test_cl/gridhard/linear_vf/online_property/dqn_fta_cl/best",
     "best": "10"
     },

    ]

# dqn_cl_maze_best = [
# #    {"label": "ReLU",
# #     "control": "data/output/test_cl/gridhard/nonlinear_vf/online_property/dqn_cl/best"
# #     },
#     {"label": "FTA eta=0.8",
#      "control": "data/output/test_cl/gridhard/nonlinear_vf/online_property/dqn_fta_cl/best",
#      "best": "4"
#      },
#     ]
dqn_cl_extra_maze_sweep = [
    {"label": "ReLU",
     "control": "data/output/test_cl/gridhard/nonlinear_vf/online_property/dqn_cl_extra/sweep",
     },
    ]
dqn_cl_extra_maze_best = [
    {"label": "ReLU",
     "control": "data/output/test_cl/gridhard/nonlinear_vf/dqn_cl_extra/best",
     },
    ]


dqn_maze_sweep = [
    {"label": "ReLU",
     "control": "data/output/test_cl/gridhard/nonlinear_vf/online_property/dqn/sweep",
     },
    ]

dqn_cl_transfer_maze_sweep = [
    {"label": "ReLU_0",
     "control": "data/output/test_cl/gridhard/nonlinear_vf/transfer_new/goal_id_0/dqn_cl/sweep",
    },
    {"label": "ReLU_27",
     "control": "data/output/test_cl/gridhard/nonlinear_vf/transfer_new/goal_id_27/dqn_cl/sweep",
    },
    {"label": "ReLU_61",
     "control": "data/output/test_cl/gridhard/nonlinear_vf/transfer_new/goal_id_61/dqn_cl/sweep",
    },
    {"label": "ReLU_64",
     "control": "data/output/test_cl/gridhard/nonlinear_vf/transfer_new/goal_id_64/dqn_cl/sweep",
    },
    {"label": "ReLU_106",
     "control": "data/output/test_cl/gridhard/nonlinear_vf/transfer_new/goal_id_106/dqn_cl/sweep",
    },
    ]



dqn_transfer_maze_sweep = [
    {"label": "ReLU_0",
     "control": "data/output/test_cl/gridhard/nonlinear_vf/transfer_new/goal_id_0/dqn/sweep",
    },
    {"label": "ReLU_27",
     "control": "data/output/test_cl/gridhard/nonlinear_vf/transfer_new/goal_id_27/dqn/sweep",
    },
    {"label": "ReLU_61",
     "control": "data/output/test_cl/gridhard/nonlinear_vf/transfer_new/goal_id_61/dqn/sweep",
    },
    {"label": "ReLU_64",
     "control": "data/output/test_cl/gridhard/nonlinear_vf/transfer_new/goal_id_64/dqn/sweep",
    },
    {"label": "ReLU_106",
     "control": "data/output/test_cl/gridhard/nonlinear_vf/transfer_new/goal_id_106/dqn/sweep",
    },
    ]

nonlinear_maze_atc_transfer_sweep = [
    {"label": "ReLU+ATC",
     "control": "data/output/test_cl/gridhard/nonlinear_vf/transfer_new/goal_id_{}/dqn_cl/sweep",
#     "online_measure": ["data/output/test_v13/gridhard/nonlinear_vf/original_0909/online_property/dqn/sweep/", 3]
     },
    {"label": "FTA+ATC",
     "control": "data/output/test_cl/gridhard/nonlinear_vf/transfer_new/goal_id_{}/dqn_fta_cl/sweep",
#     "online_measure": ["data/output/test_v13/gridhard/nonlinear_vf/original_0909/online_property/dqn/sweep/", 3]
     },
]

linear_maze_atc_transfer_sweep = [
    {"label": "ReLU+ATC",
     "control": "data/output/test_cl/gridhard/linear_vf/transfer_new/goal_id_{}/dqn_cl/sweep",
#     "online_measure": ["data/output/test_v13/gridhard/nonlinear_vf/original_0909/online_property/dqn/sweep/", 3]
     },
    {"label": "FTA+ATC",
     "control": "data/output/test_cl/gridhard/linear_vf/transfer_new/goal_id_{}/dqn_fta_cl/sweep",
#     "online_measure": ["data/output/test_v13/gridhard/nonlinear_vf/original_0909/online_property/dqn/sweep/", 3]
     },
]

nonlinear_maze_atc_transfer_sweep_fa = [
    {"label": "ReLU+ATC",
     "control": "data/output/test_cl/gridhard/nonlinear_vf/transfer_new/goal_id_{}/dqn_cl/sweep",
     "online_measure": ["data/output/test_cl/gridhard/nonlinear_vf/online_property/dqn_cl/best", 0]
     },
     
    {"label": "FTA+ATC",
     "control": "data/output/test_cl/gridhard/nonlinear_vf/transfer_new/goal_id_{}/dqn_fta_cl/sweep",
     "online_measure": ["data/output/test_cl/gridhard/nonlinear_vf/online_property/dqn_fta_cl/best", 0]
     },
]


nonlinear_maze_atc_transfer_sweep_fa_delta_sweep = [

    {"label": "ReLU+ATC delta=2",
     "control": "data/output/test_cl/gridhard/nonlinear_vf/transfer_new/goal_id_{}/dqn_cl/sweep_delta_2",
     "online_measure": ["data/output/test_cl/gridhard/nonlinear_vf/online_property/dqn_cl/best_delta_2", 0]
     },
    {"label": "ReLU+ATC delta=3",
     "control": "data/output/test_cl/gridhard/nonlinear_vf/transfer_new/goal_id_{}/dqn_cl/sweep",
     "online_measure": ["data/output/test_cl/gridhard/nonlinear_vf/online_property/dqn_cl/best", 0]
     },
    {"label": "ReLU+ATC delta=4",
     "control": "data/output/test_cl/gridhard/nonlinear_vf/transfer_new/goal_id_{}/dqn_cl/sweep_delta_4",
     "online_measure": ["data/output/test_cl/gridhard/nonlinear_vf/online_property/dqn_cl/best_delta_4", 0]
     },
    {"label": "ReLU+ATC delta=5",
     "control": "data/output/test_cl/gridhard/nonlinear_vf/transfer_new/goal_id_{}/dqn_cl/sweep_delta_5",
     "online_measure": ["data/output/test_cl/gridhard/nonlinear_vf/online_property/dqn_cl/best_delta_5", 0]
     },
     
]


nonlinear_maze_atc_transfer_sweep_fa_encode_size_sweep = [

    # {"label": "ReLU+ATC encoder-size=8",
    #  "control": "data/output/test_cl/gridhard/nonlinear_vf/transfer_new/goal_id_{}/dqn_cl/sweep_hs_8",
    #  "online_measure": ["data/output/test_cl/gridhard/nonlinear_vf/online_property/dqn_cl/best_hs_8", 0]
    #  },
    # {"label": "ReLU+ATC encoder-size=16",
    #  "control": "data/output/test_cl/gridhard/nonlinear_vf/transfer_new/goal_id_{}/dqn_cl/sweep_hs_16",
    #  "online_measure": ["data/output/test_cl/gridhard/nonlinear_vf/online_property/dqn_cl/best_hs_16", 0]
    #  },
    {"label": "ReLU+ATC encoder-size=32",
     "control": "data/output/test_cl/gridhard/nonlinear_vf/transfer_new/goal_id_{}/dqn_cl/sweep_hs_32",
     "online_measure": ["data/output/test_cl/gridhard/nonlinear_vf/online_property/dqn_cl/best_hs_32", 0]
     },
    {"label": "ReLU+ATC encoder-size=64",
     "control": "data/output/test_cl/gridhard/nonlinear_vf/transfer_new/goal_id_{}/dqn_cl/sweep",
     "online_measure": ["data/output/test_cl/gridhard/nonlinear_vf/online_property/dqn_cl/best", 0]
     },
    # {"label": "ReLU+ATC encoder-size=128",
    #  "control": "data/output/test_cl/gridhard/nonlinear_vf/transfer_new/goal_id_{}/dqn_cl/sweep_hs_128",
    #  "online_measure": ["data/output/test_cl/gridhard/nonlinear_vf/online_property/dqn_cl/best_hs_128", 0]
    #  },     
]


nonlinear_maze_atc_transfer_sweep_fa_shift_prob_sweep = [
    {"label": "ReLU+ATC shift-prob=0",
     "control": "data/output/test_cl/gridhard/nonlinear_vf/transfer_new/goal_id_{}/dqn_cl/sweep_sp_0",
     "online_measure": ["data/output/test_cl/gridhard/nonlinear_vf/online_property/dqn_cl/best_sp_0", 0]
     },
    {"label": "ReLU+ATC shift-prob=0.01",
     "control": "data/output/test_cl/gridhard/nonlinear_vf/transfer_new/goal_id_{}/dqn_cl/sweep_sp_0.01",
     "online_measure": ["data/output/test_cl/gridhard/nonlinear_vf/online_property/dqn_cl/best_sp_0.01", 0]
     },
    {"label": "ReLU+ATC shift-prob=0.1",
     "control": "data/output/test_cl/gridhard/nonlinear_vf/transfer_new/goal_id_{}/dqn_cl/sweep",
     "online_measure": ["data/output/test_cl/gridhard/nonlinear_vf/online_property/dqn_cl/best", 0]
     },
    {"label": "ReLU+ATC shift-prob=0.2",
     "control": "data/output/test_cl/gridhard/nonlinear_vf/transfer_new/goal_id_{}/dqn_cl/sweep_sp_0.2",
     "online_measure": ["data/output/test_cl/gridhard/nonlinear_vf/online_property/dqn_cl/best_sp_0.2", 0]
     },
    {"label": "ReLU+ATC shift-prob=0.3",
     "control": "data/output/test_cl/gridhard/nonlinear_vf/transfer_new/goal_id_{}/dqn_cl/sweep_sp_0.3",
     "online_measure": ["data/output/test_cl/gridhard/nonlinear_vf/online_property/dqn_cl/best_sp_0.3", 0]
     },     
]

nonlinear_maze_atc_transfer_large_sweep = [
    {"label": "ReLU(L)+ATC",
     "control": "data/output/test_cl/gridhard/nonlinear_vf/transfer_new/goal_id_{}/dqn_cl/sweep_large",
     "online_measure": ["data/output/test_cl/gridhard/nonlinear_vf/online_property/dqn_cl/best_large", 0]
     }
]


nonlinear_maze_aug_sweep = [
    # {"label": "ReLU+Aug",
    #  "control": "data/output/test_cl/gridhard/nonlinear_vf/online_property/dqn_aug/sweep",
    #  "best": "2"
    #  },
     {"label": "ReLU+SR+Aug",
      "control": "data/output/test_cl/gridhard/nonlinear_vf/online_property/dqn_fta_aux/successor_as/sweep_aug",
          "best": '1'
      },
     {"label": "ReLU+SR+VFAug",
      "control": "data/output/test_cl/gridhard/nonlinear_vf/online_property/dqn_fta_aux/successor_as/sweep_aug_vf",
         "best": '2'
      },
    # {"label": "ReLU+SR+AuxAug",
    #  "control": "data/output/test_cl/gridhard/nonlinear_vf/online_property/dqn_fta_aux/successor_as/sweep_aug_aux",
    #  "best": "2"
    #  },
      {"label": "FTA eta=0.8",
      "control": "data/output/test_cl/gridhard/nonlinear_vf/online_property/dqn_fta_aug/sweep",
     "best": '2'
    },
]

nonlinear_maze_aug_best = [
    # {"label": "ReLU+Aug",
    #  "control": "data/output/test_cl/gridhard/nonlinear_vf/online_property/dqn_aug/best",
    #  "best": "2"
    #  },
     {"label": "ReLU+SR+Aug",
      "control": "data/output/test_cl/gridhard/nonlinear_vf/online_property/dqn_fta_aux/successor_as/best_aug",
      },
     {"label": "ReLU+SR+VFAug",
      "control": "data/output/test_cl/gridhard/nonlinear_vf/online_property/dqn_fta_aux/successor_as/best_aug_vf",
      },    
      {"label": "FTA eta=0.8",
      "control": "data/output/test_cl/gridhard/nonlinear_vf/online_property/dqn_fta/sweep",
     "best": '2'
    },
    # {"label": "ReLU+SR+AuxAug",
    #  "control": "data/output/test_cl/gridhard/nonlinear_vf/online_property/dqn_fta_aux/successor_as/best_aug_aux",
    #  "best": "2"
    #  },
]


nonlinear_maze_aug_transfer_sweep = [
    {"label": "ReLU+ATC shift-prob=0.1",
     "control": "data/output/test_cl/gridhard/nonlinear_vf/transfer_new/goal_id_{}/dqn_aug/sweep",
     "online_measure": ["data/output/test_cl/gridhard/nonlinear_vf/online_property/dqn_aug/best" , 0]
     },  
    {"label": "ReLU+ATC shift-prob=0.3",
     "control": "data/output/test_cl/gridhard/nonlinear_vf/transfer_new/goal_id_{}/dqn_fta_aux/successor_as/sweep_aug_aux",
     "online_measure": ["data/output/test_cl/gridhard/nonlinear_vf/online_property/dqn_fta_aux/successor_as/best_aug_aux" , 0]
     },   
]
linear_maze_atc_sweep = [
    {"label": "ReLU(L)+ATC",
     "control": "data/output/test_cl/gridhard/linear_vf/online_property/dqn_cl/sweep_large",
     "best": "8"
#     "online_measure": ["data/output/test_v13/gridhard/nonlinear_vf/original_0909/online_property/dqn/sweep/", 3]
     },
]

linear_maze_atc_best = [
    {"label": "ReLU(L)+ATC",
     "control": "data/output/test_cl/gridhard/linear_vf/online_property/dqn_cl/best_large",
     "best": "8"
#     "online_measure": ["data/output/test_v13/gridhard/nonlinear_vf/original_0909/online_property/dqn/sweep/", 3]
     },
]

nonlinear_maze_ortho_sweep = [
    {"label": "ReLU+ATC shift-prob=0.1",
     "control": "data/output/test_cl/gridhard/nonlinear_vf/online_property/dqn_ortho/sweep_large_1",
     "best": "2"
     },
    {"label": "ReLU+ATC shift-prob=0.2",
     "control": "data/output/test_cl/gridhard/nonlinear_vf/online_property/dqn_ortho/sweep_large_0.1",
     "best": "3"
     },
    {"label": "ReLU+ATC shift-prob=0.3",
     "control": "data/output/test_cl/gridhard/nonlinear_vf/online_property/dqn_ortho/sweep_large_0.01",
     "best": "3"
     },
    {"label": "ReLU+ATC shift-prob=0",
      "control": "data/output/test_cl/gridhard/nonlinear_vf/online_property/dqn_ortho/sweep_large_0.001",
     "best": '3'
    },
    {"label": "FTA eta=0.8",
      "control": "data/output/test_cl/gridhard/nonlinear_vf/online_property/dqn_ortho/sweep_0.001",
     "best": '2'
    },
]
nonlinear_maze_transfer_ortho_sweep = [
    {"label": "ReLU+ATC shift-prob=0.1",
     "control": "data/output/test_cl/gridhard/nonlinear_vf/transfer_new/goal_id_{}/dqn_ortho/sweep_large_1",
     "online_measure": ["data/output/test_cl/gridhard/nonlinear_vf/online_property/dqn_ortho/best_large_1" , 0]
     },
    {"label": "ReLU+ATC shift-prob=0.2",
     "control": "data/output/test_cl/gridhard/nonlinear_vf/transfer_new/goal_id_{}/dqn_ortho/sweep_0.001",
     "online_measure": ["data/output/test_cl/gridhard/nonlinear_vf/online_property/dqn_ortho/best_0.001" , 0]
     },
]

nonlinear_maze_transfer_ftaa_sweep = [
    {"label": "ReLU+ATC",
     "control": "data/output/test_cl/gridhard/linear_vf/online_property/dqn_fta_cl/sweep",
     "best": "0"
     },

]
nonlinear_maze_laplacian_ortho_sweep = [
     {"label": "ReLU+ATC",
      "control": "data/output/test_cl/gridhard/nonlinear_vf/online_property/dqn_aux/laplacian/best",
      "best": "6"
     },
    #  {"label": "ReLU",
    #  "control": "data/output/test_cl/gridhard/nonlinear_vf/online_property/dqn_aux/orthogonality/best_v1",
    #  "best": "21"
    #  },
    #  {"label": "ReLU+ATC",
    #  "control": "data/output/test_cl/gridhard/nonlinear_vf/online_property/dqn_aux/orthogonality/best_v2",
    #  "best": "22"
    #  },
    # {"label": "ReLU+ATC shift-prob=0.1",
    #  "control": "data/output/test_cl/gridhard/nonlinear_vf/online_property/dqn_ortho/best_large_1",
    #  "best": "2"
    #  },
    # {"label": "ReLU+ATC shift-prob=0.2",
    #   "control": "data/output/test_cl/gridhard/nonlinear_vf/online_property/dqn_ortho/best_0.001",
    #  "best": '2'
    # },
    
]


nonlinear_maze_laplacian_ortho_sweep = [
    {
        "label": "ReLU+ATC",
        "control": "data/output/test_cl/gridhard/nonlinear_vf/online_property/dqn_aux/laplacian/best_extend",
        "best": "21"
     },
#      {
#         "label": "ReLU",
#         "control": "data/output/test_cl/gridhard/nonlinear_vf/online_property/dqn_aux/laplacian/sweep_large",
#  #    "online_property": ["data/output/test_cl/gridhard/nonlinear_vf/online_property/dqn_aux/laplacian/best", 0],
#         "best": "27"
#     },
]

nonlinear_maze_diversity_sweep = [
    # {
    #     "label": "ReLU+ATC",
    #     "control": "data/output/test_cl/gridhard/nonlinear_vf/online_property/dqn_aux/diversity/sweep",
    #     "best": "21"
    #  },
    #  {"label": "ReLU",
    #   "control": "data/output/test_cl/gridhard/nonlinear_vf/online_property/dqn_aug/best",
    #   "best": "21"
    #  },
    #  {"label": "ReLU",
    #   "control": "data/output/test_cl/gridhard/nonlinear_vf/online_property/dqn_aux/diversity/sweep_v1",
    #   "best": "21"
    #  },
    #  {"label": "ReLU+ATC shift-prob=0",
    #   "control": "data/output/test_cl/gridhard/nonlinear_vf/online_property/dqn_aux/diversity/sweep_v2",
    #   "best": "22"
    #  },
    # {"label": "ReLU+ATC shift-prob=0.01",
    #  "control": "data/output/test_cl/gridhard/nonlinear_vf/online_property/dqn_aux/diversity/sweep_v3",
    #  "best": "2"
    #  },
    # {"label": "ReLU+ATC shift-prob=0.1",
    #  "control": "data/output/test_cl/gridhard/nonlinear_vf/online_property/dqn_aux/diversity/sweep_v4",
    #  "best": '2'
    # },
    # {"label": "ReLU+ATC shift-prob=0.2",
    #  "control": "data/output/test_cl/gridhard/nonlinear_vf/online_property/dqn_aux/diversity/sweep_v5",
    #  "best": '2'
    #  },
    # {"label": "ReLU+ATC shift-prob=0",
    #  "control": "data/output/test_cl/gridhard/nonlinear_vf/online_property/dqn_aux/diversity/sweep_v6",
    #  "best": "27"
    #  },
    # {"label": "ReLU+ATC shift-prob=0.01",
    #  "control": "data/output/test_cl/gridhard/nonlinear_vf/online_property/dqn_aux/diversity/sweep_v7",
    #  "best": "27"
    #  },
    # {"label": "ReLU+ATC shift-prob=0.1",
    #  "control": "data/output/test_cl/gridhard/nonlinear_vf/online_property/dqn_aux/diversity/sweep_v8",
    #  "best": '27'
    #  },
    # {"label": "ReLU+ATC shift-prob=0.2",
    #  "control": "data/output/test_cl/gridhard/nonlinear_vf/online_property/dqn_aux/diversity/sweep_v9",
    #  "best": '27'
    #  },
    {"label": "ReLU+ATC shift-prob=0",
     "control": "data/output/test_cl/gridhard/nonlinear_vf/online_property/dqn_aux/diversity/sweep_v6_small",
     "best": "25"
     },
    {"label": "ReLU+ATC shift-prob=0.01",
     "control": "data/output/test_cl/gridhard/nonlinear_vf/online_property/dqn_aux/diversity/sweep_v7_small",
     "best": "25"
     },
#      {
#         "label": "ReLU",
#         "control": "data/output/test_cl/gridhard/nonlinear_vf/online_property/dqn_aux/laplacian/sweep_large",
#  #    "online_property": ["data/output/test_cl/gridhard/nonlinear_vf/online_property/dqn_aux/laplacian/best", 0],
#         "best": "27"
#     },
]

nonlinear_maze_diversity_best = [
    # {
    #     "label": "ReLU+ATC",
    #     "control": "data/output/test_cl/gridhard/nonlinear_vf/online_property/dqn_aux/diversity/sweep",
    #     "best": "21"
    #  },
    #  {"label": "ReLU",
    #   "control": "data/output/test_cl/gridhard/nonlinear_vf/online_property/dqn_aug/best",
    #   "best": "21"
    #  },
    #  {"label": "ReLU",
    #   "control": "data/output/test_cl/gridhard/nonlinear_vf/online_property/dqn_aux/diversity/sweep_v1",
    #   "best": "21"
    #  },
    #  {"label": "ReLU+ATC shift-prob=0",
    #   "control": "data/output/test_cl/gridhard/nonlinear_vf/online_property/dqn_aux/diversity/sweep_v2",
    #   "best": "22"
    #  },
    # {"label": "ReLU+ATC shift-prob=0.01",
    #  "control": "data/output/test_cl/gridhard/nonlinear_vf/online_property/dqn_aux/diversity/sweep_v3",
    #  "best": "2"
    #  },
    # {"label": "ReLU+ATC shift-prob=0.1",
    #  "control": "data/output/test_cl/gridhard/nonlinear_vf/online_property/dqn_aux/diversity/sweep_v4",
    #  "best": '2'
    # },
    # {"label": "ReLU+ATC shift-prob=0.2",
    #  "control": "data/output/test_cl/gridhard/nonlinear_vf/online_property/dqn_aux/diversity/sweep_v5",
    #  "best": '2'
    #  },
    # {"label": "ReLU+ATC shift-prob=0",
    #  "control": "data/output/test_cl/gridhard/nonlinear_vf/online_property/dqn_aux/diversity/best_v6",
    #  "best": "27"
    #  },
    # {"label": "ReLU+ATC shift-prob=0.01",
    #  "control": "data/output/test_cl/gridhard/nonlinear_vf/online_property/dqn_aux/diversity/best_v7",
    #  "best": "27"
    #  },
    # {"label": "ReLU+ATC shift-prob=0.1",
    #  "control": "data/output/test_cl/gridhard/nonlinear_vf/online_property/dqn_aux/diversity/best_v8",
    #  "best": '27'
    #  },
    # {"label": "ReLU+ATC shift-prob=0.2",
    #  "control": "data/output/test_cl/gridhard/nonlinear_vf/online_property/dqn_aux/diversity/best_v9",
    #  "best": '27'
    #  },
    {"label": "ReLU+ATC shift-prob=0",
     "control": "data/output/test_cl/gridhard/nonlinear_vf/online_property/dqn_aux/diversity/best_v6_small",
     "best": "22"
     },
    {"label": "ReLU+ATC shift-prob=0.01",
     "control": "data/output/test_cl/gridhard/nonlinear_vf/online_property/dqn_aux/diversity/best_v7_small",
     "best": "2"
     },
#      {
#         "label": "ReLU",
#         "control": "data/output/test_cl/gridhard/nonlinear_vf/online_property/dqn_aux/laplacian/sweep_large",
#  #    "online_property": ["data/output/test_cl/gridhard/nonlinear_vf/online_property/dqn_aux/laplacian/best", 0],
#         "best": "27"
#     },
]

nonlinear_maze_transfer_diversity_sweep = [
    {
        "label": "ReLU",
        "control": "data/output/test_cl/gridhard/nonlinear_vf/transfer_new/goal_id_{}/dqn_aux/diversity/sweep_v6_small",
        "online_measure": ["data/output/test_cl/gridhard/nonlinear_vf/online_property/dqn_aux/diversity/sweep_v6_small", 0]
    },
    {
        "label": "ReLU+ATC",
        "control": "data/output/test_cl/gridhard/nonlinear_vf/transfer_new/goal_id_{}/dqn_aux/diversity/sweep_v7_small",
        "online_measure": ["data/output/test_cl/gridhard/nonlinear_vf/online_property/dqn_aux/diversity/sweep_v7_small",
                           0]
    },
    {
        "label": "ReLU+ATC shift-prob=0",
        "control": "data/output/test_cl/gridhard/nonlinear_vf/transfer_new/goal_id_{}/dqn_aux/diversity/sweep_v6",
        "online_measure": ["data/output/test_cl/gridhard/nonlinear_vf/online_property/dqn_aux/diversity/sweep_v6", 0]
    },
    {
        "label": "ReLU+ATC shift-prob=0.01",
        "control": "data/output/test_cl/gridhard/nonlinear_vf/transfer_new/goal_id_{}/dqn_aux/diversity/sweep_v7",
        "online_measure": ["data/output/test_cl/gridhard/nonlinear_vf/online_property/dqn_aux/diversity/sweep_v7", 0]
    },
    {
        "label": "ReLU+ATC shift-prob=0.1",
        "control": "data/output/test_cl/gridhard/nonlinear_vf/transfer_new/goal_id_{}/dqn_aux/diversity/sweep_v8",
        "online_measure": ["data/output/test_cl/gridhard/nonlinear_vf/online_property/dqn_aux/diversity/sweep_v8", 0]
    },
    {
        "label": "ReLU+ATC shift-prob=0.2",
        "control": "data/output/test_cl/gridhard/nonlinear_vf/transfer_new/goal_id_{}/dqn_aux/diversity/sweep_v9",
        "online_measure": ["data/output/test_cl/gridhard/nonlinear_vf/online_property/dqn_aux/diversity/sweep_v9", 0]
    },
]

nonlinear_maze_transfer_laplacian_ortho_sweep = [

    {
        "label": "ReLU+ATC",
        "control": "data/output/test_cl/gridhard/nonlinear_vf/transfer_new/goal_id_{}/dqn_aux/laplacian/sweep_extend",
        "online_measure": ["data/output/test_cl/gridhard/nonlinear_vf/online_property/dqn_aux/laplacian/best_extend", 0]
     },
#      {
#         "label": "ReLU",
#         "control": "data/output/test_cl/gridhard/nonlinear_vf/online_property/dqn_aux/laplacian/sweep_large",
#  #    "online_property": ["data/output/test_cl/gridhard/nonlinear_vf/online_property/dqn_aux/laplacian/best", 0],
#         "best": "27"
#     },
]


nonlinear_maze_online_ortho_diversity_sweep = [
    {
        "label": "ReLU",
        "control": "data/output/test_cl/gridhard/nonlinear_vf/online_property/dqn_aux/orthogonality/sweep",
        "best": "23"
    },
    {
        "label": "ReLU+ATC",
        "control": "data/output/test_cl/gridhard/nonlinear_vf/online_property/dqn_aux/laplacian/sweep_extra",
        "best": "17"
    },
]

# best chosen based on the properties and not only the performance
nonlinear_maze_online_property_ortho_diversity_sweep = [
     {
         "label": "ReLU",
         "control": "data/output/test_cl/gridhard/nonlinear_vf/online_property/dqn_aux/laplacian/sweep_final",
         # "best": "13"
         "best": "8"
     },
    {
        "label": "FTA+ATC",
        "control": "data/output/test_cl/gridhard/nonlinear_vf/online_property/dqn_aux/laplacian/sweep_large_final",
#        "best": "34",
        "best": "39"
    },
    {
        "label": "ReLU+ATC",
        "control": "data/output/test_cl/gridhard/nonlinear_vf/online_property/dqn_aux/orthogonality/sweep_large_final",
        #"best": "13"
         "best": "3"
    },
]

# best chosen based on the properties and not only the performance
#nonlinear_maze_online_property_ortho_diversity_best_property = [
#    {
#        "label": "ReLU",
#        "control": "data/output/test_cl/gridhard/nonlinear_vf/online_property/dqn_aux/orthogonality/best_property_final",
#        #"best": "13"
#         "best": "7"
#    },
#    {
#        "label": "ReLU+ATC",
#        "control": "data/output/test_cl/gridhard/nonlinear_vf/online_property/dqn_aux/laplacian/best_property_final",
#        "best": "12"
#    },
#]
nonlinear_maze_online_property_ortho_diversity_best_property = [
     {
         "label": "ReLU",
         "control": "data/output/test_cl/gridhard/nonlinear_vf/online_property/dqn_aux/laplacian/best_property_final",
         # "best": "13"
         "best": "8"
     },
    {
        "label": "FTA+ATC",
        "control": "data/output/test_cl/gridhard/nonlinear_vf/online_property/dqn_aux/laplacian/best_property_large_final",
#        "best": "34",
        "best": "39"
    },
    {
        "label": "ReLU+ATC",
        "control": "data/output/test_cl/gridhard/nonlinear_vf/online_property/dqn_aux/orthogonality/best_property_large_final",
        #"best": "13"
         "best": "3"
    },
]

nonlinear_maze_transfer_ortho_diversity_sweep = [
    {
        "label": "ReLU",
        "control": "data/output/test_cl/gridhard/nonlinear_vf/transfer_new/goal_id_{}/dqn_aux/orthogonality/sweep",
        "online_measure": ["data/output/test_cl/gridhard/nonlinear_vf/online_property/dqn_aux/orthogonality/best", 0],
    },
    {
        "label": "ReLU+ATC",
        "control": "data/output/test_cl/gridhard/nonlinear_vf/transfer_new/goal_id_{}/dqn_aux/laplacian/sweep_extra",
        "online_measure": ["data/output/test_cl/gridhard/nonlinear_vf/online_property/dqn_aux/laplacian/best_extra", 0],
    },
]

nonlinear_maze_transfer_property_ortho_diversity_best_property = [
     {
         "label": "ReLU",
         "control": "data/output/test_cl/gridhard/nonlinear_vf/transfer_new/goal_id_{}/dqn_aux/laplacian/sweep_property_final",
         "online_measure": ["data/output/test_cl/gridhard/nonlinear_vf/online_property/dqn_aux/laplacian/best_property_final",
                            0],
         # "best": "13"
         "best": "8"
     },
    {
        "label": "FTA+ATC",
        "control": "data/output/test_cl/gridhard/nonlinear_vf/transfer_new/goal_id_{}/dqn_aux/laplacian/sweep_property_large_final",
        "online_measure": [
            "data/output/test_cl/gridhard/nonlinear_vf/online_property/dqn_aux/laplacian/best_property_large_final",
            0],
    },
    {
        "label": "ReLU+ATC",
        "control": "data/output/test_cl/gridhard/nonlinear_vf/transfer_new/goal_id_{}/dqn_aux/orthogonality/sweep_property_large_final",
        "online_measure": [
            "data/output/test_cl/gridhard/nonlinear_vf/online_property/dqn_aux/orthogonality/best_property_large_final",
            0],
    },
    {
        "label": "ReLU+ATC shift-prob=0",
        "control": "data/output/test_cl/gridhard/nonlinear_vf/transfer_new/goal_id_{}/dqn_aux/orthogonality/sweep_property_final",
        "online_measure": [
            "data/output/test_cl/gridhard/nonlinear_vf/online_property/dqn_aux/orthogonality/best_property_final",
            0],
    },
]


nonlinear_maze_online_dynamic = [
    {
        "label": "ReLU+ATC",
        "control": "data/output/test_cl/gridhard/nonlinear_vf/online_property/dqn_aux/dynamic_awareness/best_property",
    },
    {
        "label": "FTA+ATC",
        "control": "data/output/test_cl/gridhard/nonlinear_vf/online_property/dqn_aux/dynamic_awareness/best_property_large"
    },
  #  {"label": "ReLU",
  #   "control": "data/output/test_v13/gridhard/nonlinear_vf/original_0909/transfer/goal_id_{}/dqn/sweep/",
  #   "online_measure": ["data/output/test_v13/gridhard/nonlinear_vf/original_0909/online_property/dqn/sweep/", 3]
  #   },
]

nonlinear_maze_online_dyna_ortho_laplacian = [
    {
        "label": "ReLU+ATC",
        "control": "data/output/test_cl/gridhard/nonlinear_vf/online_property/dqn_aux/dyna_ortho/best_property",
        "best": "1"
    },
    {
        "label": "FTA+ATC",
        "control": "data/output/test_cl/gridhard/nonlinear_vf/online_property/dqn_aux/laplacian/best_property_finall",
        "best": "42"
    },
  #  {"label": "ReLU",
  #   "control": "data/output/test_v13/gridhard/nonlinear_vf/original_0909/transfer/goal_id_{}/dqn/sweep/",
  #   "online_measure": ["data/output/test_v13/gridhard/nonlinear_vf/original_0909/online_property/dqn/sweep/", 3]
  #   },
]


nonlinear_maze_transfer_online_dynamic = [
    # {
    #     "label": "ReLU+ATC",
    #     "control": "data/output/test_cl/gridhard/nonlinear_vf/transfer_new/goal_id_{}/dqn_aux/dynamic_awareness/sweep_property",
    #     "online_measure": [
    #         "data/output/test_cl/gridhard/nonlinear_vf/online_property/dqn_aux/dynamic_awareness/best_property",
    #         0],
    # },
    # {
    #     "label": "FTA+ATC",
    #     "control": "data/output/test_cl/gridhard/nonlinear_vf/transfer_new/goal_id_{}/dqn_aux/dynamic_awareness/sweep_large_property",
    #     "online_measure": [
    #         "data/output/test_cl/gridhard/nonlinear_vf/online_property/dqn_aux/dynamic_awareness/best_large_property",
    #         0],
    # },
    {
        "label": "ReLU+ATC",
        "control": "data/output/test_cl/gridhard/nonlinear_vf/transfer_new/goal_id_{}/dqn_aux/laplacian/sweep_property_finall",
        "online_measure": [
            "data/output/test_cl/gridhard/nonlinear_vf/online_property/dqn_aux/laplacian/best_property_finall",
            0],
    },
    {
        "label": "FTA+ATC",
        "control": "data/output/test_cl/gridhard/nonlinear_vf/transfer_new/goal_id_{}/dqn_aux/dyna_ortho/sweep_property",
        "online_measure": [
            "data/output/test_cl/gridhard/nonlinear_vf/online_property/dqn_aux/dyna_ortho/best_property",
            0],
    },
  #  {"label": "ReLU",
  #   "control": "data/output/test_v13/gridhard/nonlinear_vf/original_0909/transfer/goal_id_{}/dqn/sweep/",
  #   "online_measure": ["data/output/test_v13/gridhard/nonlinear_vf/original_0909/online_property/dqn/sweep/", 3]
  #   },
]
