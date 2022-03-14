import os
import numpy as np
import matplotlib.pyplot as plt


def load_file(path='data/dataset/gridhard/srs/gridhard_srs.npy'):
    if os.path.isfile(path):
        srs = np.load(path)
        return srs
    else:
        raise FileNotFoundError

def l2dist(v1, v2):
    return np.linalg.norm(v1-v2)

def l1dist(v1, v2):
    return np.linalg.norm(v1-v2, 1)

def infdist(v1, v2):
    return np.linalg.norm(v1-v2, np.inf)

def dotproddist(v1, v2):
    return np.dot(v1, v2)

def cosinsimil(v1, v2):
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        return 0
    else:
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def distance_single_goal(sr, g_idx, distance_fn):
    dvec = np.zeros(len(sr))
    goal = sr[g_idx]
    assert goal.sum() == 1
    # g_onehot = np.where(goal==1)[0][0]
    # sr[:, g_onehot] = 0
    for idx, row in enumerate(sr):
        dvec[idx] = distance_fn(row, goal)
    return dvec

def distance_to_goals(srs, distance_fn):
    d_mat = np.zeros((len(srs), len(srs[0])))
    for goal, sr in enumerate(srs):
        dvec = distance_single_goal(sr, goal, distance_fn)
        d_mat[goal, :] = dvec
    return d_mat

def all_goals_2d(env_name='GridHardTabular'):
    if env_name=='GridHardTabular':
        from sr_computation import GridHardTabular
        env = GridHardTabular(0)
        state_idx_map = env.state_idx_map
        max_x_dim, max_y_dim = env.max_x_dim, env.max_y_dim
        return state_idx_map, (max_x_dim, max_y_dim)
    else:
        raise NotImplementedError

"""
Measure distance based on SR distance given each fixed goal
"""
def dist_between_goals_v0(dist2goal, states2d, goal, size, distance_fn):
    goals_dist = np.zeros(size)
    goal_idx = states2d.index(goal)
    goal_vec = dist2goal[goal_idx]
    # draw_sr_dist(goal_vec, states2d, goal, size)
    for i in range(len(states2d)):
        s2d = states2d[i]
        # draw_sr_dist(dist2goal[i], states2d, s2d, size)
        goals_dist[s2d] = distance_fn(dist2goal[i], goal_vec)
    return goals_dist

"""
Measure distance based on concatenated SRs
"""
def dist_between_goals_v1(csrs, states2d, goal, size, distance_fn=None):
    goals_dist = np.zeros(size)
    goal_idx = states2d.index(goal)
    goal_vec = csrs[goal_idx]
    for i in range(len(states2d)):
        s2d = states2d[i]
        # goals_dist[s2d] = cosinsimil(csrs[i], goal_vec)
        if distance_fn is None:
            goals_dist[s2d] = dotproddist(csrs[i], goal_vec)
        else:
            goals_dist[s2d] = distance_fn(csrs[i], goal_vec)
    return goals_dist

def draw_sr_dist(dists, states2d, original_goal, size):
    formated = np.zeros(size)
    for i, s in enumerate(states2d):
        formated[s] = dists[i]
    plt.figure(figsize=(12,9))
    plt.imshow(formated, interpolation='nearest', cmap="Blues")
    for k in range(formated.shape[0]):
        for j in range(formated.shape[1]):
            if formated[k, j] != 0:
                plt.text(j, k, "{:1.1f}".format(formated[k, j]),
                         ha="center", va="center", color="orange")
    plt.text(original_goal[1], original_goal[0], "O",
             ha="center", va="center", color="black")
    plt.savefig("plot/img/SR_dist2goal_{}.png".format(original_goal), dpi=100)
    # plt.show()

def test_sr_dist(dist2goal, states2d, original_goal, size):
    goal_idx = states2d.index(original_goal)
    draw_sr_dist(dist2goal[goal_idx], states2d, original_goal, size)

def draw_goal_dist(goals_dist, original, name):
    # plt.figure(figsize=(12,9))
    plt.figure(figsize=(16,12))
    plt.imshow(goals_dist, interpolation='nearest', cmap="Blues")
    for k in range(goals_dist.shape[0]):
        for j in range(goals_dist.shape[1]):
            if goals_dist[k, j] != 0:
                plt.text(j, k, "{:1.4f}".format(goals_dist[k, j]),
                         ha="center", va="center", color="orange")
    plt.text(original[1], original[0], "O",
             ha="center", va="center", color="black")

    plt.savefig("{}".format(name), dpi=100, bbox_inches='tight', pad_inches=0)
    # plt.show()

def rank_goals(goals_dist, original, name, reversed=False, all_goals=None):
    goals_dist[np.where(goals_dist==0)] = np.inf
    if not reversed:
        rank = goals_dist.ravel().argsort().argsort().reshape(goals_dist.shape)
    else:
        rank = (-1*goals_dist).ravel().argsort().argsort().reshape(goals_dist.shape)
        rank = rank - len(np.where(np.isinf(goals_dist))[0]) - 1

    rank[np.where(np.isinf(goals_dist))] = -1

    plt.figure(figsize=(12,9))
    plt.imshow(rank, interpolation='nearest', cmap="Blues")
    for k in range(rank.shape[0]):
        for j in range(rank.shape[1]):
            if rank[k, j] != -1:
                plt.text(j, k, "{:1.0f}".format(rank[k, j]+1),
                         ha="center", va="center", color="orange")
    plt.text(original[1], original[0], "O",
             ha="center", va="center", color="black")
    plt.savefig("{}".format(name), dpi=100, bbox_inches='tight', pad_inches=0)
    # plt.show()

    if all_goals:
        ranks = {}
        id2coord = {}
        for i, goal in enumerate(all_goals):
            ranks[i] = rank[goal[0], goal[1]]
            id2coord[i] = goal
            print(i, goal, ranks[i])

        savepath = "data/dataset/gridhard/srs/goal{}_simrank.npy".format(original)
        np.save(savepath, ranks)
        print("Save rank in {}".format(savepath))

        savepath = "data/dataset/gridhard/srs/goal{}_id2coord.npy".format(original)
        np.save(savepath, id2coord)
        print("Save coord in {}".format(savepath))

def concatenate_srs(srs):
    csrs = np.zeros((len(srs), np.prod(srs.shape[1:])))
    for i, sr in enumerate(srs):
        csr = np.concatenate(sr)
        csrs[i] = csr
    return csrs

def fix_start_srs(srs, chosen):
    csrs = np.zeros(srs.shape[:2])
    for i, sr in enumerate(srs):
        csr = sr[chosen]
        csrs[i] = csr
    return csrs

def main(metrics = "l2"):
    if metrics == "l2":
        distance_fn = l2dist
    elif metrics == "l1":
        distance_fn = l1dist
    elif metrics == "inf":
        distance_fn = infdist
    elif metrics == "dot":
        distance_fn = dotproddist
    elif metrics == "cos":
        distance_fn = cosinsimil
    else:
        raise NotImplementedError

    srs = load_file()
    states2d, size = all_goals_2d()

    """
    Distance
    """
    # dist2goal = distance_to_goals(srs, distance_fn)
    # # # Check SR distance
    # # test_sr_dist(dist2goal, states2d, (9, 9), size)
    # # test_sr_dist(dist2goal, states2d, (7, 7), size)
    # goals_dist = dist_between_goals_v0(dist2goal, states2d, (9,9), size, distance_fn)
    # draw_goal_dist(goals_dist, (9, 9), "plot/img/dist{}.png".format(metrics))
    # rank_goals(goals_dist, (9, 9), "plot/img/distrank{}.png".format(metrics))

    """
    Dot product similarity
    """
    csrs = concatenate_srs(srs)
    goals_dist = dist_between_goals_v1(csrs, states2d, (9,9), size)
    draw_goal_dist(goals_dist, (9, 9), "plot/img/simil.png")
    rank_goals(goals_dist, (9, 9), "plot/img/similrank.png", reversed=True, all_goals=states2d)

    """
    Fixed start similarity
    """
    # for fs in np.random.randint(0, len(states2d), size=5):
    #     csrs = fix_start_srs(srs, fs)
    #     goals_dist = dist_between_goals_v1(csrs, states2d, (9,9), size)
    #     draw_goal_dist(goals_dist, (9, 9), "plot/img/fixstart{}.png".format(states2d[fs]))
    #     rank_goals(goals_dist, (9, 9), "plot/img/fixstartrank{}.png".format(states2d[fs]), reversed=True)

def rank_by_value():
    vmax = load_file(path='data/dataset/gridhard/srs/gridhard_vmax.npy')
    states2d, size = all_goals_2d()

    goals_dist = np.zeros(size)
    goal_idx = states2d.index((9,9))
    for i in range(len(states2d)):
        s2d = states2d[i]
        goals_dist[s2d] = vmax[i]
    s2d_goal = states2d[goal_idx]
    goals_dist[s2d_goal] = goals_dist.max()+0.1
    draw_goal_dist(goals_dist, (9, 9), "plot/img/vmax_simil.png")
    rank_goals(goals_dist, (9, 9), "plot/img/vmax_similrank.png", reversed=True, all_goals=states2d)

    # vs = load_file(path='data/dataset/gridhard/srs/gridhard_vs.npy')
    # states2d, size = all_goals_2d()
    #
    # goals_dist = dist_between_goals_v1(vs, states2d, (9,9), size, distance_fn=l2dist)
    # draw_goal_dist(goals_dist, (9, 9), "plot/img/vs_simil.png")
    # rank_goals(goals_dist, (9, 9), "plot/img/vs_similrank.png", reversed=True, all_goals=states2d)

if __name__ == '__main__':
    main("dot")
    # rank_by_value()