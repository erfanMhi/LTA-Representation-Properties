import numpy as np

def load_goals(file):
    return np.load(file, allow_pickle=True).item()

def exchange_kv(dictionary):
    res = dict((v, k) for k, v in dictionary.items())
    return res

def main():
    goal_order = load_goals("../data/dataset/gridhard/srs/goal(9, 9)_simrank.npy")
    goal_coord = load_goals("../data/dataset/gridhard/srs/goal(9, 9)_id2coord.npy")
    order_goal = exchange_kv(goal_order)
    # for order in range(-1, 172):
    #     # print(len(list(range(-1, 172))))
    #     print(order_goal[order], end=", ")
    for order in range(-1, 172):
        print("Order {}: goal {}, coord {}".format(order+1, order_goal[order], goal_coord[order_goal[order]]))
    print()


if __name__ == '__main__':
    main()