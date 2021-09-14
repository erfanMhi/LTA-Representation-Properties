import numpy as np

def load_goals(file):
    return np.load(file, allow_pickle=True).item()

def exchange_kv(dictionary):
    res = dict((v, k) for k, v in dictionary.items())
    return res

def main():
    goal_order = load_goals("../data/dataset/gridhard/srs/goal(9, 9)_simrank.npy")
    order_goal = exchange_kv(goal_order)
    # for order in range(60, 85):
    for order in range(0, 172):
        # print("Order {}: goal {}".format(order, order_goal[order]))
        print(order_goal[order], end=", ")
    print()

if __name__ == '__main__':
    main()