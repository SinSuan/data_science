import os
import numpy as np

DEBUGGER = os.getenv("DEBUGGER")

# def create_dataset(random_seed: int, num_data: int) -> list:
#     if DEBUGGER=="True":
#         print("enter create_dataset")

#     np.random.seed(random_seed)
#     X_size = (num_data,2)

#     X = np.random.randint(1000, 9999, size=X_size)
#     Y = np.sum(X, axis=1)

#     # for split
#     rng = np.random.default_rng()
#     num_test = int(0.2*num_data)
#     idx_testing = rng.choice(num_data, num_test, replace=False, shuffle=False)

#     # testing data
#     X_test = X[idx_testing]
#     Y_test = Y[idx_testing]
#     # print(f"X_test.shape = {X_test.shape}")
#     # print(f"Y_test.shape = {Y_test.shape}")
    
#     # training data
#     mask = np.ones(Y.shape, dtype=bool)
#     # print(f"mask.shape = {mask.shape}")
#     mask[idx_testing] = False
#     # print(f"mask.shape = {mask.shape}")

#     X_train = X[mask]
#     Y_train = Y[mask]
#     # print(f"X_train.shape = {X_train.shape}")
#     # print(f"Y_train.shape = {Y_train.shape}")
    
#     data = [X_train, Y_train, X_test, Y_test]
    
#     if DEBUGGER=="True":
#         print("exit create_dataset")
#     return data

def create_dataset(random_seed: int, num_data: int, type_set: str) -> list:
    """
    Var:
        type_set:
            "sum" or "xor"
    """

    if type_set=="sum":
        raw_data = dataset_for_part2(random_seed, num_data)
    elif type_set=="xor":
        raw_data = dataset_for_part3(random_seed, num_data)

    data = split_dataset(raw_data)
    return data

def dataset_for_part2(random_seed, num_data):
    np.random.seed(random_seed)
    X_size = (num_data,2)

    X = np.random.randint(1000, 9999, size=X_size)
    Y = np.sum(X, axis=1)
    return X, Y

def dataset_for_part3(random_seed, num_data):
    np.random.seed(random_seed)

    len_x_flat = 2*num_data

    rng = np.random.default_rng()
    x_bool = rng.choice(2, len_x_flat, replace=True, shuffle=False)

    x_rand = x_bool.astype(float)   # choice 會強制把未來的新值都轉 int

    # # 寫法一    
    # idx_0 = np.where(x_rand==0)[0]
    # idx_1 = np.where(x_rand==1)[0]
    # rand_0 = np.random.uniform(low=-0.5, high=0.2, size=idx_0.shape)
    # rand_1 = np.random.uniform(low=0.8, high=1.5, size=idx_1.shape)
    # np.put(x_rand, idx_0, rand_0)
    # np.put(x_rand, idx_1, rand_1)
    
    # 寫法二:   比較簡潔，但不確定這樣資料是否均勻
    rand_0 = np.random.uniform(low= -0.5, high=0.2, size=len_x_flat)
    rand_1 = np.random.uniform(low=  0.8, high=1.5, size=len_x_flat)
    x_rand = np.where(x_rand==0, rand_0, rand_1)
    
    # print(f"x_bool = {x_bool}")

    X = x_rand.reshape((num_data,2))

    x_for_y = x_bool.reshape((num_data,2))
    Y = x_for_y[:,0] ^ x_for_y[:,1]
    # print(type(Y))
    # print(Y.shape)
    # print(X.shape)

    # import matplotlib.pyplot as plt
    # plt.hist(rand_0)
    # plt.show()

    return X, Y

def split_dataset(data):
    X, Y = data
    num_data = len(X)
    
    rng = np.random.default_rng()
    num_test = int(0.2*num_data)
    idx_testing = rng.choice(num_data, num_test, replace=False, shuffle=False)

    # testing data
    X_test = X[idx_testing]
    Y_test = Y[idx_testing]
    # print(f"X_test.shape = {X_test.shape}")
    # print(f"Y_test.shape = {Y_test.shape}")
    
    # training data
    mask = np.ones(Y.shape, dtype=bool)
    # print(f"mask.shape = {mask.shape}")
    mask[idx_testing] = False
    # print(f"mask.shape = {mask.shape}")
    # print(f"X = {X}")
    # print(f"mask = {mask}")
    X_train = X[mask]
    Y_train = Y[mask]
    # print(f"X_train.shape = {X_train.shape}")
    # print(f"Y_train.shape = {Y_train.shape}")
    
    data = [X_train, Y_train, X_test, Y_test]
    return data