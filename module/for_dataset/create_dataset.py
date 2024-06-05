import os
import numpy as np

DEBUGGER = os.getenv("DEBUGGER")

def create_dataset(random_seed: int, num_data: int) -> list:
    if DEBUGGER=="True":
        print("enter create_dataset")

    np.random.seed(random_seed)
    X_size = (num_data,2)

    X = np.random.randint(1000, 9999, size=X_size)
    Y = np.sum(X, axis=1)

    # for split
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

    X_train = X[mask]
    Y_train = Y[mask]
    # print(f"X_train.shape = {X_train.shape}")
    # print(f"Y_train.shape = {Y_train.shape}")
    
    data = [X_train, Y_train, X_test, Y_test]
    
    if DEBUGGER=="True":
        print("exit create_dataset")
    return data