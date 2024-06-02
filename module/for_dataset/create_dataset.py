import numpy as np

def create_dataset(random_seed: int, num_data: int) -> list:

    np.random.seed(random_seed)
    X_size = (num_data,2)

    X = np.random.randint(1000, 9999, size=X_size)
    Y = np.sum(X, axis=1)

    # for split
    rng = np.random.default_rng()
    idx_testing = rng.choice(20000, 4000, replace=False, shuffle=False)

    # testing data
    X_test = X[idx_testing]
    Y_test = Y[idx_testing]

    # training data
    mask = np.ones(Y.shape, dtype=bool)
    mask[idx_testing] = False

    X_train = X[mask]
    Y_train = Y[mask]
    
    data = [X_train, Y_train, X_test, Y_test]
    
    return data