from dotenv import load_dotenv
from module.for_dataset.create_dataset import create_dataset
from module.for_dataset.normalization import normalization

from module.for_model.shallow_nn import shallow_nn
from module.for_model.training import train

def main():
    
    X_train, Y_train, X_test, Y_test = create_dataset(0,20000)
    
    ## preprocess
    
    # reshape for the model output
    Y_train = Y_train.reshape((-1,1))
    Y_test = Y_test.reshape((-1,1))
    
    # normalize
    normalizer = normalization(X_train)
    X_train_n = normalizer.normalize(X_train)
    Y_train_n = normalizer.normalize(Y_train)
    X_test_n = normalizer.normalize(X_test)
    Y_test_n = normalizer.normalize(Y_test)
    
    # zip
    data = [X_train_n, Y_train_n, X_test_n, Y_test_n]
    
    # construct model and train
    nn_1 = \
        shallow_nn(
        n = 5,
        layer_initializer = 0,
        learning_rate = 0.0001,
        param_hidden = 1
    )
    ttl_loss, ttl_param = \
        train(
        nn = nn_1,
        num_epoch = 10000,
        size_batch = 1000,
        data = data
    )
    print(ttl_loss)
    print(ttl_param)

if __name__ == "__main__":
    load_dotenv(".env")
    main()