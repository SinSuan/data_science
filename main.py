from dotenv import load_dotenv
from module.for_dataset.create_dataset import create_dataset
from module.for_dataset.normalization import normalization
from module.for_model.shallow_nn import shallow_nn
from module.for_model.training import train
from module.for_model.save_checkpoint import save_checkpoint

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
    record_loss = []
    record_param = []
    for n in range(0,2):

        loss_n = []
        param_n = []
        for layer_initializer in range(2):

            loss_init = []
            param_init = []
            for param_hidden in range(0,2):
                ttl_loss, ttl_param = \
                    construct_and_train_model(n+1, layer_initializer, param_hidden+1, data)

                loss_init.append(ttl_loss)
                param_init.append(ttl_param)

            loss_n.append(loss_init)
            param_n.append(param_init)

        record_loss.append(loss_n)
        record_param.append(param_n)
    
    # save_checkpoint
    path_fold = "checkpoints\\2024_0603_0142\\"
    path_loss, path_param = save_checkpoint(path_fold, record_loss, record_param)
    
    print(path_loss)
    print(path_param)

def construct_and_train_model(n, layer_initializer, param_hidden, data):
    """ construct a model and train it
    """
    nn = \
        shallow_nn(
        n = n,
        layer_initializer = layer_initializer,
        learning_rate = 0.0001,
        param_hidden = param_hidden
    )
    ttl_loss, ttl_param = \
        train(
        nn = nn,
        num_epoch = 10,
        size_batch = 10,
        data = data
    )
    return ttl_loss, ttl_param

if __name__ == "__main__":
    load_dotenv(".env")
    main()