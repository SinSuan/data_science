import os
import pickle

def save_checkpoint(path_fold, record_loss, record_param):
    if not os.path.exists(path_fold):
        os.makedirs(path_fold)
        
    path_loss = f"{path_fold}loss.pkl"
    with open(path_loss, 'wb') as f:
        pickle.dump(record_loss, f)

    path_param = f"{path_fold}param.pkl"
    with open(path_param , 'wb') as f:
        pickle.dump(record_param, f)
        
    return path_loss, path_param