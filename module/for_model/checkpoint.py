import os
import pickle
import numpy as np

from module.for_model.shallow_nn import shallow_nn

DEBUGGER = os.getenv("DEBUGGER")

def save_checkpoint(path_folder, record_loss, record_param):
    if DEBUGGER=="True":
        print("enter save_checkpoint")

    if not os.path.exists(path_folder):
        os.makedirs(path_folder)
    path_loss = f"{path_folder}\\loss.pkl"
    with open(path_loss, 'wb') as f:
        pickle.dump(record_loss, f)
    path_param = f"{path_folder}\\param.pkl"
    with open(path_param , 'wb') as f:
        pickle.dump(record_param, f)

    if DEBUGGER=="True":
        print("exit save_checkpoint")
    return path_loss, path_param

def load_checkpoint(path_folder, loss_or_param):
    """
    Var:
        path_folder:  str
            experiment name
    
        loss_or_param: str
            "loss" or "param"
    """
    if DEBUGGER=="True":
        print("enter load_checkpoint")


    # path_loss = f"{path_folder}\\loss.pkl"
    # with open(path_loss, 'rb') as f:
    #     record_loss = pickle.load(f)
    # path_param = f"{path_folder}\\param.pkl"
    # with open(path_param, 'rb') as f:
    #     record_param = pickle.load(f)
    
    
    path_loss = f"{path_folder}\\{loss_or_param}.pkl"
    with open(path_loss, 'rb') as f:
        result = pickle.load(f)
    
    if DEBUGGER=="True":
        print("exit load_checkpoint")
    return result

def get_best_model(name_experiment, n):
    ## step 0: 調出該 best model 的 best checkpoint 的參數
    
    # 調出 best n 的所有 model 的 best checkpoint
    result_loss, result_idx = specific_loss(name_experiment, n, type_loss=1)
    
    # 找出 best model 編號
    idx_best = result_loss.argmin()
    
    # 方法請參閱 module\for_model\checkpoint.py 中 specific_loss 的函式說明
    path_folder = f"checkpoints\\{name_experiment}\\node_{n:02d}"
    ttl_param = load_checkpoint(path_folder, "param")
    param_best = ttl_param[idx_best][result_idx[idx_best]]
    
    # step 1: 建立 best model 的 best checkpoint
    nn_best = shallow_nn(n=n, layer_initializer=param_best)
    return nn_best, idx_best    

def specific_loss(name_experiment: str, n: int, type_loss: int, idx_checkpoint=None):
    """
    Var:
        name_experiment:  str
            experiment name
        
        n: int
            number of nodes

        type_loss:  int
            0   in-sample-loss
            1   out-of-sample-loss
        
        idx_checkpoint: None or int
            best checkpoint for None (default)
            specific checkpoint for int

    Return:
        result_loss: List[np.ndarray]
        result_idx: List[np.ndarray]
    
    Q: When to use the result? 
    A: To get the model with desired loss.
    eg:
        m: index of the model with desired loss (int)
        record_param[m][result_idx[m]]
        
    """
    path_folder = f"checkpoints\\{name_experiment}\\node_{n:02d}"
    record_loss = load_checkpoint(path_folder, "loss")
    record_loss = np.array(record_loss)
    # record_loss.shape == (30, num_checkpoint, 2)


    if idx_checkpoint==None:
        # 找出個模型最好的 loss_in 跟 loss_out
        checkpoint_loss = record_loss.min(axis=1)
        checkpoint_idx = record_loss.argmin(axis=1)
        # shape == (30, 2)
        # loss: num_checkpoint 個模型中的 "最小" loss 值
        # idx: 這個 loss 是第 idx 個 checkpoint 的結果
    else:
        checkpoint_loss = record_loss[:,idx_checkpoint,:]
        checkpoint_idx = idx_checkpoint*np.ones(checkpoint_loss.shape, dtype=int)


    # 選擇 loss_in 或 loss_out
    result_loss = checkpoint_loss[:,type_loss]
    result_idx = checkpoint_idx[:,type_loss]
    # shape == (30,)

    return result_loss, result_idx