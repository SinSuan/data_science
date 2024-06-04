import os
import pickle

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