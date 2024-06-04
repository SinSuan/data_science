import numpy as np
from module.for_model.checkpoint import load_checkpoint

def specific_loss(name_test: str, n: int, type_loss: int, idx_checkpoint=None):
    """
    Var:
        name_test:  str
            folder name
        
        n: int
            number of nodes

        type_loss:  int
            0      in-sample-loss
            1     out-of-sample-loss
        
        idx_checkpoint: None or int
            best checkpoint for None (default)
            specific checkpoint for int

    Return:
        result_loss: List[np.ndarray]
        result_idx: List[np.ndarray]
    
    Q: When to use the result? 
    A: To get the model with desired loss.
    eg:
        m: model_idx (int)
        record_param[m, result_idx[m]]
        
    """
    path_folder = f"checkpoints\\{name_test}\\node_{n:02d}"
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