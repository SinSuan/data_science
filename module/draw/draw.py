import matplotlib.pyplot as plt
from module.for_model.checkpoint import specific_loss

def display_boxplot(name_test, type_loss):
    """
    Var:
    
        name_test:  str
            experiment name
        
        type_loss:  int
            0   in-sample-loss
            1   out-of-sample-loss
            
    Return: None
    """
    #     certain ttl_loss of certain experiment
    # """

    ttl_loss = [
        specific_loss(name_test, n, type_loss)[0] \
            for n in range(1,21)
    ]
    print(ttl_loss[0])
    
    plt.boxplot(ttl_loss, patch_artist=True)

    # 添加标题和标签
    plt.title("Model with Different Hidden Nodes")
    plt.xlabel("Number of Nodes")
    plt.ylabel("Loss")

    # 显示图表
    plt.show()
    return ttl_loss
    