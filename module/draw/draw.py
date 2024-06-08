import matplotlib.pyplot as plt
import numpy as np
import pickle
from module.for_model.checkpoint import specific_loss

def display_boxplot(name_experiment, type_loss):
    """
    Var:
    
        name_experiment:  str
            experiment name
        
        type_loss:  int
            0   in-sample-loss
            1   out-of-sample-loss
            
    Return: None
    """
    #     certain ttl_loss of certain experiment
    # """

    # plot boxplot
    ttl_loss = [
        specific_loss(name_experiment, n, type_loss)[0] \
            for n in range(1,21)
    ]
    # print(ttl_loss[0])
    # print( np.array(ttl_loss).shape )
    plt.boxplot(ttl_loss, patch_artist=True)

    # indicate the best box
    ttl_median = np.median(ttl_loss, axis=1)
    median_min = ttl_median.min()
    # print( ttl_median.shape )
    y_median = np.ones(21) * median_min
    x_median = np.arange(1,22) - 0.5
    plt.plot(x_median, y_median, linewidth=1, color="r")

    # 添加标题和标签
    plt.title("Model with Different Hidden Nodes")
    plt.xlabel("Number of Nodes")
    plt.ylabel("Loss")

    # 显示图表
    plt.show()
    return ttl_loss

def plot_decision_boundary(epoch, params, ax):
    """ visulize_xor_nn (繪製分類線)
    """
    ax.clear()
    
    # 把一些 data 畫上去
    path_folder = "data\\xor_normalized"
    path_data = f"{path_folder}\\2024_0606_0243.pkl"
    with open(path_data, 'rb') as f:
        data = pickle.load(f)
    X_train_n, Y_train, X_test_n, Y_test = data
    num = 40
    train_0 = X_train_n[Y_train==0][:num]
    train_1 = X_train_n[Y_train==1][:num]
    test_0 = X_test_n[Y_test==0][:num]
    test_1 = X_test_n[Y_test==1][:num]
    
    # print(epoch)
    
    ax.scatter(train_0[:,0], train_0[:,1], s=0.5, label="xor==0", color="r")
    ax.scatter(train_1[:,0], train_1[:,1], s=0.5, label="xor==1", color="k")
    ax.scatter(test_0[:,0], test_0[:,1], s=0.5, color="r")
    ax.scatter(test_1[:,0], test_1[:,1], s=0.5, color="k")
    
    # 畫線
    weights, biases = params[epoch]
    x = np.linspace(-1, 2, 100)
    for i in range(2):  # 假設有兩條分類線
        # y = -(weights[i][0] * x + biases[i]) / weights[i][1]
        t = -(weights[i][0] * x + biases[i]) / weights[i][1]
        ax.plot(x, t, label=f'Line {i+1}')
        
        
        # exponient = np.e**t
        # y = (exponient-1)/(exponient+1)
        # ax.plot(x, y, label=f'Line {i+1}')

    ax.legend()
    ax.set_title(f'Epoch {epoch+1}')
    ax.set_xlim(-1, 2)
    ax.set_ylim(-1, 2)
