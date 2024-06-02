import os
import numpy as np

DEBUGGER = os.getenv("DEBUGGER")

def train(nn, num_epoch, size_batch, data):
    """
    Var:
        nn: shallow_nn
    """
    x_train, y_train, _, _ = data
    ttl_loss = []
    ttl_param = []
    num_batch = len(x_train)//size_batch + 1
    rec_epoch = num_epoch//10

    loss_in, loss_out, param = get_checkpoint(nn, data)
    ttl_loss.append([loss_in, loss_out])
    ttl_param.append(param)
    for epoch in range(num_epoch):

        for batch in range(num_batch):
            idx_start = batch*num_batch
            idx_end = idx_start + size_batch
            x = x_train[idx_start:idx_end]
            y = y_train[idx_start:idx_end]
            
            # Forward pass
            nn.forward(x, training=True)
            
            # BackWard pass
            nn.backward(y)

        if epoch % rec_epoch == 0:        # compute the loss
            loss_in, loss_out, param = get_checkpoint(nn, data)
            ttl_loss.append([loss_in, loss_out])
            ttl_param.append(param)
            print(f'Epoch {epoch}/{num_epoch} loss_in: {loss_in}, loss_out: {loss_out}')
    
    return np.array(ttl_loss), ttl_param

def get_checkpoint(nn, data):
    """
    Var:
        nn: shallow_nn
    """
    if DEBUGGER:
        print("enter get_checkpoint")
    
    x_train, y_train, x_test, y_test = data
    loss_in = nn.evaluate(x_train, y_train)
    loss_out = nn.evaluate(x_test, y_test)
    param = nn.get_layer()
    
    if DEBUGGER:
        print("exit get_checkpoint")
    return loss_in, loss_out, param
