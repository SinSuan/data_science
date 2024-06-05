import os
from math import ceil
from module.for_model.shallow_nn import shallow_nn

DEBUGGER = os.getenv("DEBUGGER")

def construct_and_train_model(
    n,
    layer_initializer,
    data,
    epoch_cycle,
    record_check="epoch"
                              ):
    """ construct a model and train it
    Var:

        nn: shallow_nn

        epoch_cycle: int
            record cycle
    """
    if DEBUGGER=="True":
        print("enter construct_and_train_model")
    nn = \
        shallow_nn(
        n = n,
        layer_initializer = layer_initializer
    )
    ttl_loss, ttl_param = \
        train(
        nn = nn,
        data = data,
        epoch_cycle = epoch_cycle,
        num_epoch = 1000,
        size_batch = 100,
        record_check = record_check
    )
    if DEBUGGER=="True":
        print("exit construct_and_train_model")
    return ttl_loss, ttl_param

def train(nn, data, epoch_cycle, num_epoch, size_batch):
    """
    Var:

        nn: shallow_nn

        epoch_cycle: int
            record cycle
    """
    if DEBUGGER=="True":
        print("enter train")

    x_train, y_train, _, _ = data
    ttl_loss = []
    ttl_param = []
    # num_batch = len(x_train)//size_batch + 1
    num_batch = ceil(len(x_train)//size_batch)

    loss_in, loss_out, param = get_checkpoint(nn, data)
    ttl_loss.append([loss_in, loss_out])
    ttl_param.append(param)
    for epoch in range(num_epoch):

        for batch in range(num_batch):
            idx_start = batch*size_batch
            idx_end = idx_start + size_batch
            x = x_train[idx_start:idx_end]
            y = y_train[idx_start:idx_end]
            
            # Forward pass
            nn.forward(x, training=True)
            
            # BackWard pass
            nn.backward(y)

        if epoch % epoch_cycle == 0:
            # compute the loss
            loss_in, loss_out, param = get_checkpoint(nn, data)
            ttl_loss.append([loss_in, loss_out])
            ttl_param.append(param)
            # print(f'Epoch {epoch}/{num_epoch}\tloss_in: {loss_in},\tloss_out: {loss_out}')
    
    if DEBUGGER=="True":
        print("exit train")
    return ttl_loss, ttl_param

def get_checkpoint(nn, data):
    """
    Var:
        nn: shallow_nn
    """
    if DEBUGGER=="True":
        print("enter get_checkpoint")
    
    x_train, y_train, x_test, y_test = data
    loss_in = nn.evaluate(x_train, y_train)
    loss_out = nn.evaluate(x_test, y_test)
    param = nn.get_layer()
    
    if DEBUGGER=="True":
        print("exit get_checkpoint")
    return loss_in, loss_out, param
