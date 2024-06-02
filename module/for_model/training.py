import os
from module.for_model.shallow_nn import shallow_nn

DEBUGGER = os.getenv("DEBUGGER")

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
            print(f'Epoch {epoch}/{num_epoch}\tloss_in: {loss_in},\tloss_out: {loss_out}')
    
    return ttl_loss, ttl_param

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
