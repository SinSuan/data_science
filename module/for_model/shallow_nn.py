import os
import numpy as np

DEBUGGER = os.getenv("DEBUGGER")

class shallow_nn:
    """ a shallow neural network
    
    model structure
            dim
            input = 2
            hidden layer = n
            output = 1
    
    "div" stand for "derivative"，因為我拼錯了
    """    

    def __init__(self, n, layer_initializer) -> None:
        if DEBUGGER=="True":
            print("enter __init__")

        # fool-proof
        self.training = False
        
        # number of node
        self.num_input = 2
        self.num_hidden = n
        self.num_output = 1
        
        # activation function
        self.param_hidden_act = 1
        self.param_output_act = 1
        
        # hyper param
        self.learning_rate = 1e-5
        
        # init layer
        det = type(layer_initializer)
        if det is int:
            s = layer_initializer
            np.random.seed(s) # for reproducibility
            self.weights_input_hidden = np.random.uniform(size=(self.num_input, self.num_hidden))
            self.bias_hidden = np.random.uniform(size=(1, self.num_hidden))
            self.weights_hidden_output = np.random.uniform(size=(self.num_hidden, self.num_output))
            self.bias_output = np.random.uniform(size=(1, self.num_output))
            
        elif det is list:
            ttl_layer = layer_initializer
            self.weights_input_hidden = ttl_layer[0]
            self.bias_hidden = ttl_layer[1]
            self.weights_hidden_output = ttl_layer[2]
            self.bias_output = ttl_layer[3]
        else:
            raise("missing for layer_initializer")

        if DEBUGGER=="True":
            print("exit __init__")
  
    def get_layer(self):
        ttl_layer = []
        ttl_layer.append(self.weights_input_hidden.copy())
        ttl_layer.append(self.bias_hidden.copy())
        ttl_layer.append(self.weights_hidden_output.copy())
        ttl_layer.append(self.bias_output.copy())
        return ttl_layer
  
    def act_hidden(self,x):
        """ tanh
        """
        if DEBUGGER=="True":
            print("enter act_hidden")

        exponient = np.e**(2*self.param_hidden_act*x)
        y = (exponient-1)/(exponient+1)

        if DEBUGGER=="True":
            print("exit act_hidden")
        return y
    
    def div_act_hidden(self,x):
        """ tanh
        """
        if DEBUGGER=="True":
            print("enter div_act_hidden")

        exponient = np.e**(self.param_hidden_act*x)
        exponient_minus = np.e**(-self.param_hidden_act*x)
        y = self.param_hidden_act*4/(exponient+exponient_minus)

        if DEBUGGER=="True":
            print("exit div_act_hidden")
        return y
    
    def act_output(self, x):
        """ linear
        """
        if DEBUGGER=="True":
            print("enter act_output")

        y = self.param_hidden_act*x

        if DEBUGGER=="True":
            print("exit act_output")
        return y
    
    def div_act_output(self, x):
        """ linear
        """
        if DEBUGGER=="True":
            print("enter div_act_output")

        y = self.param_hidden_act

        if DEBUGGER=="True":
            print("exit div_act_output")
        return np.array([y])
    
    def forward(self, X, training=False):
        """
        Var:
            training: bool
                True    backward later
                False   forward only
        """
        if DEBUGGER=="True":
            print(f"enter forward, training={training}")

        self.training = training
        if self.training is True:
            self.data = X
        else:
            self.data = None
        
        self.hidden_layer_in = np.dot(X, self.weights_input_hidden) + self.bias_hidden
        self.hidden_layer_out = self.act_hidden(self.hidden_layer_in)
        self.output_layer_in = np.dot(self.hidden_layer_out, self.weights_hidden_output) + self.bias_output
        self.output_layer_out = self.act_output(self.output_layer_in)
        
        pred = self.output_layer_out
        
        if DEBUGGER=="True":
            print(f"exit forward, training={training}")
        return pred
    
    def backward(self, target):
        if DEBUGGER=="True":
            print("enter backward")
            
        if self.training is False:
            if DEBUGGER=="True":
                print("\tif self.training is False")

            raise("self.training is False")
        else:
            if DEBUGGER=="True":
                print("\tif self.training is True")

            error_output = target - self.output_layer_out
            back_output = error_output * (-1) * self.div_act_output(self.output_layer_in)
            grad_output = self.hidden_layer_out.T.dot(back_output)

            back_hidden = back_output.dot(self.weights_hidden_output.T) * self.div_act_hidden(self.hidden_layer_in)
            grad_hidden = self.data.T.dot(back_hidden)
            
            # update weight and bias
            self.weights_hidden_output -= grad_output * self.learning_rate
            self.bias_output -= np.sum(back_output, axis=0, keepdims=True) * self.learning_rate
            self.weights_input_hidden -= grad_hidden * self.learning_rate
            self.bias_hidden -= np.sum(back_hidden, axis=0, keepdims=True) * self.learning_rate

            # train 過後就關起來
            self.training = False
    
        if DEBUGGER=="True":
            print("exit backward")

    def evaluate(self, X, target):
        if DEBUGGER=="True":
            print("enter evaluate")
        
        pred = self.forward(X)
        loss = 0.5*np.mean(np.square(target-pred))

        if DEBUGGER=="True":
            print("exit evaluate")
        return loss
    
    def predict(self, X):
        if DEBUGGER=="True":
            print("enter predict")

        pred = self.forward(X)

        if DEBUGGER=="True":
            print("exit predict")
        return pred
