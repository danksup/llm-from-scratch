import random
import engine.activations as unit
import engine.backend as nx
from typing import Any

class Layer:
    def __init__(self, n_neuron:int, m_weight:int, layer_type:str="hidden",activation=unit.swish, activation_derivative = unit.swish_derivative) -> None:
        '''
        Args:
            n_neuron: number of neurons
            m_weight: number of weights per neuron
            activation: activation unit
            activation_derivative: activation unit derivative
        '''
        self.layer_type = layer_type
        self.n = n_neuron
        self.m = m_weight
        self.activation = activation
        self.activation_derivative = activation_derivative
        scale = nx.float_32(1.0/nx.sqrt(self.m))
        
        self.weights = nx.uniform(low=-scale, high=scale,size=(self.n, self.m), dtype=nx.float16)
        self.biases = nx.uniform(low=-scale, high=scale,size=(self.n,), dtype=nx.float16)
    
    def __repr__(self) -> str:
        return f"Weights: {self.weights}\nBiases: {self.biases}"

    def __len__(self):
        return self.n
    
    @classmethod
    def hidden(cls, n:int, m:int, activation=unit.leaky_relu, activation_derivative= unit.leaky_relu_derivative) -> "Layer":
        return cls(n,m,"hidden",activation ,activation_derivative)
    
    @classmethod
    def output(cls, n:int, m:int) -> "Layer":
        return cls(n,m, "output",None, None)
    
    def forward(self, inputs:Any) -> Any:
        '''
        compute layer output and some values needed for layer.backward()
        '''
        inputs = inputs.astype(nx.float32)
        weights = self.weights
        biases = self.biases
        self.last_input = inputs
        
        if inputs.ndim == 1:
            z = weights @ inputs + biases
        else:
            z = inputs @ weights.T + biases

        self.last_z = z
        res = self.activation(z) if self.activation is not None else z
        self.last_ouput = res

        return res

    def backward(self, err_signal:Any) -> Any:
        '''
        compute gradients and return previous layer error signal
        '''
        if self.activation_derivative is not None:
            current_neuron_error = err_signal * self.activation_derivative(self.last_z.astype(nx.float32))
        else:
            current_neuron_error = err_signal.astype(nx.float32)

        batch_size, seq_len, _ = self.last_input.shape
        X = self.last_input.reshape(-1, self.last_input.shape[-1])
        E = current_neuron_error.reshape(-1, current_neuron_error.shape[-1])
        X32 = X.astype(nx.float32)
        E32 = E.astype(nx.float32)

        # X = np.ascontiguousarray(self.last_input).reshape(-1, self.last_input.shape[-1]) #ai suggested this, because machine loves when the data are contigous so they can just take it without jumping (row since it's based on C)
        # E = np.ascontiguousarray(current_neuron_error).reshape(-1, current_neuron_error.shape[-1])
        self.d_weight = (E32.T @ X32) / (batch_size * seq_len)
        # self.d_weight = np.einsum("bso,bsi->oi",current_neuron_error,self.last_input, dtype=np.float32) / np.float32(batch_size * seq_len)
        previous_error = current_neuron_error.astype(nx.float32) @ self.weights

        self.d_bias = nx.mean(current_neuron_error, axis=(0,1),dtype=nx.float32)

        return previous_error
    
    def to_dict(self) -> dict:
        return {
            "n":self.n,
            "m":self.m,
            "weights":self.weights.tolist(),
            "biases":self.biases.tolist(),
            "type":self.layer_type
        }
    
    @classmethod
    def from_dict(cls,thing:dict) -> "Layer":
        n = thing['n']
        m = thing['m']
        weights = nx.array(thing["weights"], dtype=nx.float16)
        biases = nx.array(thing["biases"],nx.float16)
        layer_type = thing["type"]
        if layer_type == "hidden":
            ff = cls.hidden(n,m)
        elif layer_type == "output":
            ff = cls.output(n,m)
        else:
            raise ValueError(f"unknown type {layer_type}")
        ff.weights = weights
        ff.biases = biases

        return ff
        
        
