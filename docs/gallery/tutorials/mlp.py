import alpa
import jax
import jax.numpy as jnp
from jax import random
import numpy as np


class MLPModel:
    num_layers = 0
    hidden_dim = 0
    weights = []
    rngkey = None
    lr = 0

    def __init__(self, layers, dim, lr):
        self.num_layers = layers
        self.hidden_dim = dim
        self.lr = lr

        self.rngkey = jax.random.PRNGKey(0)
        self.weights = [random.normal(self.rngkey, (dim, dim)) for i in range(layers)]
    
    def apply(self, x):
        for i in range(self.num_layers):
            x = jnp.dot(x, self.weights[i])
        
        return x
    
    def update(self, grads):
        for i in range(self.num_layers):
            self.weights[i] -= self.lr * grads[i]
    


dim = 2048
batch_size = 2048
num_layers = 10
lr = 0.1

model = MLPModel(num_layers, dim, lr)

rngkey = jax.random.PRNGKey(0)
k1, k2 = random.split(rngkey)
W = random.normal(k1, (dim, dim))
b = random.normal(k2, (dim,))

ksample, knoise = random.split(k1)
x = random.normal(ksample, (batch_size, dim))
y = (x @ W + b) + 0.1 * random.normal(knoise, (batch_size, dim))

def train_step(x, y):
    def loss_func(x_one, y_one):
        out = model.apply(x_one)
        loss = jnp.mean((out - y_one)**2)
        return loss

    grads = jax.grad(loss_func)(x, y)
    model.update(grads)
    return model.weights


@alpa.parallelize
def alpa_train_step(x, y):
    def loss_func(x_one, y_one):
        out = model.apply(x_one)
        loss = jnp.mean((out - y_one)**2)
        return loss

    grads = jax.grad(loss_func)(x, y)
    model.update(grads)
    return model.weights

alpa_train_step(x, y)