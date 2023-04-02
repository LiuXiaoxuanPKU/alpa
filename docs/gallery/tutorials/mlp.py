import alpa
from alpa.testing import assert_allclose

import jax
import jax.numpy as jnp
from jax import random
import numpy as np
from flax import struct
from typing import Callable, List, Any

class MLPTrainState(struct.PyTreeNode):
    step : int
    apply_fn : Callable = struct.field(pytree_node=False)
    params : List[Any] = struct.field(pytree_node=True)
    lr : float

    def apply_gradients(self, grads):
        new_params = []
        for i in range(len(self.params)):
            new_params.append(params[i] - lr * grads[i])
        return MLPTrainState.create(
            step = self.step + 1,
            apply_fn = self.apply_fn,
            params = new_params
        )

    @classmethod
    def create(cls, step, apply_fn, params):
        return cls(step=step, apply_fn=apply_fn, params=params, lr=0.01)

class MLPModel:
    num_layers = 0
    hidden_dim = 0
    
    def __init__(self, num_layers, hidden_dim):
        self.num_layers, self.hidden_dim = num_layers, hidden_dim

    # get initial params  
    def init(self, rngkey, x):
        return [random.normal(rngkey, (self.hidden_dim, self.hidden_dim)) for i in range(self.num_layers)]
    
    def apply(self, params, x):
        for i in range(len(params)):
            x = jnp.dot(x, params[i])
        
        return x


dim = 2048
batch_size = 2048
num_layers = 2
lr = 0.1

rngkey = jax.random.PRNGKey(0)
k1, k2 = random.split(rngkey)
W = random.normal(k1, (dim, dim))
b = random.normal(k2, (dim,))

ksample, knoise = random.split(k1)
x = random.normal(ksample, (batch_size, dim))
y = (x @ W + b) + 0.1 * random.normal(knoise, (batch_size, dim))

model = MLPModel(num_layers, dim)
params = model.init(rngkey, x)
state = MLPTrainState.create(step=0, apply_fn=model.apply, params=params)

@jax.jit
def serial_train_step(state, batch):
    def loss_func(params):
        out = state.apply_fn(params, batch["x"])
        loss = jnp.mean((out - batch["y"])**2)
        return loss

    grads = jax.grad(loss_func)(state.params)
    new_state = state.apply_gradients(grads=grads)
    return new_state

batch = {"x": x, "y": y}
expected_state = serial_train_step(state, batch)


@alpa.parallelize
def alpa_train_step(state, batch):
    def loss_func(params):
        out = state.apply_fn(params, batch["x"])
        loss = jnp.mean((out - batch["y"])**2)
        return loss

    grads = jax.grad(loss_func)(state.params)
    new_state = state.apply_gradients(grads=grads)
    return new_state

# Test correctness
actual_state = alpa_train_step(state, batch)
assert_allclose(expected_state.params, actual_state.params, atol=5e-3)


def sync_func():
    jax.local_devices()[0].synchronize_all_activity()

# # Speed Comparison
# from alpa.util import benchmark_func
# costs = benchmark_func(serial_exec, sync_func, warmup=5, number=10, repeat=5) * 1e3
# print(f"Serial execution time. Mean: {np.mean(costs):.2f} ms, Std: {np.std(costs):.2f} ms")

# costs = benchmark_func(alpa_ckmt_exet, sync_func, warmup=5, number=10, repeat=5) * 1e3
# print(f"Alpa Ckmt execution time. Mean: {np.mean(costs):.2f} ms, Std: {np.std(costs):.2f} ms")

# # Memory Comparison
# GB = 1024 ** 3

# executable = serial_exec.lower().compile().runtime_executable()
# print(f"Serial execution per GPU memory usage: {executable.total_allocation_size() / GB:.2f} GB")

# alpa_executable = alpa_ckmt_exet.get_executable()
# print(f"Alpa execution per GPU memory usage:   {alpa_executable.get_total_allocation_size() / GB:.2f} GB")
