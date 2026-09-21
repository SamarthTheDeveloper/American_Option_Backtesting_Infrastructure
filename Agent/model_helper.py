import jax
import jax.numpy as jnp
from flax import nnx
import optax
from jax import nn


class Scaling(nnx.Module):
  def __init__(self, D, m,M, rngs: nnx.Rngs):
    self.linear_in = nnx.Linear(D//2 + m, M, rngs=rngs)
    self.linear_mid = nnx.Linear(M, M, rngs=rngs)
    self.linear_out = nnx.Linear(M, D//2, rngs=rngs,kernel_init=nnx.initializers.zeros_init(),)
    self.log_scale_cap = 3

  def __call__(self, x):
    x = nnx.leaky_relu(self.linear_in(x))
    x = nnx.leaky_relu(self.linear_mid(x))
    return self.log_scale_cap * jnp.tanh(self.linear_out(x))

class Transition(nnx.Module):
  def __init__(self, D, m, M, rngs: nnx.Rngs):
    self.linear_in = nnx.Linear(D // 2 + m, M, rngs=rngs)
    self.linear_mid = nnx.Linear(M, M, rngs=rngs)
    self.linear_out = nnx.Linear(M, D // 2, rngs=rngs,kernel_init=nnx.initializers.zeros_init(),)

  def __call__(self, x):
    x = nnx.leaky_relu(self.linear_in(x))
    x = nnx.leaky_relu(self.linear_mid(x))
    return self.linear_out(x)