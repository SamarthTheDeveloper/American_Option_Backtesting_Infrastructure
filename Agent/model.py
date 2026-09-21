import jax
import jax.numpy as jnp
import numpyro.distributions as dist
from flax import nnx
import optax
from model_helper import Scaling, Transition


class NVP_Flow(nnx.Module):
    def __init__(self, num_flows, D, h): # h is the feature vector of the state
        super(NVP_Flow, self).__init__()

        self.prior = dist.MultivariateNormal(loc=jnp.zeros(D), covariance_matrix=jnp.eye(D))

        self.t = nnx.List([
            Transition() for _ in range(num_flows)
        ])

        self.s = nnx.List([
            Scaling() for _ in range(num_flows)
        ])  

        self.num_flows = num_flows

        self.D = D

        self.h = h

        self.__root = jax.random.key(128)
        self.keys = [jax.random.fold_in(self.__root, k) for k in range(num_flows)]

        self.perms = [ jax.random.permutation(key, D) for key in range(self.keys)]
        self.inv_perms = [ jnp.argsort(perm) for perm in range(self.perms) ]

    def coupling(self, x, index, forward=True):

        (xa, xb) = jnp.array_split(x, 2, 1)
        # note ya = xa

        s = self.s[index](jnp.concat((xa,self.h),axis=1))
        t = self.t[index](jnp.concat((xb,self.h),axis=1))

        if not forward:
            yb = jnp.exp(s) * xb + t
        else:
            yb = (xb - t) * jnp.exp(-s) # this seems reversed but the forward pass stacks inverse functions

        return jnp.concat((xa, yb), axis=1), s

    def permute(self, x, i):
        return x[..., self.perms[i]]

    def inverse_permute(self, x, i):
        return x[..., self.inv_perms[i]]

    def forward(self, x):

        log_det_j, z = jnp.zeros(x.shape[0]), x

        for i in range(self.num_flows):

            z, s = self.coupling(z, i, forward=True)

            z = self.permute(z, i)

            log_det_j = log_det_j - s.sum(axis=1)

        return z, log_det_j

    def inv_f(self, z):

        x = z

        for i in reversed(range(self.num_flows)):

            x = self.inverse_permute(x, i)

            x, _ = self.coupling(x, i, forward=False)

        return x

    def __call__(self, x, reduction='avg'):

        z, log_det_j = self.forward(x)

        if reduction == 'sum':
            return -(self.prior.log_prob(z) + log_det_j).sum()
        return -(self.prior.log_prob(z) + log_det_j).mean()

