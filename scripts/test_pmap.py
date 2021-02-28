from functools import partial 
import jax
import jax.numpy as jnp


def debug_pmap():
    #f = lambda x: jax.lax.psum(x, axis_name='batch')
    #y = jax.pmap(f, axis_name='batch')(jnp.ones((4, 4)))
    #print(y, type(y))

    @jax.pmap
    def func(x, w):
        print("pmap A")
        return x @ w

    print("pmap B")
    y = func(jnp.ones((1, 4)), jnp.ones((1, 4)))
    print("pmap C")
    print(y, type(y))


def test_nested_pmap():
    @partial(jax.pmap, axis_name='a0', in_axes=(0, None), out_axes=0)
    def add(a, b):
        # a.shape = (32, 64)
        # b.shape = (64, 2, 32)
        @partial(jax.pmap, axis_name='a1', in_axes=(None, 1), out_axes=1)
        def add_inner(x, y):
            # x.shape = (32, 64)
            # y.shape = (64, 32)
            return x @ y

        # ret.shape = (32, 2, 32)
        ret = add_inner(a, b)
        return ret

    a = jnp.ones((2, 32, 64))
    b = jnp.ones((64, 2, 32))

    #jaxpr = jax.make_jaxpr(add)(a, b)
    #print(jaxpr)
    #print(jaxpr.jaxpr.outvars[0].aval.shape)

    c = add(a, b)
    print(c)


if __name__ == "__main__":
    test_nested_pmap()

