#!/usr/bin/env python
# coding: utf-8

# In[1]:


import gpjax as gpx
import jax
import jax.numpy as jnp
import jax.random as jr
from jax.config import config
import matplotlib.pyplot as plt
import distrax as dx

key = jr.PRNGKey(123)
config.update("jax_enable_x64", True)


# In[2]:


n_data = 100
n_realisations = 5
noise_limits = (0.3, 0.5)
xlims = (-5, 5)
jitter = 1e-6
n_inducing = 20

true_kernel = gpx.kernels.Matern32()
true_params = gpx.initialise(true_kernel, key)

X = jnp.linspace(*xlims, n_data).reshape(-1, 1)
true_kxx = true_kernel.gram(true_kernel, true_params.params, X) + jnp.eye(n_data) * jitter
true_L = true_kxx.triangular_lower()
latent_dist = dx.MultivariateNormalTri(jnp.zeros(n_data), true_L)
group_y = latent_dist.sample(seed=key, sample_shape=(1,)).T


noise_terms = dx.Uniform(*noise_limits).sample(seed= key, sample_shape=(n_realisations, ))

# def add_sig(i):
#     X = jnp.linspace(*xlims, n_data).reshape(-1, 1)
#     group_y = tfp.distributions.MultivariateNormalTriL(np.zeros(n_data), tf.linalg.cholesky(Kxx)).sample(seed=tfp_seed + 10 * i)
#     sample_y = group_y.numpy()
#     return sample_y

realisations = []
individuals_ys = []
for idx, (noise, skey) in enumerate(zip(noise_terms, jr.split(key, n_realisations))):
    # Split the key
    noise_vector = dx.Normal(0, noise).sample(seed=skey, sample_shape=group_y.shape)
    y = group_y + noise_vector
    individuals_ys.append(y)
    realisations.append(gpx.Dataset(X=X, y=y))
    plt.plot(X, y, color='tab:blue')
plt.plot(X, group_y, color='tab:red')


# In[4]:


inducing_points = jnp.linspace(*xlims, n_inducing).reshape(-1, 1)

individual_priors = [gpx.Prior(kernel = gpx.kernels.RBF()) for _ in range(n_realisations)]
group_prior = gpx.Prior(kernel = gpx.kernels.RBF())
likelihood = gpx.Gaussian(num_datapoints=n_data)


# In[17]:


import typing as tp 
from jaxtyping import Float, Array
from chex import PRNGKey as PRNGKeyType, dataclass
from gpjax.utils import concat_dictionaries
import optax as ox
from itertools import product
from copy import deepcopy


@dataclass
class SHGP:
    individual_priors: tp.List[gpx.Prior]
    group_prior: gpx.Prior
    likelihood: gpx.likelihoods.AbstractLikelihood
    inducing_inputs: Float[Array, "M D"]
    name: str = "Sparse Hierarchical GP"
    diag: tp.Optional[bool] = False
    
    def _initialise_params(self, key: PRNGKeyType) -> tp.Dict:
        params = {}
        params["kernel"] = [p._initialise_params(key)['kernel'] for p in self.individual_priors] + [self.group_prior._initialise_params(key)['kernel']]
        params['mean_function'] = {}
        params = concat_dictionaries(params,             {
                "variational_family": {"inducing_inputs": self.inducing_inputs},
                "likelihood": {
                    "obs_noise": self.likelihood._initialise_params(key)["obs_noise"]
                },
            })
        params = jax.tree_map(lambda x: jnp.atleast_2d(x), params)
        return params
    
    def fit_map(self, data: tp.List[gpx.Dataset], optimiser: ox.GradientTransformation, n_iters: int = 1, compile: bool = False, verbose: bool =True, log_rate: int =10):
        loss_fns = self._build_objective(key, data, negative=True, compile=compile)
        n_losses = jnp.arange(len(loss_fns))
        initial_params = gpx.initialise(self, key)
        parameters, _, bijectors = initial_params.unpack()

        
        def objective(params: tp.Dict):
            # Evaluate each loss function in the loss_fns list with the params variable and sum the result
            # return jax.tree_util.tree_reduce(lambda x, y: x + y, [loss_fn(params) for loss_fn in loss_fns])
            vmap_fn = jax.vmap(lambda i, x: jax.lax.switch(i, loss_fns, x))
            return jnp.sum(vmap_fn(n_losses, params))
        
        parameters = gpx.unconstrain(parameters, bijectors)
        dict_to_array, array_to_dict = gpx.utils.dict_array_coercion(parameters)
        parameters = dict_to_array(parameters)

        opt_state = optimiser.init(parameters)
        iter_nums = jnp.arange(n_iters)

        # Optimisation step
        def step(carry, iter_num: int):
            parameters, opt_state = carry
            loss_val, loss_gradient = jax.value_and_grad(objective)(parameters)
            print(loss_gradient)
            updates, opt_state = optimiser.update(loss_gradient, opt_state, parameters)
            parameters = ox.apply_updates(parameters, updates)
            carry = parameters, opt_state
            return carry, loss_val


        if verbose:
            step = gpx.abstractions.progress_bar_scan(n_iters, log_rate)(step)

        # Run the optimisation loop
        (parameters, _), history = jax.lax.scan(step, (parameters, opt_state), iter_nums)

        # Tranform final params to constrained space
        parameters = gpx.constrain(parameters, bijectors)
        return gpx.abstractions.InferenceState(params=parameters, history=history)
    
        
    def fit(self, data: tp.List[gpx.Dataset], optimiser: ox.GradientTransformation, n_iters: int = 1, compile: bool = False, verbose: bool =True, log_rate: int =10):
        loss_fns = self._build_objective(key, data, negative=True, compile=compile)
        initial_params = gpx.initialise(self, key)
        parameters, _, bijectors = initial_params.unpack()
        
        @jax.jit
        def objective(params: tp.Dict):
            return jnp.sum(jax.Array([l(params) for l in loss_fns]))
        
        parameters = gpx.unconstrain(parameters, bijectors)
        
        opt_state = optimiser.init(parameters)
        iter_nums = jnp.arange(n_iters)
        
        # Optimisation step
        def step(carry, iter_num: int):
            parameters, opt_state = carry
            loss_val, loss_gradient = jax.value_and_grad(objective)(parameters)
            updates, opt_state = optimiser.update(loss_gradient, opt_state, parameters)
            parameters = ox.apply_updates(parameters, updates)
            carry = parameters, opt_state
            return carry, loss_val


        if verbose:
            step = gpx.abstractions.progress_bar_scan(n_iters, log_rate)(step)

        # Run the optimisation loop
        (parameters, _), history = jax.lax.scan(step, (parameters, opt_state), iter_nums)

        # Tranform final params to constrained space
        parameters = gpx.constrain(parameters, bijectors)
        return gpx.abstractions.InferenceState(params=parameters, history=history)
    
    def _build_objective(self, key:PRNGKeyType, datasets: tp.List[gpx.Dataset], negative: bool, compile: bool) -> tp.Callable:
        n_realisations = len(datasets)
        idxs = list(product(range(n_realisations), range(n_realisations)))
        losses = []
        params = []

        param_state = self._initialise_params(key)
        
        for idx in idxs:
            # For diagonal entries the group prior is summed with the relevant individual prior
            kernel_list = [gpx.kernels.RBF()] * (n_realisations+1)
            # if idx[0] == idx[1]:
            #     # Due to the parameter's structure, it is important that the group kernel is last.
            #     ik1 = self.individual_priors[idx[0]].kernel
            #     gk = self.group_prior.kernel
            #     kernel = ik1 + gk
            # else:
            #     kernel = self.group_prior.kernel
            kernel_list[idx[0]] = self.individual_priors[idx[0]].kernel
            if idx[0] == idx[1]:
                # Due to the parameter's structure, it is important that the group kernel is last.
                kernel_list[-1] = self.group_prior.kernel                
            
            kernel = kernel_list[0]
            for k in kernel_list[1:]:
                kernel += k

            prior = gpx.Prior(kernel=kernel)
            posterior = prior * self.likelihood
            q = gpx.CollapsedVariationalGaussian(prior=prior, likelihood=self.likelihood, inducing_inputs=self.inducing_inputs)
            sgpr = gpx.CollapsedVI(posterior=posterior, variational_family=q)
            
            param_state = gpx.initialise(sgpr, key)
            
            D = datasets[idx[0]]
            temp_param_copy = deepcopy(param_state)
            if idx[0] == idx[1]:
                for temp_idx, kernel_term in enumerate(temp_param_copy.trainables['kernel'][:-1]):
                    if temp_idx != idx[0]:
                        for kernel_parameter_status, _ in kernel_term.items():
                            kernel_term[kernel_parameter_status] = False
                        temp_param_copy.params['kernel'][temp_idx]['variance'] = jax.Array([0.])
            else:
                for temp_idx, kernel_term in enumerate(temp_param_copy.trainables['kernel']):
                    if temp_idx != idx[0]:
                        for kernel_parameter_status, _ in kernel_term.items():
                            kernel_term[kernel_parameter_status] = False 
                        temp_param_copy.params['kernel'][temp_idx]['variance'] = jax.Array([0.])
                        
            param_state.trainables = temp_param_copy.trainables
            dict_to_array, array_to_dict = gpx.utils.dict_array_coercion(param_state.params)

            def loss_fn(params: tp.Dict) -> Float[Array, ""]:
                _, trainables, _ = param_state.unpack()
                params = array_to_dict(params)
                params = gpx.parameters.trainable_params(params, trainables)
                return sgpr.elbo(D, negative=negative)(params)
            if compile:
                loss_fn = jax.jit(loss_fn)
            losses.append(loss_fn)
        return losses
shgp = SHGP(individual_priors=individual_priors, group_prior=group_prior, likelihood=likelihood, inducing_inputs=inducing_points)


# In[18]:


loss_fns = shgp._build_objective(key, realisations, negative=True, compile=False)


# In[20]:


param_state = gpx.initialise(shgp, key)
loss_fns[1](param_state.params)


# In[10]:





# In[6]:


get_ipython().run_line_magic('time', 'shgp.fit_map(realisations, optimiser=ox.adam(learning_rate=0.01), compile=False)')


# In[ ]:


get_ipython().run_line_magic('time', 'shgp.fit(realisations, optimiser=ox.adam(learning_rate=0.01), compile=True)')


# In[ ]:


losses = shgp.fit(realisations, optimiser=ox.adam(learning_rate=0.01), n_iters=1, compile=False)


# In[ ]:


params = gpx.initialise(shgp, key)
jnp.sum(jax.Array([l(params) for l in loss_fn]))


# In[ ]:


wparams = gpx.kernels.RBF()._initialise_params(key)
wparams['variance'] = jax.Array([0.])
gpx.kernels.RBF().gram(gpx.kernels.RBF(), wparams, X)


# In[ ]:


def _subset_param_state(params: gpx.parameters.ParameterState, idx: int, group: bool):
    # Remove all the parameters except for the one at idx
    params.params['individuals'] = params.params['individuals'][idx]


# In[ ]:


losses, params = shgp.build_objective(datasets=realisations, negative=True)


# In[ ]:


gpx.initialise(shgp, key).trainables


# In[ ]:


shgp._initialise_params(key)


# In[ ]:


opt = ox.adam(0.01)
shgp.fit(datasets=realisations, key=key, opt=opt).params


# In[ ]:


qs = []

for p in individual_priors:
    q = gpx.CollapsedVariationalGaussian(prior=p, likelihood=likelihood, inducing_inputs=inducing_points)
    qs.append(q)


# In[ ]:





# In[ ]:




