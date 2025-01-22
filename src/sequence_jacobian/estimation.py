"""Functions for calculating the log likelihood of a model from its impulse responses"""

import numpy as np
import scipy.linalg as linalg
from numba import njit
import warnings

from .blocks.combined_block import create_model

'''Part 1: compute covariances at all lags and log likelihood'''


def all_covariances(M, sigmas):
    """Use Fast Fourier Transform to compute covariance function between O vars up to T-1 lags.

    See equation (108) in appendix B.5 of paper for details.

    Parameters
    ----------
    M      : array (T*O*Z), stacked impulse responses of nO variables to nZ shocks (MA(T-1) representation)
    sigmas : array (Z), standard deviations of shocks

    Returns
    ----------
    Sigma : array (T*O*O), covariance function between O variables for 0, ..., T-1 lags
    """
    T = M.shape[0]
    dft = np.fft.rfftn(M, s=(2 * T - 2,), axes=(0,))
    total = (dft.conjugate() * sigmas**2) @ dft.swapaxes(1, 2)
    return np.fft.irfftn(total, s=(2 * T - 2,), axes=(0,))[:T]


def gaussian_log_likelihood(Y, Sigma, sigma_measurement=None):
    """Given second moments, compute log-likelihood of data Y.

    Parameters
    ----------
    Y       : array (Tobs*O)
                stacked data for O observables over Tobs periods
    Sigma   : array (T*O*O)
                covariance between observables in model for 0, ... , T lags (e.g. from all_covariances)
    sigma_measurement : [optional] array (O)
                            std of measurement error for each observable, assumed zero if not provided

    Returns
    ----------
    L : scalar, log-likelihood
    """
    Tobs, nO = Y.shape
    if sigma_measurement is None:
        sigma_measurement = np.zeros(nO)
    V = build_full_covariance_matrix(Sigma, sigma_measurement, Tobs)
    y = Y.ravel()
    return log_likelihood_formula(y, V)

def log_likelihood(data, shocks, jacobian, outputs, exogenous, T, **kwargs):
    """Given the Jacobian of a parameterized model, calculate the Gaussian likelihood for a provided sequence
       of shocks.
    """
    # construct impulse response functions
    impulses = shocks.generate_impulses(T)
    irfs = {i: jacobian @ {i: impulses[i]} for i in exogenous}

    M = np.empty((T, len(outputs), len(exogenous)))
    for no, o in enumerate(outputs):
        for ns, s in enumerate(exogenous):
            M[:, no, ns] = irfs[s][o]

    # approximate the autocovariances
    Sigma = all_covariances(M, 1)

    # compute the log posterior likelihood
    return gaussian_log_likelihood(data, Sigma, **kwargs)


'''Part 2: helper functions'''


def log_likelihood_formula(y, V):
    """Implements multivariate normal log-likelihood formula using Cholesky with data vector y and variance V.
       Calculates -log det(V)/2 - y'V^(-1)y/2
    """
    V_factored = linalg.cho_factor(V)
    quadratic_form = np.dot(y, linalg.cho_solve(V_factored, y))
    log_determinant = 2*np.sum(np.log(np.diag(V_factored[0])))
    return -(log_determinant + quadratic_form) / 2


@njit
def build_full_covariance_matrix(Sigma, sigma_measurement, Tobs):
    """Takes in T*O*O array Sigma with covariances at each lag t,
    assembles them into (Tobs*O)*(Tobs*O) matrix of covariances, including measurement errors.
    """
    T, O, O = Sigma.shape
    V = np.empty((Tobs, O, Tobs, O))
    for t1 in range(Tobs):
        for t2 in range(Tobs):
            if abs(t1-t2) >= T:
                V[t1, :, t2, :] = np.zeros((O, O))
            else:
                if t1 < t2:
                    V[t1, : , t2, :] = Sigma[t2-t1, :, :]
                elif t1 > t2:
                    V[t1, : , t2, :] = Sigma[t1-t2, :, :].T
                else:
                    # want exactly symmetric
                    V[t1, :, t2, :] = (np.diag(sigma_measurement**2) + (Sigma[0, :, :]+Sigma[0, :, :].T)/2)
    return V.reshape((Tobs*O, Tobs*O))


try:
    import pytensor
    import pytensor.tensor as pt
    from pytensor.graph import Apply, Op

    # TODO: add dictionary handling into likelihood call
    # TODO: improve shock parameterization
    class DensityModel(Op):
        """
        Operation class for estimating a DSGE model in PyMC; specifically given a
        model object, its steady state, some data, and a likelihood function which
        can be customized to recompute a Jacobian only when necessary.
        """
        def __init__(
                self, data, steady_state, model, shock_func, unknowns, targets, exogenous, T, **kwargs
            ):
            # save important model info
            self.steady_state = steady_state
            self.model = model
            # self.logpdf = likelihood_func
            self.assemble_shocks = shock_func
            self.T = T

            # check that all series are of equal length
            T_data = [len(v) for v in data.values()]
            assert all(x == T_data[0] for x in T_data)

            # munge the data into a numpy array
            self.data = np.empty((T_data[0], len(data.keys())))
            for no, o in enumerate(data.keys()):
                self.data[:, no] = data[o]

            # record the initial Jacobian
            outputs = list(data.keys())
            self.jacobian = model.solve_jacobian(
                steady_state, unknowns, targets, exogenous, outputs, T=T, **kwargs
            )

            # this is certainly sloppy, but it works for the meantime
            self.precomputed = False
            self.model_info = {
                "unknowns": unknowns,
                "targets": targets,
                "inputs": exogenous,
                "outputs": outputs
            }
        
        def construct_jacobian_model(self, inputs):
            inp_names = set([inp.name for inp in inputs])
            self.model = self.reduce_model(inp_names, T=self.T)
            self.precomputed = False
            return None            

        def reduce_model(self, param_names, T):
            model = self.model

            inputs   = model.make_ordered_set(self.model_info["inputs"])
            unknowns = model.make_ordered_set(self.model_info["unknowns"])
            targets  = model.make_ordered_set(self.model_info["targets"])
            outputs  = model.make_ordered_set(self.model_info["outputs"])

            actual_outputs, unknowns_as_outputs = model.process_outputs(
                self.steady_state, unknowns, outputs
            )

            ss = model.M.inv @ self.steady_state
            reqs = model._required
            vector_valued = ss._vector_valued()
            
            inputs  = (model.M.inv @ (inputs | unknowns) | reqs) - vector_valued
            outputs = (model.M.inv @ ((actual_outputs | targets) - unknowns) | reqs) - vector_valued
            
            new_blocks = []
            for block in model.blocks:
                if model.make_ordered_set(block.inputs) & set(param_names):
                    new_blocks.append(block)
                else:
                    new_blocks.append(
                        block.jacobian(ss, inputs & block.inputs, outputs & block.outputs, T=T)
                    )

            return create_model(new_blocks, name="reduced Jacobian")

        def make_node(self, *args) -> Apply:
            inputs = [pt.as_tensor(arg) for arg in args]
            outputs = [pt.dscalar()]

            return Apply(self, inputs, outputs)
        
        def perform(self, node: Apply, inputs: list[np.ndarray], outputs: list[list[None]]) -> None:
            # convert model to jacobian blocks
            self.construct_jacobian_model(node.inputs)

            # TODO: there should be a better way to consolidate parameters
            is_not_shock = {
                inp.name: (inp.name in self.model.inputs) for inp in node.inputs
            }

            shock_params = [i for (i,v) in zip(inputs, is_not_shock.values()) if not v]
            model_params = {
                k: inputs[i] for i,(k,v) in enumerate(is_not_shock.items()) if v
            }

            shocks = self.assemble_shocks(*shock_params)
            logposterior = self.likelihood(
                model_params, shocks
            )

            outputs[0][0] = np.asarray(logposterior)

        def likelihood(self, model_params, shocks):
            outputs, inputs = self.model_info["outputs"], self.model_info["inputs"]
            unknowns, targets = self.model_info["unknowns"], self.model_info["targets"]
            
            # reparameterize the model and recompute the Jacobian
            ss_new = self.steady_state.copy()
            ss_new.update(model_params)
            new_jacobian = self.model.solve_jacobian(
                ss_new, unknowns, targets, inputs, outputs, T=self.T
            )

            return log_likelihood(
                self.data, shocks, new_jacobian, outputs, inputs, T=self.T
            )

except ImportError:
    class DensityModel:
        def __init__(self, *args, **kwargs):
            warnings.warn(
                "Attempted to call DensityModel when PyMC is not yet installed"
            )