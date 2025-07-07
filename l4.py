import torch
import numpy as np
from typing import Callable


class L4EarlyStopping:
    """
    Implements an early stopping mechanism based on the average loss. Due to the L4 optimization is prune to diverge if the hyperparameters are not properly set
    """
    def __init__(self, tau: int = 20, gamma: float = 5):
        self.avg_loss = None
        self.tau = tau
        self.gamma = gamma

    def stop(self, loss) -> bool:

        if self.avg_loss is None:
            self.avg_loss = loss
            return False

        prev = self.avg_loss
        self.avg_loss = self.avg_loss * (1 - 1/self.tau) + loss * 1/self.tau

        return self.avg_loss > prev + (prev * self.gamma)


class L4Optim(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        base: torch.optim.Optimizer,
        alpha=0.15,
        gamma=0.9,
        gamma_zero=0.75,
        tau=1e3,
        epsilon=1e-12,
        beta=0.9,
        method = "dot",
        k=4
    ):
        """
        Initialize an L4 optimizer instance which wraps a base optimizer to provide adaptive learning rate optimization.
        Parameters:
            params (iterable): An iterable of parameters to optimize.
            base (torch.optim.Optimizer): A core optimizer instance to be wrapped by L4 optim; must not be an instance of L4Optim.
            alpha (float, optional): The learning rate scaling factor. Default is 0.15.
            gamma (float, optional): The factor for the min. loss. Default is 0.9.
            gamma_zero (float, optional): The initial min. loss scaling factor. Default is 0.75.
            tau (float, optional): A time constant used for increasing the minimum loss. Default is 1e3.
            epsilon (float, optional): A small constant to avoid division by zero in calculations. Default is 1e-12.
            beta (float, optional): The momentum term factor for the gradient estimation. Default is 0.9.
            method (str, optional): The method used for calculating the denominator; must be either "dot", "cossim", "clamp". Default is "dot".
            k (int, optional): Only used if method is set to cossim, sets the exponent. Default is 4.
        Raises:
            ValueError: If the base optimizer is an instance of L4Optim or if the method is not one of "dot" or "cossim".
        This constructor also initializes the optimizer's state dictionary by storing the hyperparameters,
        tracking the learning rate, step count, and setting up buffers for gradient history and learning rate dynamics.
        """

        if type(base) is L4Optim:
            raise ValueError("base cannot be L4")
        
        if method not in ["dot", "cossim", "clamp"]:
            raise ValueError("method must be 'dot', 'cossim' or 'clamp'")

        super(L4Optim, self).__init__(params, defaults={})

        # since L4 is a wrapper around a core optimizer, we need to store the base optimizer
        self.base = base

        # Save the hyperparameters in the state dictionary so that they are saved in the checkpoint (The dict is initialized in the Optimizer class)
        self.state["alpha"] = alpha
        self.state["gamma"] = gamma
        self.state["gamma_zero"] = gamma_zero
        self.state["tau"] = tau
        self.state["epsilon"] = epsilon
        self.state["beta"] = beta
        self.state["method"] = method
        self.state["k"] = k

        # Initialize the variables
        self.state["lr"] = 0
        self.state["lmin"] = 0
        self.state["denom"] = 0
        self.state["step"] = 0
        self.state["grad_buffer"] = {}

    def step(self, closure: Callable[[], float]) -> float:
        # Get the hyperparameters from the state dictionary
        alpha = self.state["alpha"]
        gamma = self.state["gamma"]
        gamma_zero = self.state["gamma_zero"]
        tau = self.state["tau"]
        epsilon = self.state["epsilon"]

        # Get the loss from the closure
        if closure is None:
            raise ValueError("closure is None")
        
        # Make sure that the closure is executed in a context where the gradients are computed
        closure = torch.enable_grad()(closure)

        # Compute the loss and calculate the gradients
        loss = float(closure())

        cached_params = {}
        cached_grads = {}
        for group in self.base.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                cached_params[p] = p.data.clone().detach()
                cached_grads[p] = p.grad.data.clone().detach()

        # Run the base optimizer step
        self.base.step(closure)

        # Initialize the minimum loss
        if self.state["step"] == 0:
            self.state["lmin"] = loss * gamma_zero


        # Update the step counter
        self.state["step"] += 1

        # Update the minimum loss
        self.state["lmin"] = min(self.state["lmin"], loss)

        denom = 0
        mag_v = 0
        mag_g = 0
        for group in self.base.param_groups:
            p : torch.Tensor
            for p in group["params"]:
                if p.grad is None:
                    continue
                v : torch.Tensor = p.data.sub(cached_params[p])

                # Initialize the estimated gradient
                if p not in self.state["grad_buffer"]:
                    self.state["grad_buffer"][p] = torch.zeros_like(p.data)

                # Update the estimated gradient, do this here before the base optimizer step might alter it
                self.state["grad_buffer"][p] = (self.state["beta"]) * self.state["grad_buffer"][p] + (1 - self.state["beta"]) * cached_grads[p]

                grad : torch.Tensor = self.state["grad_buffer"][p]/(1 - (self.state["beta"])**self.state["step"])

                denom += float(torch.dot(-grad.data.view(-1), v.data.view(-1)))

                if self.state["method"] == "cossim":
                    # We need to compute the magnitudes of v and g for the cosine distance, so lets gather all the scuares
                    mag_v += float(v.data.pow(2).sum())
                    mag_g += float(grad.data.pow(2).sum())
                    

        if self.state["method"] == "cossim":
            # Compute the magnitude
            mag = np.sqrt(mag_v * mag_g)
            # use the cosine similarity as more stable version of the dot product
            denom = mag * (0.5 + denom / (2 * mag + epsilon))**self.state["k"]

        if self.state["method"] == "clamp":
            denom = max(0, denom)

        # Store the denominator in the state dictionary (for outside access)
        self.state["denom"] = denom

        # Compute the learning rate
        lr = alpha * (loss - gamma * self.state["lmin"]) / (denom + epsilon)

        # Update the learning rate
        self.state["lr"] = lr

        if np.isnan(self.state["lr"]):
            raise ValueError("Learning rate is NaN: loss={}, denom={}, lmin={}".format(loss, denom, self.state["lmin"]))

        for group in self.base.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                # Compute the proposed update for the parameter
                v = p.data.sub(cached_params[p])

                p.data.copy_(cached_params[p])

                p.data.add_(v * self.state["lr"])

        self.state["lmin"] = self.state["lmin"] * (1 + 1/tau)

        return loss
    
    def zero_grad(self):
        self.base.zero_grad()
