from typing import Callable, Iterable, Tuple

import torch
from torch.optim import Optimizer


class AdamW(Optimizer):
    def __init__(
            self,
            params: Iterable[torch.nn.parameter.Parameter],
            lr: float = 1e-3,
            betas: Tuple[float, float] = (0.9, 0.999),
            eps: float = 1e-6,
            weight_decay: float = 0.0,
            correct_bias: bool = True,
            max_grad_norm: float = None,
    ):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[1]))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {} - should be >= 0.0".format(eps))
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, correct_bias=correct_bias, max_grad_norm=max_grad_norm)
        super().__init__(params, defaults)

    def step(self, closure: Callable = None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:

            # TODO: Clip gradients if max_grad_norm is set
            if group['max_grad_norm'] is not None:
                params_with_grad = [p for p in group["params"] if p.grad is not None]
                grads = [p.grad for p in params_with_grad]
                torch.nn.utils.clip_grad_norm_(params_with_grad, group['max_grad_norm'])

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")

                # State should be stored in this dictionary
                state = self.state[p]
                if len(state) == 0:
                    state["m"] = torch.zeros_like(p.data)
                    state["v"] = torch.zeros_like(p.data)
                    state["t"] = 0

                state["t"] += 1

                # TODO: Access hyperparameters from the `group` dictionary
                alpha = group["lr"]
                t = state["t"]
                eps, weight_decay = group["eps"], group["weight_decay"]
                beta1, beta2 = group["betas"][0], group["betas"][1]

                # TODO: Update first and second moments of the gradients
                state["m"] = beta1 * state["m"] + (1 - beta1) * grad
                state["v"] = beta2 * state["v"] + (1 - beta2) * (grad ** 2)

                # TODO: Bias correction
                # Please note that we are using the "efficient version" given in Algorithm 2
                # https://arxiv.org/pdf/1711.05101
                step_size = alpha * ((1 - beta2 ** t) ** 0.5) / (1 - beta1 ** t)

                # TODO: Update parameters
                p.data.addcdiv_(state["m"], state["v"].sqrt().add_(eps) ,value=-step_size)

                # TODO: Add weight decay after the main gradient-based updates.
                # Please note that the learning rate should be incorporated into this update.
                if group.get("weight_decay", 0) != 0:
                    p.data.add_(p.data, alpha=-alpha * weight_decay)

        return loss
