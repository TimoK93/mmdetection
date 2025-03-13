# Copyright (c) OpenMMLab. All rights reserved.
from contextlib import contextmanager
from typing import Optional, Union

import torch
import torch.nn as nn

# a circular import will be caused by
# from mmengine.model.wrappers import is_model_wrapper
import mmengine
from mmengine.registry import OPTIM_WRAPPERS
from mmengine.optim.optimizer.optimizer_wrapper import OptimWrapper

from HyperSparse.loss.HyperSparse import hyperSparse

try:
    from apex.contrib.sparsity import ASP
except ImportError:
    raise RuntimeError("Failed to import ASP. Please install Apex from https:// github.com/nvidia/apex .")



@OPTIM_WRAPPERS.register_module()
class HypersparseWrapper(OptimWrapper):

    def __init__(self,
                 warmup_steps: int = 100,
                 final_prune_step: int = 1000,
                 prune_rate: float = 0.5,
                 lambda_init: float = 0.0005,
                 eta: float = 1.005,
                 **kwargs):
        assert ASP is not None, \
            'Apex is not installed. Please check ' \
            'https://github.com/NVIDIA/apex#linux.'
        super().__init__(**kwargs)

        # ASP
        self.is_asp_initialized = False
        self.model = None

        self.warmup_steps = warmup_steps
        self.final_prune_step = final_prune_step

        self.prune_rate = prune_rate
        self.lambda_init = lambda_init
        self.eta = eta

    def backward(self, loss: torch.Tensor, **kwargs) -> None:
        """Perform gradient back propagation with :attr:`loss_scaler`.

        Args:
            loss (torch.Tensor): The loss of current iteration.
            kwargs: Keyword arguments passed to :meth:`torch.Tensor.backward`
        """

        if self.final_prune_step > self._inner_count > self.warmup_steps:
            alpha = (self.lambda_init *
                     (self.eta ** float(self._inner_count - self.warmup_steps)))
            hyperloss = hyperSparse(self.model, self.prune_rate) * alpha
        else:
            hyperloss = 0.0

        loss += hyperloss

        loss.backward(**kwargs)

        self._inner_count += 1
        if self._inner_count == self.final_prune_step:
            self.prune()

    def prune(self):
        if not self.is_asp_initialized:
            self.is_asp_initialized = True
            ASP.init_model_for_pruning(
                self.model,
                mask_calculator="m4n2_1d",
                verbosity=2,
                whitelist=[torch.nn.Linear, torch.nn.Conv2d,
                           torch.nn.MultiheadAttention],
                allow_recompute_mask=True,
                allow_permutation=False, # True
            )
            ASP.init_optimizer_for_pruning(self.optimizer)
        ASP.compute_sparse_masks()

    def state_dict(self) -> dict:
        """Get the state dictionary of :attr:`optimizer` and :attr:`apex_amp`.

        Based on the state dictionary of the optimizer, the returned state
        dictionary will add a key named "apex_amp".

        Returns:
            dict: The merged state dict of :attr:`apex_amp` and
            :attr:`optimizer`.
        """
        state_dict = self.optimizer.state_dict()
        return state_dict

    def load_state_dict(self, state_dict: dict) -> None:
        """Load and parse the state dictionary of :attr:`optimizer` and
        :attr:`apex_amp`.

        If state_dict contains "apex_amp", the :attr:`apex_amp` will
        load the corresponding keys. Otherwise, only the :attr:`optimizer`
        will load the state dictionary.

        Note:
            :meth:`load_state_dict` shuold be called after
            `apex_amp.initialize` is called.
        Args:
            state_dict (dict): The state dict of :attr:`optimizer` and
                :attr:`apex_amp`
        """
        self.optimizer.load_state_dict(state_dict)

    @contextmanager
    def optim_context(self, model: nn.Module):
        """Enables the context for mixed precision training, and enables the
        context for disabling gradient synchronization during gradient
        accumulation context.

        Args:
            model (nn.Module): The training model.
        """
        if self.model is None:
            self.model = model

        yield