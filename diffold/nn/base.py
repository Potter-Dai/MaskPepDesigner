import torch
from torch import nn
from typing import List, Optional, Union

import logging
logger = logging.getLogger(__name__)

class RobustModule(nn.Module):
    def __init__(
        self,
        enable_nan_check: bool = False,
        tolerance: float = 1e3,
        strict: bool = False
    ):
        super().__init__()
        self.enable_nan_check = enable_nan_check
        self.tolerance = tolerance
        self.strict = strict
    
    @staticmethod
    def _display_warning(tensors: List[torch.Tensor], key: str):
        t: torch.Tensor
        if tensors is None:
            return
        for i, t in enumerate(tensors):   # todo use logging instead
            logger.warn(f"{key}_{i}: min={t.min():.2f} max={t.max():.2f}.")
    
    @staticmethod
    def _convert_to_lists(*args) -> tuple:
        return (
            v if type(v) in (tuple, list) else [v] for v in args
        )
    
    def check_nan(
        self,
        outputs: List[torch.Tensor],
        inputs: Optional[List[torch.Tensor]] = None,
        key: str = '',
        enable: Optional[bool] = None,
        tolerance: Optional[float] = None,
        strict: Optional[bool] = None,
    ):
        enable = enable if enable is not None else self.enable_nan_check
        if not enable:
            return
        tolerance = tolerance or self.tolerance
        strict = strict if strict is not None else self.strict
        outputs, inputs = self._convert_to_lists(outputs, inputs)
        for outv in outputs:
            if torch.any(outv.isnan()) or torch.any(outv.abs() > tolerance):
                logger.warn(f"nan detected in {key} (tol.={tolerance})")
                self._display_warning(outputs, "output")
                if inputs is not None:
                    self._display_warning(inputs, "input")
                if strict:
                    raise ValueError(f"nan detected in {key} (tol.={tolerance})")
