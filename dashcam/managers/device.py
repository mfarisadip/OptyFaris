"""
Device management for CUDA/CPU selection
"""
import torch
from typing import Tuple, Union

class DeviceManager:
    """Manage computation devices"""

    @staticmethod
    def check_cuda():
        """Check CUDA availability"""
        if torch.cuda.is_available():
            return True, torch.cuda.device_count(), torch.cuda.current_device(), torch.cuda.get_device_name()
        return False, 0, None, None

    @staticmethod
    def get_device(requested: str = 'auto', cuda_idx: int = 0) -> Tuple[str, Union[int, str]]:
        """Determine optimal device — cuda → mps (Apple) → cpu"""
        req = requested.lower()
        cuda_available, cuda_count, _, _ = DeviceManager.check_cuda()
        mps = getattr(torch.backends, 'mps', None)
        mps_available = bool(mps and mps.is_available())

        if req == 'cpu':
            return 'cpu', 'cpu'
        if cuda_available and (req == 'cuda' or req == 'auto') and cuda_idx < cuda_count:
            return f'cuda:{cuda_idx}', int(cuda_idx)
        if mps_available:
            return 'mps', 'mps'
        return 'cpu', 'cpu'
