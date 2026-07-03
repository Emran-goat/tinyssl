import sys
from pathlib import Path

import pytest
import torch

# Ensure tinyssl is importable from the project root
_root = str(Path(__file__).resolve().parent.parent.parent)
if _root not in sys.path:
    sys.path.insert(0, _root)


@pytest.fixture(scope="session")
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture(scope="session")
def dummy_images(device):
    """Batch of 4 random 224x224 images."""
    return torch.randn(4, 3, 224, 224, device=device)
