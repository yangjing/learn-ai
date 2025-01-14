import torch
from pathlib import Path


DATA_DIR: Path = Path.home() / "data"
DEVICE: torch.device = torch.device(
  "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
)
