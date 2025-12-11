from pydantic import BaseModel
from typing import Literal

class MambaConfig(BaseModel):
    # Always the same
    expand: int = 2
    d_conv: int = 4

    # Can vary
    d_state: Literal[64, 128] = 64
    headdim: Literal[64, 128] = 64
