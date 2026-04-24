"""Cloud response decoder — unmasks symbols back to the session's real values."""

from __future__ import annotations

from .encoder import MaskMap


class Decoder:
    def __init__(self, mask_map: MaskMap) -> None:
        self.mask_map = mask_map

    def decode(self, session_id: str, cloud_answer: str) -> str:
        return self.mask_map.unmask(session_id, cloud_answer)
