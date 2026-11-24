#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations


class InferenceCancelled(Exception):
    """Raised when an inference job is cancelled by the caller."""


class FullSequenceTooLong(ValueError):
    """Raised when full-sequence inference exceeds its configured safety limit."""

    def __init__(self, num_frames: int, max_frames: int):
        self.num_frames = int(num_frames)
        self.max_frames = int(max_frames)
        super().__init__(
            f"Full-sequence inference received {self.num_frames} frames, but the "
            f"configured limit is {self.max_frames}. Use chunked mode or raise the limit."
        )
