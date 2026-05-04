# This code is part of Qiskit.
#
# (C) Copyright IBM 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""ExecutorQuantumProgramResult."""

from __future__ import annotations

from collections.abc import Iterator
import numpy as np


class ExecutorQuantumProgramResult:
    """A container to store results from executing on a quantum hardware.

    Args:
        data: A list of dictionaries with array-valued data.
        passthrough_data: Dictionary passed through execution without modification.
    """

    def __init__(
        self,
        data: list[dict[str, np.ndarray]],
        passthrough_data: dict | None = None,
    ):
        self._data = data
        self.passthrough_data = passthrough_data

    def __iter__(self) -> Iterator[dict[str, np.ndarray]]:
        yield from self._data

    def __getitem__(self, idx: int) -> dict[str, np.ndarray]:
        return self._data[idx]

    def __len__(self) -> int:
        return len(self._data)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(<{len(self)} results>)"
