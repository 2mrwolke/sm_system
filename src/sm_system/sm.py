from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Union

import numpy as np
from numpy.typing import NDArray


Array = NDArray[np.generic]
BranchFn = Callable[[Array], Array]
BranchComponent = Union[BranchFn, Any]


class SM_System:
    """Parallel-branch system container.

    An SM-System is a structure of parallel branches, each with a pre, nl,
    and post stage.

    Notes
    -----
    When ``is_analytic=True``, ``__call__`` returns a mapping from branch name
    to branch output rather than summing outputs.
    """

    systems: Dict[str, Dict[str, Any]]
    is_analytic: bool

    def __init__(
        self,
        pre: Optional[BranchComponent] = None,
        nl: Optional[BranchComponent] = None,
        post: Optional[BranchComponent] = None,
        name: str = "sm",
        is_analytic: bool = False,
    ) -> None:
        """Create a new SM_System.

        Parameters
        ----------
        pre:
            First function of a branch.
        nl:
            Second function of a branch.
        post:
            Third function of a branch.
        name:
            Identifies the initial branch.
        is_analytic:
            If ``True``, enables functionality where ``pre``, ``nl``, and
            ``post`` are not required to be callables.
        """

        self.systems = {
            name: {
                "pre": pre,
                "nl": nl,
                "post": post,
            }
        }
        self.is_analytic = is_analytic

    def update(self, subsystem: str, key: str, value: Any) -> None:
        """Update a single component (``pre``, ``nl``, or ``post``) in a branch."""

        self.systems[subsystem].update({key: value})

    def call_branch(self, inputs: Array, name: str = "sm") -> Array:
        """Evaluate one branch by name."""

        pre = self.systems[name]["pre"]
        nl = self.systems[name]["nl"]
        post = self.systems[name]["post"]

        result = post(nl(pre(inputs)))
        return result

    def __str__(self) -> str:  # pragma: no cover
        return str(self.systems)

    def __add__(self, sm: "SM_System") -> "SM_System":
        """Combine two SM_System objects in parallel."""

        new_system = SM_System(is_analytic=self.is_analytic or sm.is_analytic)
        new_system.systems.clear()
        new_system.systems.update(self.systems)
        new_system.systems.update(sm.systems)
        return new_system

    def __call__(self, inputs: Array) -> Union[Array, Dict[str, Array]]:
        """Evaluate the system.

        Returns
        -------
        Union[numpy.ndarray, dict[str, numpy.ndarray]]
            If ``is_analytic=False`` (default), returns the sum of all branch
            outputs.
            If ``is_analytic=True``, returns a mapping of branch name to output.
        """

        if self.is_analytic:
            return {name: self.call_branch(inputs, name) for name in self.systems}

        result_list = [self.call_branch(inputs, name) for name in self.systems]
        return np.sum(np.array(result_list), axis=0)
