"""Optional reference visualizer interface.

MolmoSpaces disables the Pinocchio visualizer in normal operation.  Keeping a
no-op implementation makes that dependency explicit and avoids import hacks.
"""


class RuntimeReferencePublisher:
    def __init__(self, *args, **kwargs) -> None:
        del args, kwargs

    def publish(self, *args, **kwargs) -> None:
        del args, kwargs

    def close(self) -> None:
        return None
