"""Framework-specific exceptions with actionable failure boundaries."""


class SolarWMError(RuntimeError):
    """Base class for expected framework failures."""


class ConfigurationError(SolarWMError):
    """The resolved run configuration violates a public contract."""


class DataContractError(SolarWMError):
    """An index, sample, camera, or tensor violates the data contract."""


class BackendContractError(SolarWMError):
    """A model backend cannot satisfy the requested route."""


class CheckpointError(SolarWMError):
    """A checkpoint transaction or compatibility gate is invalid."""
