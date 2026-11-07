"""Attempt all owned cleanups; retain the original exception and secondaries."""


def cleanup(actions, primary=None):
    failures = []
    for name, callback in actions:
        try:
            callback()
        except BaseException as error:
            failures.append((name, error))
    if primary is not None:
        primary.solarwm_cleanup_failures = failures
        return failures
    if failures:
        error = failures[0][1]
        error.solarwm_cleanup_failures = failures[1:]
        raise error
    return failures
