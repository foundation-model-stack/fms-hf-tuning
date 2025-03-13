# Local
from .fast_moe import get_callbacks


def get_additional_accel_framework_callbacks(active_plugins, **kwargs):
    callbacks = []
    for active_plugin in active_plugins:
        if "ScatterMoEAccelerationPlugin" == active_plugin[0]:
            callbacks.extend(get_callbacks(**kwargs))
    return callbacks
