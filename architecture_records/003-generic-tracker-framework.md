# Generic Tracker Framework

**Deciders(s)**: Sukriti Sharma (sukriti.sharma4@ibm.com), Alexander Brooks (alex.brooks@ibm.com), Raghu Ganti (rganti@us.ibm.com), Dushyant Behl (dushyantbehl@in.ibm.com), Ashok Pon Kumar (ashokponkumar@in.ibm.com)

**Date (YYYY-MM-DD)**:  2024-03-06

**Obsoletes ADRs**:  NA

**Modified By ADRs**:  NA

**Relevant Issues**: [1](https://github.com/foundation-model-stack/fms-hf-tuning/issues/34), [2](https://github.com/foundation-model-stack/fms-hf-tuning/issues/33)

- [Summary and Objective](#summary-and-objective)
  - [Motivation](#motivation)
  - [User Benefit](#user-benefit)
- [Decision](#decision)
  - [Alternatives Considered](#alternatives-considered)
- [Consequences](#consequences)
- [Detailed Design](#detailed-design)

## Summary and Objective

This PR introduces a generic interface `class Tracker` which implements basic functionality needed to be satisfied by any tracker to be implemented inside `fms-hf-tuning`.
Tracker here means an agent which can track AI and system metrics like [Aimstack](https://aimstack.io/) or [WandB](https://wandb.ai/site).

### Motivation

The current code in `fms-hf-tuning` has [Aimstack](https://aimstack.io/) as the sole integration point in the file [aim_loader.py](https://github.com/foundation-model-stack/fms-hf-tuning/blob/74caf85140a112cd9289502b0777baac636adf1d/tuning/aim_loader.py) (taken from the latest head of the tree at the point of proposing this change).

Users of `fms-hf-tuning` in the current state are forced to used Aimstack and cannot interface with any other tracker like quite popular WandB. If a user would want to add any new tracker to the current code they would need to implement all functionality and change the code in a heavy manner.
To interface with other tracker we need a more Modular interface in our tuning script, further in the current code it is not possible to disable Aimstack and users of this repo have raised concern regarding the same (https://github.com/foundation-model-stack/fms-hf-tuning/pull/20).

With the new modular inteface we also add support for tracking custom experiment metadata and custom metrics with just one line of change in the code.

So, due to the limitations of current code and lack of modular structure this ADR introduces modular design to tracking in fms-hf-tuning which enables future support for any tracker.

### User Benefit

Users of this updated design will be able to,

1. Implement and interface with any tracker they want by just implementing 4 functions of an interface.
1. Run the code without any tracker if they do not want to use it.
1. Track any custom metrics or associate any experiment metadata they need with the training runs.

## Decision

### Alternatives Considered

Alternatives to this design is already implemented in the code, we considered expanding that but then for every tracker we would need to introduce a new loader and with no well defined structure or interface to tracking the core tuning script would need to have too much boiler plate and conditional checks implemented to use one or the other tracker.

## Consequences

### Advantages

- Modular design to keep the core training loop clean while working with trackers
- Support for tracking any metrics and metatada
- Simple interface to expand and attach any tracker

### Impact on performance

None. The interface does not add any extra overhead to the tracking. If a tracker is not implemented the tracker interface functions are NOOP modules.

## Detailed Design

The changes to the code are these,

```
class Tracker:
    def __init__(self, name=None, tracker_config=None) -> None:
        if tracker_config is not None:
            self.config = tracker_config
        if name is None:
            self._name = "None"
        else:
            self._name = name

    # we use args here to denote any argument.
    def get_hf_callback(self):
        return None

    def track(self, metric, name, stage):
        pass

    # Object passed here is supposed to be a KV object
    # for the parameters to be associated with a run
    def set_params(self, params, name):
        pass
```

This interface expects any tracker to implement just 4 basic functions. 

1. `init` to initialise the tracker using the config passed from command line
1. `get_hf_callback` to be called to get the hugging face callback for the specific tracker.
1. `track` to track any custom metrics
1. `set_params` to set any experiment metadata as additional parameters

In addition, we also.

1. We also introduce a tracker factory which initializes the available tracker.
1. We remove the file `aim_loader.py` and implement the same code as a `Tracker` in the folder `trackers/aimstack_tracker.py`
1. We also implement the `track` and `set_params` functions for `Aimstack`
1. We change the main tuning script to use `Tracker` inteface instead of directly calling `Aimstack` functions.
