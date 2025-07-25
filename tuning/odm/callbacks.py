# First Party
from transformers import TrainerCallback, TrainerState
import os


class ODMCallback(TrainerCallback):
    def __init__(self, eval_iter=10):
        self.eval_iter = 10

    def on_step_end(self, args, state: TrainerState, control, **kwargs):
        if state.is_world_process_zero:
            state.dataset.update_mixer(state.inputs, state.loss, None)
            with open(os.path.join(args.output_dir, "rl_proj.jsonl"), "a") as f:
                f.write(str(state.dataset.rl_agent._probabilities) + "\n")
