# Copyright The FMS HF Tuning Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Third Party
from transformers.utils import logging
from trl import SFTTrainer
import torch

logger = logging.get_logger(__name__)

DEFAULT_LABELS_KEY = "labels"

################### Some Notes on the below loss calculation. ##############################
# This loss function just replaces trainer loss function to calculate sum reduction
# over the model forward pass. This function is useful while using high amount of
# gradient accumulation to ensure all tokens are accounted for equally.
#
# In HF Trainer performing a *true* reduce loss sum calculation requires to change
# the trainier.training_step function as well to ensure the `backwards` call is done
# on the combined `sum` loss rather than a loss which is scaled with respect to GAS.
# See these lines - https://github.com/huggingface/transformers/blob/\
#                           08e3217bafddc5d11ce0e7369bcfaaabe5501ba5/\
#                           src/transformers/trainer.py#L3765C1-L3774C54
#
# The methodology we have is not too intrusive and not to change `training_step()` as
# that is equivalent of almost recreating a training loop. Our approach is more towards
# providing a close approximation to the reduce loss sum calculation.
#
# This feature is provided as an experimental feature and not fully claimed to be supported.
#
# Known limitation (will be fixed in the next releases) -
#   Not fully tested and compatible with PEFT especially PEFT PT.
#############################################################################################


class SumLossSFTTrainer(SFTTrainer):

    vocab_size: int

    def __init__(
        self,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        vocab = self.model.get_input_embeddings()
        self.vocab_size = vocab.weight.shape[0]

        # Disable model loss kwargs as we are overriding the model loss
        # This is so that the loss calculated by us is divided over actual
        # actual gradient accumulation steps inside HF Trainer
        #
        # See this code -
        # https://github.com/huggingface/transformers/blob/\
        #      41b9b92b52215bed472c9a534a06abbc3a9a95cd/src/transformers/trainer.py#L3769
        self.model_accepts_loss_kwargs = False
        logger.info(
            "✅ Initialized SumLossSFTTrainer. "
            + "Switching trainer loss function with cross entropy loss sum reduction.\n"
            + "This is an experimental feature and should be used as such "
            + " please report any issue you see with this function to the maintainers"
        )

    # Overrides trl/sft_trainer::SFTTrainer compute_loss function.
    #
    # This loss function is taken from OpenInstruct
    #
    # https://github.com/allenai/open-instruct/blob/open_instruct/finetune.py
    #
    # Using sum reduction for CrossEntropyLoss according to OpenInstruct
    # helps in ensuring all tokens are weighed equally in the dataset which is
    # important for high amount of gradient accumulation steps.
    # For more details see their discussion on this transformers issue
    # URL - https://github.com/huggingface/transformers/issues/24725
    #
    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        """
        Function to switch loss function calculation to reduce_loss=sum
        """
        if self.model_accepts_loss_kwargs:
            loss_kwargs = {}
            if num_items_in_batch is not None:
                loss_kwargs["num_items_in_batch"] = num_items_in_batch
            inputs = {**inputs, **loss_kwargs}

        # Run the forward pass
        outputs = model(**inputs, use_cache=False)

        # Extract logits and perform calculation for loss outside the modelling class
        logits = outputs.logits
        labels = inputs[DEFAULT_LABELS_KEY]

        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        # get the loss function from torch with sum reduction
        loss_fct = torch.nn.CrossEntropyLoss(reduction="sum")

        # Flatten tensors as expected by crossentropyloss
        shift_logits = shift_logits.view(-1, self.vocab_size)
        shift_labels = shift_labels.view(-1)

        # Shift the data to device and run loss.
        shift_labels = shift_labels.to(shift_logits.device)
        loss = loss_fct(shift_logits, shift_labels)

        return (loss, outputs) if return_outputs else loss
