from transformers import AutoTokenizer
from peft import AutoPeftModelForCausalLM

class TunedCausalLM:
    def __init__(self, model, tokenizer):
        self.peft_model = model
        self.tokenizer = tokenizer

    @classmethod
    def load(cls, checkpoint_path: str, base_model_name_or_path: str=None) -> "TunedCausalLM":
        """Loads an instance of this model.
        
        By default, the paths for the base model and tokenizer are contained within the adapter
        config of the tuned model. Note that in this context, a path may refer to a model to be
        downloaded from HF hub, or a local path on disk, the latter of which we must be careful
        with when using a model that was written on a different device.
        """
        if base_model_name_or_path is not None:
            raise NotImplementedError("WIP: override support not yet implemented")
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
        try:
            peft_model = AutoPeftModelForCausalLM.from_pretrained(checkpoint_path)
        except OSError as e:
            print("Failed to initialize checkpoint model!")
            raise e
        return cls(peft_model, tokenizer)

    def run(self, text: str) -> str:
        """Runs inference on an instance of this model."""
        input_ids = self.tokenizer(text, return_tensors="pt").input_ids
        peft_outputs = self.peft_model.generate(input_ids=input_ids)
        decoded_result = self.tokenizer.batch_decode(peft_outputs, skip_special_tokens=False)[0]
        return decoded_result
