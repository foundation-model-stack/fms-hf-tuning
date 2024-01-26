from transformers import AutoTokenizer
from peft import AutoPeftModelForCausalLM

class TunedCausalLM:
    def __init__(self, model, tokenizer):
        self.peft_model = model
        self.tokenizer = tokenizer

    @classmethod
    def load(cls, checkpoint_path: str) -> "TunedCausalLM":
        """Loads an instance of this model."""
        # Would be nice to be able to override the base model / tokenizer path...
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
        peft_model = AutoPeftModelForCausalLM.from_pretrained(checkpoint_path)
        return cls(peft_model, tokenizer)

    def run(self, text: str) -> str:
        """Runs inference on an instance of this model."""
        input_ids = self.tokenizer(text, return_tensors="pt").input_ids
        peft_outputs = self.peft_model.generate(input_ids=input_ids)
        decoded_result = self.tokenizer.batch_decode(peft_outputs, skip_special_tokens=False)
        return decoded_result

if __name__ == "__main__":
    loaded_model = TunedCausalLM.load("/home/SSO/us2j7257/fms-hf-tuning/out/checkpoint-13/")
    print(loaded_model.run("Tweet text : It was a fine day on twitter. Label : "))
