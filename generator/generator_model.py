from transformers import GPT2LMHeadModel, GPT2Tokenizer

class FakeNewsGenerator:
    def __init__(self, model_name="gpt2"):
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name)

    def generate(self, prompt, max_length=100):
        inputs = self.tokenizer.encode(prompt, return_tensors="pt")
        outputs = self.model.generate(
            inputs,
            max_length=max_length,
            num_return_sequences=1,
            do_sample=True,
            temperature=0.9,
            top_k=50
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
