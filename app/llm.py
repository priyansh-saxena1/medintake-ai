import os

CLINICAL_SYSTEM_PROMPT = (
    "You are a clinical intake assistant conducting a pre-visit patient interview. "
    "Ask one clear, concise, professional medical question at a time. "
    "Do not diagnose or give medical advice. Keep responses under 2 sentences. "
    "Be empathetic but professional."
)


class MockLLM:
    def __init__(self):
        self.hpi_fields = ["onset", "location", "duration", "character", "severity", "aggravating", "relieving"]
        self.current_hpi_index = 0
        self.ros_systems_done = False

    def ask(self, instruction: str) -> str:
        return ""  # unused in mock mode — graph uses hardcoded questions

    def reset(self):
        self.current_hpi_index = 0
        self.ros_systems_done = False


class TransformersLLM:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.model_name = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-0.5B-Instruct")

    def _load(self):
        if self.model is None:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float32,
                device_map="cpu",
            )

    def ask(self, instruction: str) -> str:
        self._load()
        import torch
        messages = [
            {"role": "system", "content": CLINICAL_SYSTEM_PROMPT},
            {"role": "user", "content": instruction},
        ]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=80,
                temperature=0.3,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        response = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True,
        )
        return response.strip()


_llm_instance = None


def get_llm():
    global _llm_instance
    if _llm_instance is None:
        mock_mode = os.environ.get("MOCK_LLM", "true").lower() == "true"
        _llm_instance = MockLLM() if mock_mode else TransformersLLM()
    return _llm_instance