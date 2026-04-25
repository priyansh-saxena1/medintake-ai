import os
import json
from pydantic import BaseModel

CLINICAL_SYSTEM_PROMPT = (
    "You are a clinical intake assistant conducting a pre-visit patient interview. "
    "Be empathetic, warm, and highly professional. "
    "Do not diagnose or give medical advice. Keep responses under 2 sentences. "
)

class MockLLM:
    def __init__(self):
        pass

    def ask(self, instruction: str, system: str = CLINICAL_SYSTEM_PROMPT) -> str:
        # We will heavily mock the responses in graph.py for tests
        if "empathetic reply" in instruction.lower():
            if "chest" in instruction.lower():
                return "I'm sorry to hear about your chest pain. When did it start?"
            return "I understand. Can you tell me more?"
        
        # General fallback that allows tests to check for context
        if "onset" in instruction.lower():
            return "When did this start?"
        elif "severity" in instruction.lower() or "scale" in instruction.lower():
            return "On a scale of 1 to 10, how severe is this?"
        elif "location" in instruction.lower():
            return "Where exactly do you feel this?"
        
        return "Can you elaborate on that?"

    def ask_json(self, transcript: str, current_state: str, schema_cls: type[BaseModel]) -> BaseModel:
        # Mocking extraction logic for deterministic testing
        t_low = transcript.lower()
        state_dict = json.loads(current_state)
        
        # very basic test logic
        if "chest pain" in t_low:
            state_dict["chief_complaint"] = "chest pain"
        if "yesterday" in t_low or "morning" in t_low:
            if not state_dict.get("hpi"): state_dict["hpi"] = {}
            state_dict["hpi"]["onset"] = "this morning" if "morning" in t_low else "yesterday"
        if "center" in t_low:
            if not state_dict.get("hpi"): state_dict["hpi"] = {}
            state_dict["hpi"]["location"] = "center of chest"
        if "constant" in t_low:
            if not state_dict.get("hpi"): state_dict["hpi"] = {}
            state_dict["hpi"]["duration"] = "constant"
        if "pressure" in t_low or "tight" in t_low:
            if not state_dict.get("hpi"): state_dict["hpi"] = {}
            state_dict["hpi"]["character"] = "tight pressure"
        if "7" in t_low or "seven" in t_low:
            if not state_dict.get("hpi"): state_dict["hpi"] = {}
            state_dict["hpi"]["severity"] = "7/10"
        if "walk" in t_low or "running" in t_low:
            if not state_dict.get("hpi"): state_dict["hpi"] = {}
            state_dict["hpi"]["aggravating"] = "walking"
        if "rest" in t_low:
            if not state_dict.get("hpi"): state_dict["hpi"] = {}
            state_dict["hpi"]["relieving"] = "resting"
            
        if "palpitations" in t_low:
            if not state_dict.get("ros"): state_dict["ros"] = {}
            state_dict["ros"]["cardiac"] = ["palpitations", "no syncope"]
        if "breath" in t_low:
            if not state_dict.get("ros"): state_dict["ros"] = {}
            state_dict["ros"]["respiratory"] = ["shortness of breath", "no cough"]
        if "nausea" in t_low:
            if not state_dict.get("ros"): state_dict["ros"] = {}
            state_dict["ros"]["gi"] = ["no nausea"]
            
        if "crushing chest pain" in t_low or "heart attack" in t_low or "emergency" in t_low:
            state_dict["emergency_detected"] = True
            
        # Guarantee schema matches via Pydantic model_validate
        return schema_cls.model_validate(state_dict)

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

    def ask(self, instruction: str, system: str = CLINICAL_SYSTEM_PROMPT) -> str:
        self._load()
        import torch
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": instruction},
        ]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=0.4,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        response = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True,
        )
        return response.strip()

    def ask_json(self, transcript: str, current_state: str, schema_cls: type[BaseModel]) -> BaseModel:
        self._load()
        import torch
        
        system = (
            "You are a clinical data extraction engine. "
            "Your objective is to read the patient transcript and output exactly a valid JSON document "
            "that matches the requested schema. Extract all relevant medical facts you can find. "
            "Merge new facts into the existing state."
        )
        instruction = (
            f"CURRENT STATE JSON (Update this based on the transcript):\n{current_state}\n\n"
            f"TRANSCRIPT:\n{transcript}\n\n"
            f"Output ONLY valid JSON matching this schema structure:\n"
            f"{schema_cls.model_json_schema()}"
        )
        
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": instruction},
        ]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=400,
                temperature=0.1, # Keep low for JSON determinism
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        response = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True,
        )
        
        # Attempt to parse json from output
        json_str = response.strip()
        if "```json" in json_str:
            json_str = json_str.split("```json")[-1].split("```")[0]
        elif "```" in json_str:
            json_str = json_str.split("```")[-1].split("```")[0]
            
        try:
            parsed = json.loads(json_str)
            return schema_cls.model_validate(parsed)
        except Exception:
            # Fallback to current state if extraction fails (avoids crashing)
            try:
                return schema_cls.model_validate_json(current_state)
            except Exception:
                return schema_cls()


_llm_instance = None

def get_llm():
    global _llm_instance
    if _llm_instance is None:
        mock_mode = os.environ.get("MOCK_LLM", "true").lower() == "true"
        _llm_instance = MockLLM() if mock_mode else TransformersLLM()
    return _llm_instance