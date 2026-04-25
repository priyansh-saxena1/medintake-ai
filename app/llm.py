import os
import json
import re
from pydantic import BaseModel

COMBINED_SYSTEM_PROMPT = """You are a clinical intake assistant AI. You have two jobs per turn:

JOB 1 (EXTRACT): Read the FULL conversation and update the clinical JSON state with any new information the patient provided. Only extract facts explicitly stated.

JOB 2 (RESPOND): Based on what is STILL MISSING from the clinical state, ask the patient ONE natural, empathetic question. Do NOT ask about things already filled in.

CRITICAL RULES:
- Output ONLY valid JSON, nothing else.
- Do NOT diagnose or give medical advice.
- Do NOT ask more than one question.
- If all fields are complete, set reply to "Thank you — I have everything I need."

OUTPUT FORMAT (strictly follow this, no extra text):
{
  "chief_complaint": "...",
  "onset": "...",
  "location": "...",
  "duration": "...",
  "character": "...",
  "severity": "...",
  "aggravating": "...",
  "relieving": "...",
  "ros": {"system_name": ["finding1", "finding2"]},
  "reply": "The single question to ask the patient next"
}

Use null for any field not yet known. Keep existing values if the patient didn't add new info."""


class CombinedOutput(BaseModel):
    chief_complaint: str | None = None
    onset: str | None = None
    location: str | None = None
    duration: str | None = None
    character: str | None = None
    severity: str | None = None
    aggravating: str | None = None
    relieving: str | None = None
    ros: dict[str, list[str]] = {}
    reply: str = ""


class MockLLM:
    def combined_call(self, transcript: str, current_json: str) -> CombinedOutput:
        """Single call: extract + generate reply. No real inference in mock mode."""
        t = transcript.lower()
        try:
            state = json.loads(current_json)
        except Exception:
            state = {}

        # --- Extraction ---
        if "chest pain" in t and not state.get("chief_complaint"):
            state["chief_complaint"] = "chest pain"
        if any(w in t for w in ["yesterday", "this morning", "last night", "hours ago", "days ago", "since"]):
            if not state.get("onset"):
                if "yesterday" in t:
                    state["onset"] = "yesterday"
                elif "this morning" in t or "morning" in t:
                    state["onset"] = "this morning"
                else:
                    state["onset"] = "recently"
        if any(w in t for w in ["center", "left", "right", "chest", "stomach", "head", "arm"]):
            if not state.get("location"):
                if "center" in t:
                    state["location"] = "center of chest"
                elif "left" in t:
                    state["location"] = "left side of chest"
        if any(w in t for w in ["constant", "intermittent", "comes and goes", "all day", "hours"]):
            if not state.get("duration"):
                state["duration"] = "constant" if "constant" in t else "intermittent"
        if any(w in t for w in ["pressure", "tight", "squeezing", "sharp", "dull", "burning", "stabbing"]):
            if not state.get("character"):
                if "tight" in t or "squeezing" in t:
                    state["character"] = "tight, squeezing pressure"
                elif "sharp" in t:
                    state["character"] = "sharp"
        # Severity — match "N out of 10", "N/10", or isolated score digit
        sev_match = re.search(r'\b([1-9]|10)\s*(?:out of|/|over)\s*10\b', t, re.IGNORECASE)
        if not sev_match:
            sev_match = re.search(r'\bseverity\s+(?:is\s+)?([1-9]|10)\b', t, re.IGNORECASE)
        if sev_match and not state.get("severity"):
            state["severity"] = f"{sev_match.group(1)}/10"
        if any(w in t for w in ["walk", "run", "climb", "exert", "stress", "eating", "lying"]):
            if not state.get("aggravating"):
                if "walk" in t: state["aggravating"] = "walking"
                elif "run" in t: state["aggravating"] = "running"
                elif "climb" in t: state["aggravating"] = "climbing stairs"
        if any(w in t for w in ["rest", "sit", "antacid", "medication", "nitroglycerin"]):
            if not state.get("relieving"):
                state["relieving"] = "resting"
        if "palpitation" in t:
            ros = state.get("ros", {})
            ros["cardiac"] = ["palpitations present"] + (["no leg swelling"] if "no" in t and "swell" in t else [])
            state["ros"] = ros
        if "breath" in t or "wheez" in t or "cough" in t:
            ros = state.get("ros", {})
            ros["respiratory"] = ["shortness of breath" if "breath" in t else "no shortness of breath",
                                    "no cough" if ("no" in t and "cough" in t) else ("cough" if "cough" in t else "no cough")]
            state["ros"] = ros
        if "nausea" in t or "vomit" in t or "heartburn" in t:
            ros = state.get("ros", {})
            ros["gi"] = ["no nausea" if ("no" in t and "nausea" in t) else "nausea",
                         "no vomiting" if ("no" in t and "vomit" in t) else "vomiting present"]
            state["ros"] = ros
        
        state["emergency"] = any(e in t for e in ["crushing chest pain", "heart attack", "can't breathe", "suicide", "kill myself"])

        # --- Determine next question ---
        if not state.get("chief_complaint"):
            state["reply"] = "What brings you in today?"
        elif not state.get("onset"):
            cc = state.get("chief_complaint", "this")
            state["reply"] = f"When did the {cc} start?"
        elif not state.get("location"):
            state["reply"] = "Where exactly do you feel it?"
        elif not state.get("duration"):
            state["reply"] = "Is it constant or does it come and go?"
        elif not state.get("character"):
            state["reply"] = "How would you describe it — sharp, dull, pressure, or tightness?"
        elif not state.get("severity"):
            state["reply"] = "On a scale of 1 to 10, how severe is it right now?"
        elif not state.get("aggravating"):
            state["reply"] = "Does anything make it worse, like physical activity?"
        elif not state.get("relieving"):
            state["reply"] = "What helps relieve it?"
        else:
            ros = state.get("ros", {})
            cc = state.get("chief_complaint", "chest pain")
            if "cardiac" not in ros:
                state["reply"] = "Any heart-related symptoms — palpitations or leg swelling?"
            elif "respiratory" not in ros:
                state["reply"] = "Any shortness of breath, wheezing, or coughing?"
            elif "gi" not in ros:
                state["reply"] = "Any nausea, vomiting, or heartburn?"
            else:
                state["reply"] = "Thank you — I have everything I need."

        return CombinedOutput.model_validate(state)


class OllamaLLM:
    def __init__(self):
        self.model_name = os.environ.get("MODEL_NAME", "qwen2.5:0.5b")
        self.api_url = "http://localhost:11434/api/chat"

    def combined_call(self, transcript: str, current_json: str) -> CombinedOutput:
        """
        Calls the local Ollama instance using the /chat endpoint so system tags 
        are properly applied.
        """
        prompt = (
            f"CURRENT CLINICAL STATE (update with any new patient info):\n{current_json}\n\n"
            f"FULL CONVERSATION TRANSCRIPT:\n{transcript}\n\n"
            "Instructions: Extract all new clinical facts from the transcript, merge them into the state, "
            "and generate exactly ONE empathetic follow-up question for whatever is still missing. "
            "Return ONLY the JSON object, no other text."
        )

        import time
        import requests
        
        t_start = time.time()
        print(f"[Ollama] Starting inference for model '{self.model_name}'...")
        
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": COMBINED_SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            "format": "json",
            "stream": False,
            "options": {
                "temperature": 0.0,
                "num_predict": 250
            }
        }
        
        try:
            response = requests.post(self.api_url, json=payload, timeout=60)
            response.raise_for_status()
            data = response.json()
            raw = data.get("response", "")
        except Exception as e:
            print(f"[Ollama] ERROR calling local Ollama API: {e}")
            print("[Ollama] Make sure Ollama is installed and running, and the model is downloaded!")
            return CombinedOutput.model_validate_json(current_json)

        print(f"[Ollama] Inference completed in {time.time() - t_start:.2f}s total.")

        # Parse JSON robustly
        json_str = raw
        if "```json" in json_str:
            json_str = json_str.split("```json", 1)[1].split("```")[0]
        elif "```" in json_str:
            json_str = json_str.split("```", 1)[1].split("```")[0]

        start = json_str.find("{")
        end = json_str.rfind("}") + 1
        if start != -1 and end > start:
            json_str = json_str[start:end]

        try:
            parsed = json.loads(json_str)
            return CombinedOutput.model_validate(parsed)
        except Exception as e:
            print(f"[Ollama] JSON parse error: {e}\nRaw output: {raw[:300]}")
            try:
                base = CombinedOutput.model_validate_json(current_json)
                base.reply = "Could you please repeat that? I want to make sure I understood correctly."
                return base
            except Exception:
                return CombinedOutput(reply="Could you please repeat that?")


_llm_instance = None

def get_llm():
    global _llm_instance
    if _llm_instance is None:
        mock_mode = os.environ.get("MOCK_LLM", "true").lower() == "true"
        _llm_instance = MockLLM() if mock_mode else OllamaLLM()
    return _llm_instance