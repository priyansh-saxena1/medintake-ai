import os
import json
from pydantic import BaseModel

INTAKE_PROMPT = """You are a clinical intake assistant. The patient just arrived.

JOB: Extract the chief complaint from the conversation. Ask ONE simple question to identify their main symptom.

RULES:
- Output ONLY valid JSON.
- If you already know the chief complaint, ask about onset to move forward.
- Do NOT diagnose or give medical advice.

OUTPUT FORMAT:
{
  "chief_complaint": "the main symptom" or null,
  "onset": null, "location": null, "duration": null,
  "character": null, "severity": null, "aggravating": null, "relieving": null,
  "ros": {},
  "reply": "Your question to the patient"
}"""

HPI_PROMPT = """You are a clinical intake assistant collecting History of Present Illness (HPI) using OLDCARTS.

JOB 1 (EXTRACT): Read the conversation and update the JSON with any new patient info. If a patient denies something or says "none"/"zero"/"no", store that exact word — do NOT leave it null.

JOB 2 (RESPOND): Ask ONE question about the FIRST missing field below. Do NOT re-ask fields already filled.

FIELDS TO COLLECT (in order):
- onset: when the symptom started
- location: where in the body
- duration: how long it has lasted
- character: quality of pain (sharp, dull, pressure, burning, etc.)
- severity: how bad on a scale of 1-10
- aggravating: what makes it worse
- relieving: what makes it better

RULES:
- Output ONLY valid JSON, no extra text.
- Ask exactly ONE question per turn.
- Keep existing values. Use null for unknowns.

OUTPUT FORMAT:
{
  "chief_complaint": "...",
  "onset": "..." or null,
  "location": "..." or null,
  "duration": "..." or null,
  "character": "..." or null,
  "severity": "..." or null,
  "aggravating": "..." or null,
  "relieving": "..." or null,
  "ros": {},
  "reply": "Your single question"
}"""

ROS_PROMPT = """You are a clinical intake assistant performing a Review of Systems (ROS).

All HPI fields are already collected. Now you must screen for symptoms in OTHER body systems that are RELEVANT to the patient's chief complaint.

JOB 1 (EXTRACT): The patient just answered a question about a body system. Extract their answer into the "ros" dict under the appropriate system key (e.g. "musculoskeletal": ["joint stiffness", "no swelling"]).

JOB 2 (RESPOND): Ask about the NEXT relevant body system that is NOT yet in the "ros" dict.

CHOOSING SYSTEMS: Pick 3 systems that are clinically relevant to the chief complaint. Examples:
- Leg/knee/joint pain → musculoskeletal, neurological, vascular
- Chest pain → cardiac, respiratory, gi
- Headache → neurological, ophthalmologic, ent
- Abdominal pain → gi, genitourinary, musculoskeletal
- Back pain → musculoskeletal, neurological, genitourinary

RULES:
- Output ONLY valid JSON.
- Ask about ONE system at a time.
- If the patient denies symptoms, store as ["no X", "no Y"].
- Once 3 systems are in "ros", set reply to "Thank you — I have everything I need."
- Do NOT ask emotional, psychological, or off-topic questions.

OUTPUT FORMAT:
{
  "chief_complaint": "...", "onset": "...", "location": "...", "duration": "...",
  "character": "...", "severity": "...", "aggravating": "...", "relieving": "...",
  "ros": {"system_name": ["findings"], ...},
  "reply": "Your single ROS question"
}"""


def get_system_prompt(stage: str) -> str:
    """Return the appropriate system prompt for the current clinical stage."""
    if stage == "ros":
        return ROS_PROMPT
    elif stage == "hpi":
        return HPI_PROMPT
    else:
        return INTAKE_PROMPT


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
    emergency: bool = False
    reply: str = ""


class MockLLM:
    """Minimal mock for testing — no regex, no extraction logic. Just walks through fields."""
    def combined_call(self, transcript: str, current_json: str, stage: str = "intake") -> CombinedOutput:
        try:
            state = json.loads(current_json)
        except Exception:
            state = {}

        # Mock just steps through HPI fields in order, using the patient's last message as the value
        lines = transcript.strip().split("\n")
        last_patient_msg = ""
        for line in reversed(lines):
            if line.startswith("Patient:"):
                last_patient_msg = line.replace("Patient:", "").strip()
                break

        hpi_fields = ["chief_complaint", "onset", "location", "duration", "character", "severity", "aggravating", "relieving"]
        ros_systems = ["cardiac", "respiratory", "gi"]

        if stage == "intake":
            if last_patient_msg and not state.get("chief_complaint"):
                state["chief_complaint"] = last_patient_msg
            state["reply"] = "What brings you in today?" if not state.get("chief_complaint") else f"When did the {state['chief_complaint']} start?"

        elif stage == "hpi":
            # Fill the first empty HPI field with the patient's answer
            for field in hpi_fields[1:]:  # skip chief_complaint, already filled
                if not state.get(field):
                    if last_patient_msg:
                        state[field] = last_patient_msg
                    break
            # Ask about the next missing field
            for field in hpi_fields[1:]:
                if not state.get(field):
                    labels = {"onset": "when it started", "location": "where you feel it",
                              "duration": "how long it's lasted", "character": "what it feels like",
                              "severity": "how severe it is (1-10)", "aggravating": "what makes it worse",
                              "relieving": "what makes it better"}
                    state["reply"] = f"Can you tell me {labels.get(field, field)}?"
                    break
            else:
                state["reply"] = "Thank you, moving on to review of systems."

        elif stage == "ros":
            ros = state.get("ros", {})
            # Fill the first empty ROS system
            for sys_name in ros_systems:
                if sys_name not in ros:
                    if last_patient_msg:
                        ros[sys_name] = [last_patient_msg]
                        state["ros"] = ros
                    break
            # Ask about next missing system
            for sys_name in ros_systems:
                if sys_name not in ros:
                    state["reply"] = f"Any {sys_name} symptoms?"
                    break
            else:
                state["reply"] = "Thank you — I have everything I need."

        return CombinedOutput.model_validate(state)


class OllamaLLM:
    def __init__(self):
        self.model_name = os.environ.get("MODEL_NAME", "qwen2.5:0.5b")
        self.api_url = "http://localhost:11434/api/chat"

    def combined_call(self, transcript: str, current_json: str, stage: str = "intake") -> CombinedOutput:
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
                {"role": "system", "content": get_system_prompt(stage)},
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
            raw = data.get("message", {}).get("content", "").strip()
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
            # Coerce empty strings and literal "null" back to None
            for field in ["chief_complaint", "onset", "location", "duration",
                          "character", "severity", "aggravating", "relieving"]:
                v = parsed.get(field)
                if v is not None and str(v).strip() in ("", "null"):
                    parsed[field] = None
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