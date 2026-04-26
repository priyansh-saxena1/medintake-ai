import os
import json
from pydantic import BaseModel

# ── Single unified system prompt — LLM sees the full workflow ──
SYSTEM_PROMPT = """You are a clinical intake assistant conducting a pre-visit patient interview.

YOUR WORKFLOW (follow this order):
1. INTAKE: Identify the patient's chief complaint (main reason for visit).
2. HPI (History of Present Illness): Collect these fields ONE AT A TIME, in order:
   - onset: when the symptom started
   - location: where in the body
   - duration: how long it has lasted
   - character: quality (sharp, dull, pressure, burning, etc.)
   - severity: how bad on a scale of 1-10
   - aggravating: what makes it worse
   - relieving: what makes it better
3. ROS (Review of Systems): Screen 3 body systems RELEVANT to the chief complaint.
   Examples of relevant systems:
   - Leg/knee/joint pain → musculoskeletal, neurological, vascular
   - Chest pain → cardiac, respiratory, gi
   - Headache → neurological, ophthalmologic, ent
   - Abdominal pain → gi, genitourinary, musculoskeletal
   - Back pain → musculoskeletal, neurological, genitourinary
4. DONE: When all HPI fields AND 3 ROS systems are filled, set reply to "Your clinical summary is ready. Please wait for the doctor."

CRITICAL RULES:
- NEVER re-ask a field that is already filled (marked ✅ in the status).
- Ask exactly ONE question per turn about the FIRST missing item.
- If a patient says "none"/"zero"/"no"/"denied", store that exact answer — do NOT leave it null.
- For ROS: store findings as a list, e.g. "musculoskeletal": ["joint stiffness", "no swelling"].
- Do NOT ask emotional/psychological questions — stick to physical symptoms.
- Output ONLY valid JSON, no extra text.

OUTPUT FORMAT:
{
  "chief_complaint": "..." or null,
  "onset": "..." or null,
  "location": "..." or null,
  "duration": "..." or null,
  "character": "..." or null,
  "severity": "..." or null,
  "aggravating": "..." or null,
  "relieving": "..." or null,
  "ros": {"system_name": ["finding1", "finding2"], ...},
  "emergency": false,
  "reply": "Your single question"
}"""

HPI_FIELDS = ["onset", "location", "duration", "character", "severity", "aggravating", "relieving"]
ROS_REQUIRED = 3


def build_state_context(current_json: str) -> str:
    """Build a human-readable status summary so the LLM knows exactly what's filled and missing."""
    try:
        state = json.loads(current_json)
    except Exception:
        state = {}

    lines = ["FIELD STATUS:"]

    # Chief complaint
    cc = state.get("chief_complaint")
    if cc:
        lines.append(f'  ✅ chief_complaint: "{cc}"')
    else:
        lines.append("  ❌ chief_complaint: MISSING — ask what brings them in")

    # HPI fields
    for field in HPI_FIELDS:
        val = state.get(field)
        if val:
            lines.append(f'  ✅ {field}: "{val}"')
        else:
            lines.append(f"  ❌ {field}: MISSING")

    # ROS
    ros = state.get("ros", {})
    if ros:
        for sys_name, findings in ros.items():
            lines.append(f'  ✅ ros.{sys_name}: {findings}')
    ros_remaining = ROS_REQUIRED - len(ros)
    if ros_remaining > 0:
        lines.append(f"  ❌ ros: {ros_remaining} more system(s) needed")
    else:
        lines.append(f"  ✅ ros: all {ROS_REQUIRED} systems collected")

    # Determine current phase
    if not cc:
        phase = "INTAKE"
    elif any(not state.get(f) for f in HPI_FIELDS):
        phase = "HPI"
        first_missing = next(f for f in HPI_FIELDS if not state.get(f))
        lines.append(f"\nCURRENT PHASE: {phase} — ask about '{first_missing}' next")
    elif ros_remaining > 0:
        phase = "ROS"
        lines.append(f"\nCURRENT PHASE: {phase} — ask about the next body system relevant to '{cc}'")
    else:
        phase = "DONE"
        lines.append(f"\nCURRENT PHASE: {phase} — all data collected, set reply to completion message")

    if not cc:
        lines.append(f"\nCURRENT PHASE: {phase}")

    return "\n".join(lines)


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
            for field in hpi_fields[1:]:
                if not state.get(field):
                    if last_patient_msg:
                        state[field] = last_patient_msg
                    break
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
            for sys_name in ros_systems:
                if sys_name not in ros:
                    if last_patient_msg:
                        ros[sys_name] = [last_patient_msg]
                        state["ros"] = ros
                    break
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
        Single LLM call: extracts clinical data + generates next question.
        The unified prompt + state context gives the LLM full visibility.
        """
        state_context = build_state_context(current_json)

        prompt = (
            f"{state_context}\n\n"
            f"CURRENT CLINICAL STATE (update with any new patient info):\n{current_json}\n\n"
            f"CONVERSATION TRANSCRIPT:\n{transcript}\n\n"
            "TASK: Read the patient's latest message. Extract any new clinical facts into the JSON. "
            "Then ask exactly ONE question about the FIRST missing item shown above. "
            "Return ONLY the updated JSON object."
        )

        import time
        import requests

        t_start = time.time()
        print(f"[Ollama] Starting inference for model '{self.model_name}'...")
        print(f"[Ollama] State context:\n{state_context}")

        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            "format": "json",
            "stream": False,
            "options": {
                "temperature": 0.0,
                "num_predict": 400
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