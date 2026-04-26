import os
import json
from pydantic import BaseModel, Field

# ── Reasoning is required FIRST so the model "thinks" before filling fields ──
SYSTEM_PROMPT = """You are a clinical intake assistant conducting a pre-visit patient interview.

STEP 1 — REASON (fill "_reasoning" first):
  a. Quote the patient's LATEST message verbatim.
  b. State every clinical fact it contains (onset, location, severity, etc.).
  c. List which JSON fields are still missing after applying those facts.
  d. Choose ONE question for the first missing field.

STEP 2 — OUTPUT the JSON below (no extra text):

{
  "_reasoning": "... your step-by-step analysis ...",
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
}

WORKFLOW ORDER:
1. INTAKE — ask chief complaint (what brings them in).
2. HPI — collect onset → location → duration → character → severity → aggravating → relieving ONE AT A TIME.
3. ROS — screen 3 body systems RELEVANT to the chief complaint (e.g. leg pain → musculoskeletal, neurological, vascular).
4. DONE — when all HPI fields AND 3 ROS systems are filled, set reply to:
   "Your clinical summary is ready. Please wait for the doctor."

CRITICAL RULES:
- NEVER re-ask a field already marked ✅ in the status block.
- Ask ONE question per turn about the FIRST missing item.
- Store "none"/"no"/"denied" answers — do NOT leave them null.
- For ROS findings use a descriptive phrase: "no swelling", "tingling in left calf", not bare "yes"/"no".
- Do NOT ask emotional/psychological questions.
- Output ONLY valid JSON."""

HPI_FIELDS = ["onset", "location", "duration", "character", "severity", "aggravating", "relieving"]
ROS_REQUIRED = 3

BRIEF_SYSTEM_PROMPT = """You are a clinical documentation assistant.
Given structured intake data, write a concise, professional clinical brief in plain text.
Use standard clinical language. Do NOT invent findings not present in the data.
Output ONLY a JSON object with one key "narrative" whose value is the formatted brief string."""


def build_state_context(current_json: str) -> str:
    """Build a human-readable status summary so the LLM knows exactly what's filled."""
    try:
        state = json.loads(current_json)
    except Exception:
        state = {}

    lines = ["FIELD STATUS:"]

    cc = state.get("chief_complaint")
    if cc:
        lines.append(f'  ✅ chief_complaint: "{cc}"')
    else:
        lines.append("  ❌ chief_complaint: MISSING — ask what brings them in")

    for field in HPI_FIELDS:
        val = state.get(field)
        if val:
            lines.append(f'  ✅ {field}: "{val}"')
        else:
            lines.append(f"  ❌ {field}: MISSING")

    ros = state.get("ros", {})
    for sys_name, findings in ros.items():
        lines.append(f'  ✅ ros.{sys_name}: {findings}')
    ros_remaining = ROS_REQUIRED - len(ros)
    if ros_remaining > 0:
        lines.append(f"  ❌ ros: {ros_remaining} more system(s) needed")
    else:
        lines.append(f"  ✅ ros: all {ROS_REQUIRED} systems collected")

    if not cc:
        phase = "INTAKE"
        lines.append(f"\nCURRENT PHASE: {phase}")
    elif any(not state.get(f) for f in HPI_FIELDS):
        phase = "HPI"
        first_missing = next(f for f in HPI_FIELDS if not state.get(f))
        lines.append(f"\nCURRENT PHASE: {phase} — ask about '{first_missing}' next")
    elif ros_remaining > 0:
        phase = "ROS"
        if ros:
            already = ", ".join(ros.keys())
            lines.append(f"  ℹ️ Already covered: {already} — DO NOT ask about these again")
        lines.append(f"\nCURRENT PHASE: {phase} — ask about the next body system relevant to '{cc}'")
    else:
        phase = "DONE"
        lines.append(f"\nCURRENT PHASE: {phase} — all data collected")

    return "\n".join(lines)


class CombinedOutput(BaseModel):
    # Named 'reasoning' (no leading underscore — Pydantic reserves those for private attrs).
    # The LLM prompt asks it to fill "_reasoning" in JSON; we accept that via alias.
    reasoning: str = Field(default="", alias="_reasoning")
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

    # Allow both the alias ("_reasoning") and the field name ("reasoning") when parsing
    model_config = {"populate_by_name": True}


class MockLLM:
    """Minimal mock for testing."""
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

        ros_systems = ["cardiac", "respiratory", "gi"]

        if stage == "intake":
            if last_patient_msg and not state.get("chief_complaint"):
                state["chief_complaint"] = last_patient_msg
            state["reply"] = "What brings you in today?" if not state.get("chief_complaint") else f"When did the {state['chief_complaint']} start?"

        elif stage == "hpi":
            for field in ["onset", "location", "duration", "character", "severity", "aggravating", "relieving"]:
                if not state.get(field):
                    if last_patient_msg:
                        state[field] = last_patient_msg
                    break
            labels = {"onset": "when it started", "location": "where you feel it",
                      "duration": "how long it's lasted", "character": "what it feels like",
                      "severity": "how severe it is (1-10)", "aggravating": "what makes it worse",
                      "relieving": "what makes it better"}
            for field in ["onset", "location", "duration", "character", "severity", "aggravating", "relieving"]:
                if not state.get(field):
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

    def generate_brief_narrative(self, brief_data: dict) -> str:
        """Mock brief narrative — just formats the data cleanly."""
        cc = brief_data.get("chief_complaint", "unspecified")
        hpi = brief_data.get("hpi", {})
        ros = brief_data.get("ros", {})

        parts = [
            f"Patient presents with {cc}.",
            f"Symptoms began {hpi.get('onset', 'at an unspecified time')},",
            f"localised to the {hpi.get('location', 'unspecified area')}.",
            f"Duration: {hpi.get('duration', 'not specified')}.",
            f"Character: {hpi.get('character', 'not specified')}.",
            f"Severity: {hpi.get('severity', 'not rated')}/10.",
            f"Aggravated by {hpi.get('aggravating', 'unspecified')}; relieved by {hpi.get('relieving', 'unspecified')}.",
        ]

        if ros:
            ros_lines = []
            for system, findings in ros.items():
                ros_lines.append(f"{system.capitalize()}: {', '.join(findings)}.")
            parts.append("Review of systems: " + " ".join(ros_lines))

        return " ".join(parts)


class OllamaLLM:
    def __init__(self):
        self.model_name = os.environ.get("MODEL_NAME", "qwen2.5:0.5b")
        self.api_url = "http://localhost:11434/api/chat"

    def _call_ollama(self, system: str, user: str, temperature: float = 0.0, num_predict: int = 600) -> str:
        """Single helper that calls Ollama and returns raw content string."""
        import requests, time
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user}
            ],
            "format": "json",
            "stream": False,
            "options": {"temperature": temperature, "num_predict": num_predict}
        }
        t0 = time.time()
        response = requests.post(self.api_url, json=payload, timeout=60)
        response.raise_for_status()
        data = response.json()
        print(f"[Ollama] Inference completed in {time.time() - t0:.2f}s")
        return data.get("message", {}).get("content", "").strip()

    def _parse_json(self, raw: str) -> dict:
        """Strip markdown fences and parse JSON robustly."""
        s = raw
        if "```json" in s:
            s = s.split("```json", 1)[1].split("```")[0]
        elif "```" in s:
            s = s.split("```", 1)[1].split("```")[0]
        start, end = s.find("{"), s.rfind("}") + 1
        if start != -1 and end > start:
            s = s[start:end]
        return json.loads(s)

    def combined_call(self, transcript: str, current_json: str, stage: str = "intake") -> CombinedOutput:
        import time

        state_context = build_state_context(current_json)

        # ── FIX 1: Explicitly surface the latest patient message ──────────────
        lines = transcript.strip().split("\n")
        latest_patient_msg = ""
        for line in reversed(lines):
            if line.startswith("Patient:"):
                latest_patient_msg = line.replace("Patient:", "").strip()
                break

        prompt = (
            f"PATIENT'S LATEST MESSAGE (extract facts from THIS message):\n"
            f"  \"{latest_patient_msg}\"\n\n"
            f"{state_context}\n\n"
            f"FULL CONVERSATION (for context only — extract facts from latest message above):\n"
            f"{transcript}\n\n"
            f"CURRENT CLINICAL STATE:\n{current_json}\n\n"
            "TASK:\n"
            "1. Fill '_reasoning': quote the latest message, list every fact it contains, "
            "   identify still-missing fields, choose ONE question.\n"
            "2. Update ALL fields with new facts from the latest message.\n"
            "3. Set 'reply' to ONE question about the FIRST missing field.\n"
            "Return ONLY valid JSON."
        )

        print(f"[Ollama] Starting inference for model '{self.model_name}' (stage={stage})...")
        print(f"[Ollama] State context:\n{state_context}")
        print(f"[Ollama] Latest patient message: '{latest_patient_msg}'")

        try:
            raw = self._call_ollama(SYSTEM_PROMPT, prompt, temperature=0.0, num_predict=600)
        except Exception as e:
            print(f"[Ollama] ERROR: {e}")
            return CombinedOutput.model_validate_json(current_json)

        try:
            parsed = self._parse_json(raw)

            # Coerce list → comma-string, empty/"null" → None
            for field in ["chief_complaint"] + HPI_FIELDS:
                v = parsed.get(field)
                if isinstance(v, list):
                    parsed[field] = ", ".join(str(x) for x in v) if v else None
                elif v is not None and str(v).strip() in ("", "null"):
                    parsed[field] = None

            result = CombinedOutput.model_validate(parsed)

            # ── FIX 2: Preserve already-filled fields (prevent model amnesia) ──
            prev = CombinedOutput.model_validate_json(current_json)
            for f in ["chief_complaint"] + HPI_FIELDS:
                if getattr(result, f) is None and getattr(prev, f) is not None:
                    object.__setattr__(result, f, getattr(prev, f))
            # ROS only ever grows
            merged_ros = {**prev.ros, **result.ros}
            object.__setattr__(result, "ros", merged_ros)

            # Log the reasoning so we can debug (stored as result.reasoning via alias)
            if result.reasoning:
                print(f"[Reasoning] {result.reasoning[:300]}")

            return result

        except Exception as e:
            print(f"[Ollama] JSON parse error: {e}\nRaw: {raw[:300]}")
            try:
                base = CombinedOutput.model_validate_json(current_json)
                object.__setattr__(base, "reply", "Could you please repeat that?")
                return base
            except Exception:
                return CombinedOutput(reply="Could you please repeat that?")

    def generate_brief_narrative(self, brief_data: dict) -> str:
        """
        FIX 3: Second LLM call that generates a proper clinical narrative
        instead of copy-pasting patient words verbatim.
        """
        cc = brief_data.get("chief_complaint", "unspecified")
        hpi = brief_data.get("hpi", {})
        ros = brief_data.get("ros", {})

        user_prompt = (
            f"Chief complaint: {cc}\n"
            f"HPI — Onset: {hpi.get('onset')}, Location: {hpi.get('location')}, "
            f"Duration: {hpi.get('duration')}, Character: {hpi.get('character')}, "
            f"Severity: {hpi.get('severity')}/10, "
            f"Aggravating: {hpi.get('aggravating')}, Relieving: {hpi.get('relieving')}\n"
            f"ROS: {json.dumps(ros)}\n\n"
            "Write a concise clinical narrative (3-5 sentences, present tense, third person singular). "
            "Use clinical language. Do not invent facts. "
            'Return JSON: {"narrative": "..."}'
        )

        try:
            raw = self._call_ollama(BRIEF_SYSTEM_PROMPT, user_prompt, temperature=0.1, num_predict=300)
            parsed = self._parse_json(raw)
            return parsed.get("narrative", "")
        except Exception as e:
            print(f"[Ollama] Brief narrative generation failed: {e}")
            # Graceful fallback — structured plain-text summary
            parts = [f"Patient presents with {cc}."]
            if hpi.get("onset"):
                parts.append(f"Symptoms began {hpi['onset']}.")
            if hpi.get("location"):
                parts.append(f"Located at: {hpi['location']}.")
            if hpi.get("character") and hpi.get("severity"):
                parts.append(f"Described as {hpi['character']}, severity {hpi['severity']}/10.")
            if hpi.get("aggravating"):
                parts.append(f"Aggravated by {hpi['aggravating']}; relieved by {hpi.get('relieving', 'unspecified')}.")
            return " ".join(parts)


_llm_instance = None

def get_llm():
    global _llm_instance
    if _llm_instance is None:
        mock_mode = os.environ.get("MOCK_LLM", "true").lower() == "true"
        _llm_instance = MockLLM() if mock_mode else OllamaLLM()
    return _llm_instance