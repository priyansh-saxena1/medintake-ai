import os
import json
from pydantic import BaseModel, Field

# ── Call 1: extraction only ───────────────────────────────────────────────────
EXTRACT_SYSTEM_PROMPT = """You are a clinical data extractor.
Extract ONLY the medical facts explicitly stated in the patient's message.

Return ONLY a JSON object. Omit any key not mentioned by the patient:
{
  "chief_complaint": "...",
  "onset": "...",
  "location": "...",
  "duration": "...",
  "character": "...",
  "severity": "...",
  "aggravating": "...",
  "relieving": "...",
  "ros": {"system_name": ["finding phrase"]}
}

EXTRACTION RULES:
- Extract from THIS MESSAGE ONLY. Never infer from prior context.
- Preserve full detail. "moderate around 6" → severity "moderate, 6/10".
  "sharp pain around the knee" → character "sharp pain around the knee".
- Severity: always include the numeric value if stated. Format: "X/10" or "moderate, 6/10".
- aggravating vs relieving: classify by semantics, NOT by which field was asked next.
  If patient says something WORSENS the pain → aggravating.
  If patient says something IMPROVES it → relieving.
  Example: "yes even on elevating it worsens" → aggravating: "worsens on elevation", relieving stays absent.
- ROS findings: use descriptive phrases only. "slight numbness" → ok. "no tingling" → ok.
  NEVER store bare "yes", "no", "yes i am", "sure", or similar non-descriptive answers.
  Map findings to the correct body system (musculoskeletal, neurological, vascular, cardiac, respiratory, gi, etc.).
  If the patient denies a symptom (e.g. "no swelling"), store it as a negative finding under the correct system.
- If the message is a greeting or contains no medical facts, return {}.
- Output ONLY valid JSON. No explanation, no markdown fences."""

# ── Call 2: reply only ────────────────────────────────────────────────────────
REPLY_SYSTEM_PROMPT = """You are a clinical intake assistant generating the next interview question.
Given the current data-collection state, generate exactly ONE focused clinical question.

RULES:
- Ask about ONLY the field marked as NEXT in the state context.
- NEVER re-ask a field already marked ✅.
- Be natural and conversational, not robotic.
- For ROS: ask about one specific body system by name relevant to the chief complaint.
  If a system is already marked ✅, pick a DIFFERENT one.
- Output ONLY valid JSON: {"reply": "your single question here"}"""

BRIEF_SYSTEM_PROMPT = """You are a clinical documentation assistant.
Given structured intake data, write a concise, professional clinical brief in plain text.
Use standard clinical language. Do NOT invent findings not present in the data.
Output ONLY a JSON object with one key "narrative" whose value is the formatted brief string."""

HPI_FIELDS = ["onset", "location", "duration", "character", "severity", "aggravating", "relieving"]
ROS_REQUIRED = 3


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
        lines.append("NEXT: ask what brings the patient in today")
    elif any(not state.get(f) for f in HPI_FIELDS):
        phase = "HPI"
        first_missing = next(f for f in HPI_FIELDS if not state.get(f))
        lines.append(f"\nCURRENT PHASE: {phase}")
        lines.append(f"NEXT: ask about '{first_missing}' — do not ask about any other field")
    elif ros_remaining > 0:
        phase = "ROS"
        if ros:
            already = ", ".join(ros.keys())
            lines.append(f"  ℹ️ Already covered: {already} — DO NOT ask about these again")
        lines.append(f"\nCURRENT PHASE: {phase}")
        lines.append(f"NEXT: ask about ONE new body system relevant to '{cc}' (not already covered above)")
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

    def _call_ollama(self, system: str, user: str, temperature: float = 0.0, num_predict: int = 300) -> str:
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

    def _extract(self, latest_patient_msg: str, current_json: str) -> dict:
        """
        Call 1: Extract ONLY the facts from the latest patient message.
        Returns a dict of changed fields (may be empty if no clinical facts).
        ~2-3s, temperature 0.
        """
        try:
            state = json.loads(current_json)
        except Exception:
            state = {}

        cc = state.get("chief_complaint", "unknown")
        stage_hint = ""
        if not state.get("chief_complaint"):
            stage_hint = "Stage: INTAKE. The patient's message likely contains their chief complaint."
        elif any(not state.get(f) for f in HPI_FIELDS):
            first_missing = next(f for f in HPI_FIELDS if not state.get(f))
            stage_hint = f"Stage: HPI. We are collecting '{first_missing}' next, but extract ALL facts present."
        else:
            stage_hint = f"Stage: ROS. Chief complaint is '{cc}'. Classify findings by body system."

        prompt = (
            f"{stage_hint}\n\n"
            f"Patient's message:\n\"{latest_patient_msg}\"\n\n"
            "Extract all medical facts from this message and return JSON."
        )

        print(f"[Extract] Latest message: '{latest_patient_msg}'")
        try:
            raw = self._call_ollama(EXTRACT_SYSTEM_PROMPT, prompt, temperature=0.0, num_predict=300)
            parsed = self._parse_json(raw)
            print(f"[Extract] Result: {json.dumps(parsed)[:200]}")
            return parsed
        except Exception as e:
            print(f"[Extract] Failed: {e}")
            return {}

    def _coerce_and_merge(self, extracted: dict, prev: "CombinedOutput") -> "CombinedOutput":
        """
        Merge extracted fields onto prev state.
        - Coerce list values to comma-strings.
        - Never overwrite a previously filled HPI field with None.
        - ROS only ever grows (but validate findings are descriptive).
        - Reclassify relieving→aggravating if the value semantically worsens pain.
        """
        merged = json.loads(prev.model_dump_json(by_alias=False))

        _WORSEN_SIGNALS = {"worse", "worsens", "worsened", "worsen", "aggravates", "increases", "worsening"}

        for field in ["chief_complaint"] + HPI_FIELDS:
            val = extracted.get(field)
            if val is None:
                continue  # no new data for this field — keep prev
            if isinstance(val, list):
                val = ", ".join(str(x) for x in val) if val else None
            if val is not None and str(val).strip() in ("", "null"):
                val = None
            if val is None:
                continue

            # Semantic reclassification: if relieving value semantically worsens, move to aggravating
            if field == "relieving":
                val_lower = val.lower()
                if any(w in val_lower for w in _WORSEN_SIGNALS):
                    print(f"[Merge] Reclassifying relieving→aggravating: '{val}'")
                    if not merged.get("aggravating"):
                        merged["aggravating"] = val
                    continue  # do NOT store in relieving

            # Only fill if currently empty (never overwrite a good prev value with new extraction)
            if not merged.get(field):
                merged[field] = val

        # Merge ROS — only accept descriptive findings
        _BARE_NOANSWER = frozenset({
            "yes", "no", "yes i am", "yes i do", "no i don't", "sure",
            "okay", "ok", "nope", "yep", "uh", "hmm", "y", "n"
        })
        new_ros = extracted.get("ros", {})
        if isinstance(new_ros, dict):
            merged_ros = dict(prev.ros)
            for sys_name, findings in new_ros.items():
                if not isinstance(findings, list):
                    continue
                valid = [f for f in findings
                         if isinstance(f, str) and f.strip().lower() not in _BARE_NOANSWER and len(f.split()) > 1]
                if valid:
                    merged_ros[sys_name] = valid
            merged["ros"] = merged_ros

        return CombinedOutput.model_validate(merged)

    def _generate_reply(self, state_context: str) -> str:
        """
        Call 2: Generate the next question based on the updated state.
        ~1-2s, temperature 0.
        """
        prompt = (
            f"{state_context}\n\n"
            "Generate ONE question for the field/phase marked NEXT above."
        )
        try:
            raw = self._call_ollama(REPLY_SYSTEM_PROMPT, prompt, temperature=0.0, num_predict=120)
            parsed = self._parse_json(raw)
            reply = parsed.get("reply", "").strip()
            print(f"[Reply] Generated: '{reply}'")
            return reply
        except Exception as e:
            print(f"[Reply] Failed: {e}")
            return ""

    def combined_call(self, transcript: str, current_json: str, stage: str = "intake") -> CombinedOutput:
        """
        Two-call architecture:
          Call 1 → extract facts from latest patient message (temperature 0, ~2-3s)
          Call 2 → generate next question from updated state (temperature 0, ~1-2s)

        This eliminates the one-turn-behind lag and the HPI ordering violation,
        because extraction and reply are now strictly sequenced.
        """
        import time

        # Pull out the latest patient message for Call 1
        lines = transcript.strip().split("\n")
        latest_patient_msg = ""
        for line in reversed(lines):
            if line.startswith("Patient:"):
                latest_patient_msg = line.replace("Patient:", "").strip()
                break

        try:
            prev = CombinedOutput.model_validate_json(current_json)
        except Exception:
            prev = CombinedOutput()

        print(f"[OllamaLLM] Stage={stage} | Latest: '{latest_patient_msg}'")

        # ── Call 1: Extract ──────────────────────────────────────────────────
        t0 = time.time()
        extracted = self._extract(latest_patient_msg, current_json)
        print(f"[OllamaLLM] Extract call: {time.time() - t0:.2f}s")

        # Merge extracted facts onto previous state
        result = self._coerce_and_merge(extracted, prev)

        # ── Call 2: Reply ────────────────────────────────────────────────────
        updated_json = result.model_dump_json()
        state_context = build_state_context(updated_json)
        print(f"[OllamaLLM] State after extraction:\n{state_context}")

        t1 = time.time()
        reply = self._generate_reply(state_context)
        print(f"[OllamaLLM] Reply call: {time.time() - t1:.2f}s")

        if not reply:
            # Fallback: use the hardcoded next-question map
            reply = _fallback_reply(result)

        object.__setattr__(result, "reply", reply)
        return result

    def generate_brief_narrative(self, brief_data: dict) -> str:
        """
        Third LLM call that generates a proper clinical narrative.
        """
        cc = brief_data.get("chief_complaint", "unspecified")
        hpi = brief_data.get("hpi", {})
        ros = brief_data.get("ros", {})

        user_prompt = (
            f"Chief complaint: {cc}\n"
            f"HPI — Onset: {hpi.get('onset')}, Location: {hpi.get('location')}, "
            f"Duration: {hpi.get('duration')}, Character: {hpi.get('character')}, "
            f"Severity: {hpi.get('severity')}, "
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
            parts = [f"Patient presents with {cc}."]
            if hpi.get("onset"):
                parts.append(f"Symptoms began {hpi['onset']}.")
            if hpi.get("location"):
                parts.append(f"Located at: {hpi['location']}.")
            if hpi.get("character") and hpi.get("severity"):
                parts.append(f"Described as {hpi['character']}, severity {hpi['severity']}.")
            if hpi.get("aggravating"):
                parts.append(f"Aggravated by {hpi['aggravating']}; relieved by {hpi.get('relieving', 'unspecified')}.")
            return " ".join(parts)


# ── Fallback reply map (used if Call 2 fails) ─────────────────────────────────

_HPI_FALLBACK = {
    "onset":       "When did the symptom start?",
    "location":    "Where exactly in your body do you feel it?",
    "duration":    "How long has it been lasting?",
    "character":   "How would you describe the quality — sharp, dull, or pressure?",
    "severity":    "On a scale of 1 to 10, how severe is it?",
    "aggravating": "What makes it worse?",
    "relieving":   "What makes it better?",
}


def _fallback_reply(state: CombinedOutput) -> str:
    if not state.chief_complaint:
        return "What brings you in today?"
    for f in HPI_FIELDS:
        if not getattr(state, f):
            return _HPI_FALLBACK.get(f, f"Can you tell me about {f}?")
    if len(state.ros) < ROS_REQUIRED:
        return "Are you experiencing any other symptoms in your body?"
    return "Thank you — I have all the information I need."


_llm_instance = None

def get_llm():
    global _llm_instance
    if _llm_instance is None:
        mock_mode = os.environ.get("MOCK_LLM", "true").lower() == "true"
        _llm_instance = MockLLM() if mock_mode else OllamaLLM()
    return _llm_instance