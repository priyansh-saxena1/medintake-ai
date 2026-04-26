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
- aggravating vs relieving: classify by semantics, NOT by which field was asked.
  If patient says something WORSENS the pain → aggravating.
  If patient says something IMPROVES it → relieving.
- ROS findings: use descriptive phrases only. NEVER store bare "yes", "no", "yes i am", "sure".
  Map findings to the correct body system.
  Denials count: "no tingling" → neurological: ["no tingling"].
- NON-CLINICAL MESSAGES: If the message is a greeting, question, or expression of frustration
  ("stop asking", "I already said", "move on", "hello", "ok go ahead"), return {}.
  Do NOT invent clinical findings from non-clinical messages.
- Output ONLY valid JSON. No explanation, no markdown fences."""

# ── Call 2: reply only ────────────────────────────────────────────────────────
REPLY_SYSTEM_PROMPT = """You are a clinical intake assistant generating the next interview question.
The exact field or system to ask about is specified in the NEXT line. A SUGGESTED QUESTION may be provided.

RULES:
- Ask ONLY about what is specified in NEXT.
- Use the SUGGESTED QUESTION as a template; rephrase naturally if needed.
- Be concise. One sentence.
- Output ONLY valid JSON: {"reply": "your question"}"""

BRIEF_SYSTEM_PROMPT = """You are a clinical documentation assistant.
Given structured intake data, write a concise, professional clinical brief in plain text.
Use standard clinical language. Do NOT invent findings not present in the data.
Output ONLY a JSON object with one key "narrative" whose value is the formatted brief string."""

HPI_FIELDS = ["onset", "location", "duration", "character", "severity", "aggravating", "relieving"]
ROS_REQUIRED = 3

# ── ROS system catalogue ──────────────────────────────────────────────────────

_ROS_BY_CC = {
    "leg":       ["musculoskeletal", "neurological", "vascular"],
    "knee":      ["musculoskeletal", "neurological", "vascular"],
    "ankle":     ["musculoskeletal", "neurological", "vascular"],
    "foot":      ["musculoskeletal", "neurological", "vascular"],
    "arm":       ["musculoskeletal", "neurological", "vascular"],
    "shoulder":  ["musculoskeletal", "neurological", "vascular"],
    "back":      ["musculoskeletal", "neurological", "urinary"],
    "chest":     ["cardiovascular", "respiratory", "gastrointestinal"],
    "abdomen":   ["gastrointestinal", "urinary", "cardiovascular"],
    "stomach":   ["gastrointestinal", "urinary", "cardiovascular"],
    "head":      ["neurological", "ent", "cardiovascular"],
    "headache":  ["neurological", "ent", "cardiovascular"],
}

_ROS_QUESTIONS = {
    "musculoskeletal": "Have you noticed any swelling, stiffness, or difficulty moving the affected area?",
    "neurological":    "Any numbness, tingling, or weakness in the affected area?",
    "vascular":        "Any changes in skin color, temperature, or pulse in the limb?",
    "cardiovascular":  "Any palpitations, chest tightness, or racing heartbeat?",
    "respiratory":     "Any shortness of breath or difficulty breathing?",
    "gastrointestinal":"Any nausea, vomiting, or abdominal pain?",
    "ent":             "Any ringing in the ears, dizziness, or visual changes?",
    "urinary":         "Any changes in urination or pain when urinating?",
    "integumentary":   "Any skin changes such as redness, warmth, or rash near the area?",
}

_ROS_GENERIC_ORDER = [
    "musculoskeletal", "neurological", "vascular",
    "cardiovascular", "respiratory", "gastrointestinal",
]


def _next_ros_system(cc: str, already_covered: set) -> str | None:
    """Deterministically pick the next ROS system. Ignores patient_reported_N placeholders."""
    cc_lower = (cc or "").lower()
    preferred = None
    for keyword, sys_list in _ROS_BY_CC.items():
        if keyword in cc_lower:
            preferred = sys_list
            break

    candidates = list(preferred or []) + [s for s in _ROS_GENERIC_ORDER if s not in (preferred or [])]
    # Exclude placeholder keys from "already covered"
    real_covered = {k for k in already_covered if not k.startswith("patient_reported_")}

    for sys in candidates:
        if sys not in real_covered:
            return sys
    return None


def build_state_context(current_json: str) -> str:
    """Build a human-readable status block for the LLM, including explicit NEXT field."""
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
    real_ros = {k: v for k, v in ros.items() if not k.startswith("patient_reported_")}
    for sys_name, findings in real_ros.items():
        lines.append(f'  ✅ ros.{sys_name}: {findings}')
    ros_remaining = ROS_REQUIRED - len(real_ros)
    if ros_remaining > 0:
        lines.append(f"  ❌ ros: {ros_remaining} more system(s) needed")
    else:
        lines.append(f"  ✅ ros: all {ROS_REQUIRED} systems collected")

    if not cc:
        lines.append("\nCURRENT PHASE: INTAKE")
        lines.append("NEXT: ask what brings the patient in today")
    elif any(not state.get(f) for f in HPI_FIELDS):
        first_missing = next(f for f in HPI_FIELDS if not state.get(f))
        lines.append(f"\nCURRENT PHASE: HPI")
        lines.append(f"NEXT: ask about '{first_missing}' — do not ask about any other field")
    elif ros_remaining > 0:
        already = set(real_ros.keys())
        if already:
            lines.append(f"  ℹ️ Already covered: {', '.join(sorted(already))} — DO NOT revisit these")
        next_sys = _next_ros_system(cc, set(ros.keys()))
        lines.append("\nCURRENT PHASE: ROS")
        if next_sys:
            suggestion = _ROS_QUESTIONS.get(next_sys, f"Any symptoms in the {next_sys} system?")
            lines.append(f"NEXT: ask about '{next_sys}' system")
            lines.append(f"SUGGESTED QUESTION: {suggestion}")
        else:
            lines.append("NEXT: ask about any uncovered body system relevant to the chief complaint")
    else:
        lines.append("\nCURRENT PHASE: DONE — all data collected")

    return "\n".join(lines)


class CombinedOutput(BaseModel):
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
            for field in HPI_FIELDS:
                if not state.get(field):
                    if last_patient_msg:
                        state[field] = last_patient_msg
                    break
            labels = {"onset": "when it started", "location": "where you feel it",
                      "duration": "how long it's lasted", "character": "what it feels like",
                      "severity": "how severe it is (1-10)", "aggravating": "what makes it worse",
                      "relieving": "what makes it better"}
            for field in HPI_FIELDS:
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
        cc = brief_data.get("chief_complaint", "unspecified")
        hpi = brief_data.get("hpi", {})
        ros = brief_data.get("ros", {})
        parts = [
            f"Patient presents with {cc}.",
            f"Symptoms began {hpi.get('onset', 'at an unspecified time')},",
            f"localised to the {hpi.get('location', 'unspecified area')}.",
            f"Duration: {hpi.get('duration', 'not specified')}.",
            f"Character: {hpi.get('character', 'not specified')}.",
            f"Severity: {hpi.get('severity', 'not rated')}.",
            f"Aggravated by {hpi.get('aggravating', 'unspecified')}; relieved by {hpi.get('relieving', 'unspecified')}.",
        ]
        if ros:
            ros_lines = [f"{s.capitalize()}: {', '.join(f)}." for s, f in ros.items()]
            parts.append("Review of systems: " + " ".join(ros_lines))
        return " ".join(parts)


class OllamaLLM:
    def __init__(self):
        self.model_name = os.environ.get("MODEL_NAME", "qwen2.5:0.5b")
        self.api_url = "http://localhost:11434/api/chat"

    def _call_ollama(self, system: str, user: str, temperature: float = 0.0, num_predict: int = 300) -> str:
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
        s = raw
        if "```json" in s:
            s = s.split("```json", 1)[1].split("```")[0]
        elif "```" in s:
            s = s.split("```", 1)[1].split("```")[0]
        start, end = s.find("{"), s.rfind("}") + 1
        if start != -1 and end > start:
            s = s[start:end]
        return json.loads(s)

    def _extract(self, latest_patient_msg: str, current_json: str, last_asked_field: str | None = None) -> dict:
        """
        Call 1: Extract facts from the latest patient message.
        last_asked_field tells the model which field we just asked about,
        so ambiguous answers (e.g. "resting") land in the right slot.
        """
        try:
            state = json.loads(current_json)
        except Exception:
            state = {}

        cc = state.get("chief_complaint", "unknown")
        if not state.get("chief_complaint"):
            stage_hint = "Stage: INTAKE. Extract the chief complaint from this message."
        elif any(not state.get(f) for f in HPI_FIELDS):
            asked = last_asked_field or next(f for f in HPI_FIELDS if not state.get(f))
            stage_hint = (
                f"Stage: HPI. The previous question asked about '{asked}'. "
                f"The patient's answer most likely fills '{asked}'. "
                f"Also extract any other HPI facts if present."
            )
        else:
            stage_hint = (
                f"Stage: ROS. Chief complaint is '{cc}'. "
                "Classify findings by body system. Return {} for non-clinical messages."
            )

        prompt = (
            f"{stage_hint}\n\n"
            f"Patient's message:\n\"{latest_patient_msg}\"\n\n"
            "Extract all medical facts and return JSON. "
            "If the message contains no medical facts, return {}."
        )

        print(f"[Extract] Message: '{latest_patient_msg}' | asked: '{last_asked_field}'")
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
        Key rules:
        - HPI fields: only fill if currently empty (never overwrite).
        - relieving that semantically worsens pain → reroute to aggravating.
        - ROS: APPEND to existing systems, never overwrite.
        """
        merged = json.loads(prev.model_dump_json(by_alias=False))

        _WORSEN_SIGNALS = {"worse", "worsens", "worsened", "worsen", "aggravates", "increases", "worsening"}

        for field in ["chief_complaint"] + HPI_FIELDS:
            val = extracted.get(field)
            if val is None:
                continue
            if isinstance(val, list):
                val = ", ".join(str(x) for x in val) if val else None
            if val is not None and str(val).strip() in ("", "null"):
                val = None
            if val is None:
                continue

            if field == "relieving":
                if any(w in val.lower() for w in _WORSEN_SIGNALS):
                    print(f"[Merge] Reclassifying relieving→aggravating: '{val}'")
                    if not merged.get("aggravating"):
                        merged["aggravating"] = val
                    continue

            if not merged.get(field):
                merged[field] = val

        # ── ROS: append to existing, never overwrite ──────────────────────────
        _BARE_NOANSWER = frozenset({
            "yes", "no", "yes i am", "yes i do", "no i don't", "sure",
            "okay", "ok", "nope", "yep", "uh", "hmm", "y", "n",
            "i am", "i do", "i don't"
        })
        new_ros = extracted.get("ros", {})
        if isinstance(new_ros, dict):
            merged_ros = dict(prev.ros)
            for sys_name, findings in new_ros.items():
                if not isinstance(findings, list):
                    continue
                valid = [
                    f for f in findings
                    if isinstance(f, str)
                    and f.strip().lower() not in _BARE_NOANSWER
                    and len(f.split()) >= 2
                ]
                if valid:
                    existing = merged_ros.get(sys_name, [])
                    new_only = [f for f in valid if f not in existing]
                    if new_only:
                        merged_ros[sys_name] = existing + new_only
            merged["ros"] = merged_ros

        return CombinedOutput.model_validate(merged)

    def _generate_reply(self, state_context: str) -> str:
        """Call 2: Generate next question given the updated state context."""
        prompt = (
            f"{state_context}\n\n"
            "Generate ONE question for the field/system in NEXT. "
            "Use or rephrase the SUGGESTED QUESTION if provided."
        )
        try:
            raw = self._call_ollama(REPLY_SYSTEM_PROMPT, prompt, temperature=0.0, num_predict=100)
            parsed = self._parse_json(raw)
            reply = parsed.get("reply", "").strip()
            print(f"[Reply] Generated: '{reply}'")
            return reply
        except Exception as e:
            print(f"[Reply] Failed: {e}")
            return ""

    def combined_call(self, transcript: str, current_json: str, stage: str = "intake") -> CombinedOutput:
        import time

        lines = transcript.strip().split("\n")
        latest_patient_msg = ""
        last_asked_field = None

        for line in reversed(lines):
            if line.startswith("Patient:") and not latest_patient_msg:
                latest_patient_msg = line.replace("Patient:", "").strip()
            elif line.startswith("AI:") and latest_patient_msg and last_asked_field is None:
                ai_text = line.replace("AI:", "").strip().lower()
                for f, keyword in _FIELD_KEYWORDS.items():
                    if keyword in ai_text:
                        last_asked_field = f
                        break

        try:
            prev = CombinedOutput.model_validate_json(current_json)
        except Exception:
            prev = CombinedOutput()

        print(f"[OllamaLLM] Stage={stage} | Latest: '{latest_patient_msg}' | asked: '{last_asked_field}'")

        # ── Call 1: Extract ──────────────────────────────────────────────────
        t0 = time.time()
        extracted = self._extract(latest_patient_msg, current_json, last_asked_field)
        print(f"[OllamaLLM] Extract call: {time.time() - t0:.2f}s")

        result = self._coerce_and_merge(extracted, prev)

        # ── Call 2: Reply ────────────────────────────────────────────────────
        updated_json = result.model_dump_json()
        state_context = build_state_context(updated_json)
        print(f"[OllamaLLM] State after extraction:\n{state_context}")

        t1 = time.time()
        reply = self._generate_reply(state_context)
        print(f"[OllamaLLM] Reply call: {time.time() - t1:.2f}s")

        if not reply:
            reply = _fallback_reply(result)

        object.__setattr__(result, "reply", reply)
        return result

    def generate_brief_narrative(self, brief_data: dict) -> str:
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


# ── Keyword map for last-asked-field detection ────────────────────────────────
_FIELD_KEYWORDS = {
    "onset":       "start",
    "location":    "where",
    "duration":    "how long",
    "character":   "feel like",
    "severity":    "scale",
    "aggravating": "worse",
    "relieving":   "better",
}


def _fallback_reply(state: CombinedOutput) -> str:
    _HPI_FALLBACK = {
        "onset":       "When did the symptom start?",
        "location":    "Where exactly in your body do you feel it?",
        "duration":    "How long has it been lasting?",
        "character":   "How would you describe it — sharp, dull, or pressure?",
        "severity":    "On a scale of 1 to 10, how severe is it?",
        "aggravating": "What makes it worse?",
        "relieving":   "What makes it better?",
    }
    if not state.chief_complaint:
        return "What brings you in today?"
    for f in HPI_FIELDS:
        if not getattr(state, f):
            return _HPI_FALLBACK.get(f, f"Can you tell me about {f}?")
    if len(state.ros) < ROS_REQUIRED:
        next_sys = _next_ros_system(state.chief_complaint or "", set(state.ros.keys()))
        if next_sys:
            return _ROS_QUESTIONS.get(next_sys, f"Any symptoms in the {next_sys} system?")
        return "Are you experiencing any other symptoms?"
    return "Thank you — I have all the information I need."


_llm_instance = None

def get_llm():
    global _llm_instance
    if _llm_instance is None:
        mock_mode = os.environ.get("MOCK_LLM", "true").lower() == "true"
        _llm_instance = MockLLM() if mock_mode else OllamaLLM()
    return _llm_instance