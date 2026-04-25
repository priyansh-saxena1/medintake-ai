import os


class MockLLM:
    """Mock LLM for testing - returns hardcoded clinical responses."""

    def __init__(self):
        self.hpi_fields = ["onset", "location", "duration", "character", "severity", "aggravating", "relieving"]
        self.current_hpi_index = 0
        self.ros_systems_done = False
        self.ros_current_system = 0

    def generate_response(self, conversation_history: list[dict], current_node: str) -> str:
        if current_node == "intake":
            return "I have chest pain since this morning"

        if current_node == "hpi":
            responses = [
                "It started about 3 hours ago",
                "In the center of my chest",
                "It has been constant",
                "It feels like pressure",
                "About a 7 out of 10",
                "It gets worse when I walk",
                "Resting helps a little"
            ]
            if self.current_hpi_index < len(responses):
                response = responses[self.current_hpi_index]
                self.current_hpi_index += 1
                return response
            return "I already answered all those questions"

        if current_node == "ros":
            if not self.ros_systems_done:
                self.ros_systems_done = True
                return "cardiac:palpitations present,no syncope|respiratory:mild shortness of breath,no cough"
            return "done"

        return ""

    def reset(self):
        self.current_hpi_index = 0
        self.ros_systems_done = False
        self.ros_current_system = 0


class RealLLM:
    """Real LLM using llama-cpp-python with lazy loading."""

    def __init__(self):
        self.model = None
        self.model_path = "/models/qwen2.5-0.5b-instruct-q4_k_m.gguf"

    def _load_model(self):
        if self.model is None:
            from llama_cpp import Llama
            self.model = Llama(
                model_path=self.model_path,
                n_ctx=2048,
                n_threads=4
            )

    def generate_response(self, conversation_history: list[dict], current_node: str) -> str:
        self._load_model()

        system_prompt = (
            "You are a clinical AI assistant conducting patient intake. "
            "Ask one question at a time. Be concise and professional."
        )

        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(conversation_history)

        output = self.model.create_chat_completion(messages, max_tokens=256)
        return output["choices"][0]["message"]["content"]


def get_llm():
    mock_mode = os.environ.get("MOCK_LLM", "false").lower() == "true"
    if mock_mode:
        return MockLLM()
    return RealLLM()
