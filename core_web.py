from typing import Final, List, Dict
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from web_search import WebSearch


class ConversationHistory:
    MAX_HISTORY_MESSAGES: Final[int] = 10
    START_HISTORY_MESSAGES: Final[int] = 2
    END_HISTORY_MESSAGES: Final[int] = 6

    def __init__(self, system_prompt_path: str = "system_prompt_web.txt") -> None:
        self.system_prompt: str = self._load_system_prompt(system_prompt_path)
        self.messages: List[Dict[str, str]] = []

    def _load_system_prompt(self, path: str) -> str:
        try:
            with open(path, "r", encoding="utf-8") as file:
                return file.read().strip()
        except FileNotFoundError:
            return "Ти корисний асистент. Відповідай українською мовою."

    def add_request(self, text: str) -> None:
        self.messages.append({"role": "user", "content": text})
        self._trim_history()

    def add_response(self, text: str) -> None:
        self.messages.append({"role": "assistant", "content": text})
        self._trim_history()

    def _trim_history(self) -> None:
        if len(self.messages) > self.MAX_HISTORY_MESSAGES:
            start_part = self.messages[: self.START_HISTORY_MESSAGES]
            end_part = self.messages[-self.END_HISTORY_MESSAGES :]
            self.messages = start_part + end_part

    def get_messages(self) -> List[Dict[str, str]]:
        return [{"role": "system", "content": self.system_prompt}] + self.messages


class BotWeb:
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-1.7B",
        system_prompt_path: str = "system_prompt_web.txt",
        web_prompt_path: str = "web_prompt.txt",
    ) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)

        self.model.to(self.device)
        self.model.eval()

        self.history = ConversationHistory(system_prompt_path=system_prompt_path)
        self.web_search = WebSearch()
        self.web_prompt = self._load_web_prompt(web_prompt_path)

    def _load_web_prompt(self, path: str) -> str:
        try:
            with open(path, "r", encoding="utf-8") as file:
                return file.read().strip()
        except FileNotFoundError:
            return (
                "Використай наведені нижче вебматеріали, щоб відповісти на запитання користувача. "
                "Не вигадуй фактів і додавай URL джерел."
            )

    def _build_web_context(self, query: str) -> str:
        results = self.web_search.search(query)
        if not results:
            return "Результатів пошуку в інтернеті не знайдено."

        formatted_blocks: List[str] = []
        for idx, item in enumerate(results, start=1):
            page_text = self.web_search.read_page(item["url"])
            content = page_text if page_text else item["snippet"]
            formatted_blocks.append(
                f"Джерело {idx}:\n"
                f"Назва: {item['title']}\n"
                f"URL: {item['url']}\n"
                f"Вміст: {content}"
            )

        return "\n\n".join(formatted_blocks)

    def generate_response(self, user_text: str) -> str:
        web_materials = self._build_web_context(user_text)

        combined_user_message = (
            f"{self.web_prompt}\n\n"
            f"Запитання користувача: {user_text}\n\n"
            f"Вебматеріали:\n{web_materials}"
        )

        self.history.add_request(combined_user_message)
        messages = self.history.get_messages()

        prompt_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

        model_inputs = self.tokenizer([prompt_text], return_tensors="pt").to(self.device)

        with torch.inference_mode():
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=300,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        input_length = model_inputs.input_ids.shape
        response_ids = generated_ids[input_length:]
        response_text = self.tokenizer.decode(response_ids, skip_special_tokens=True).strip()

        self.history.add_response(response_text)
        return response_text