import re
from typing import Tuple, Optional

class ReasoningParser:
    """
    Парсер для обработки вывода reasoning-моделей (например, дистилляций Qwen-R1 и SmolLM2-Rethink).
    Извлекает внутренние рассуждения из тегов <think>...</think> и отделяет их от финального ответа.
    """

    @staticmethod
    def parse(text: str) -> Tuple[Optional[str], str]:
        """
        Извлекает блок рассуждений и финальный текст ответа.

        Args:
            text: Сгенерированный текст от LLM.

        Returns:
            Tuple[thought, final_answer]
        """
        pattern = r'<think>(.*?)</think>'
        matches = re.findall(pattern, text, flags=re.DOTALL | re.IGNORECASE)

        if not matches:
            # Попробуем найти незакрытый тег (оборванная генерация)
            open_match = re.search(r'<think>(.*)', text, flags=re.DOTALL | re.IGNORECASE)
            if open_match:
                return open_match.group(1).strip(), ""
            return None, text.strip()

        thought = matches[0].strip()
        final_answer = re.sub(pattern, '', text, flags=re.DOTALL | re.IGNORECASE).strip()

        return thought, final_answer
