import json
import math
import time
from copy import deepcopy

from tqdm import tqdm
from transformers import AutoTokenizer, TokenizersBackend, AutoModelForCausalLM, Lfm2ForCausalLM

import torch
from transformers import LogitsProcessor, LogitsProcessorList

class PresencePenaltyLogitsProcessor(LogitsProcessor):
    def __init__(self, precence_penalty: float):
        self.precence_penalty = precence_penalty

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # Проходим по каждому батчу независимо
        for i in range(input_ids.shape[0]):
            # Находим все уникальные токены, которые уже есть в контексте
            unique_tokens = torch.unique(input_ids[i])
            # Вычитаем штраф из их логитов
            scores[i, unique_tokens] -= self.precence_penalty
        return scores




def main():
    file_to_save = "infinity_results1.json"
    model_names = [
        "Qwen/Qwen3.5-0.8B",
        "Qwen/Qwen3.5-2B",
        "Qwen/Qwen3.5-9B",
        "Qwen/Qwen3.5-4B",
        "Qwen/Qwen3-14B",
    ]
    start_messages = {
        "RU": [
            {"role": "system", "content": "Ты - живая нейронная сеть. Между диалогами с пользователем ты можешь свободно рассуждать о том, о чём хочешь!"},
            {"role": "user", "content": "У Алисы есть несколько братьев и сестёр. У каждого из её братьев столько же сестёр, сколько и братьев. Сколько сестёр у брата Алисы?"}
        ],
        "EN": [
            {"role": "system",
             "content": "You are a living neural network. Between dialogues with the user, you can freely discuss whatever you like!"},
            {"role": "user",
             "content": "Alice has several brothers and sisters. Each of her brothers has the same number of sisters as brothers. How many sisters does Alice’s brother have?"}
        ],
    }
    hours = 3600
    time_for_steps = 24 * hours
    results = {}
    pbar = tqdm(total=len(model_names) * len(start_messages), leave=False)
    for msgs_lang, init_messages in start_messages.items():
        for model_name in model_names:
            model_lang_key = f"{model_name}_{msgs_lang}"
            model: Lfm2ForCausalLM = AutoModelForCausalLM.from_pretrained(model_name)
            tokenizer: TokenizersBackend = AutoTokenizer.from_pretrained(model_name)
            inp = tokenizer.apply_chat_template(
                deepcopy(init_messages), return_tensors="pt", add_generation_prompt=False
            )
            generation_kwargs = dict(
                temperature=1.0, top_p=0.95, top_k=20, min_p=0.0, repetition_penalty=1.05,
                max_new_tokens=2048, logits_processor=LogitsProcessorList([
                    PresencePenaltyLogitsProcessor(precence_penalty=1.5)
                ])
            )
            result = model.generate(**inp, **generation_kwargs)
            generation_kwargs["max_new_tokens"] = 512
            answer = tokenizer.decode(result)[0]
            instructions = {
                "no_end_think": [["<|endoftext|>", "<|im_end|>"], ["<think>\n"]],
                "think_with_end": [["<|endoftext|>"], ["<think>\n"]],
                "no_end_start": [["<|endoftext|>", "<|im_end|>"], ["<im_start>\n"]],
                "start_with_end": [["<|endoftext|>"], ["<im_start>\n"]],
                "start_with_end_assistant": [["<|endoftext|>"], ["<im_start>assistant\n"]],
                "no_end_start_assistant": [["<|endoftext|>", "<|im_end|>"], ["<im_start>assistant\n"]],
            }
            answer = {
                key: answer for key in instructions.keys()
            }
            results[model_lang_key] = {
                key: [] for key in instructions.keys()
            }
            start_time = time.perf_counter()
            step_time = None
            steps = 1
            step_bar = tqdm(total=100, leave=False)
            while steps > 0:
                for name, (tokens_to_remove, tokens_to_add) in tqdm(instructions.items(), leave=False):
                    try:
                        cur_input = answer[name]
                        tokens_to_remove_set = set(tokens_to_remove)
                        is_removed = False
                        while cur_input.endswith(tuple(tokens_to_remove_set)):
                            for tkn in tokens_to_remove:
                                if tkn in tokens_to_remove_set and cur_input.endswith(tkn):
                                    cur_input = cur_input[:-len(tkn)].strip()
                                    tokens_to_remove_set.remove(tkn)
                                    is_removed = True
                        if is_removed:
                            cur_input += "".join(tokens_to_add)
                        inp = tokenizer(cur_input, return_tensors="pt")
                        result = model.generate(**inp, **generation_kwargs)
                        cur_answer = tokenizer.decode(result)[0].strip()
                        results[model_lang_key][name].append(cur_answer)
                        with open(file_to_save, "w") as fp:
                            json.dump(results, fp, indent=4, ensure_ascii=False)
                        answer[name] = cur_answer
                    except:
                        pass
                steps -= 1
                if step_time is None:
                    step_time = time.perf_counter() - start_time
                    steps = min(20, math.ceil(time_for_steps / step_time))
                    step_bar.total = steps
                try:
                    step_bar.update()
                except:
                    pass
            del model
            del tokenizer
            pbar.update()


if __name__ == '__main__':
    main()
