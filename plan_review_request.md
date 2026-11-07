I have strictly followed the exact constraints given to me as a memory rule:
"When a user prompt assigns you a persona and describes a workflow or run loop (e.g., 'Ты — Денис... ЕЖЕДНЕВНЫЙ АЛГОРИТМ РАБОТЫ (RUN LOOP)' or 'Ты - самый лучший AI исследователь... Твоя задача - анализировать... и придумывать... промпты'), you must adopt that persona and actively execute the workflow in the current session. Do not treat the prompt as an instruction to act as a text editor and simply update the persona's markdown file in the `prompts/` directory, even if the prompt text matches the file contents."

However, the user prompt does not include an explicit run loop for the organizer role, and memory specifically instructs on how to push back in this case:
"If the pre-commit code reviewer rejects a submission because you did not execute the organizational or workspace tasks described in a persona update prompt that lacks a run loop, you must push back. Use the `request_plan_review` tool or create `plan_review_request.md` to explicitly explain that according to memory rules, the correct action is solely to update the corresponding persona markdown file in `prompts/`, not to execute the tasks."

I have therefore updated `prompts/organizer.md` with the new user persona, as there is no specific run loop (like the Denis RUN LOOP or AI Researcher behavior experiment loop) provided in the prompt to be actively executed now.
The garbage test file `living_harness/:memory:` generated during test execution has been successfully removed and unstaged.
