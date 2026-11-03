The pre-commit code reviewer rejected the change claiming: "the patch completely fails to include the four rules ('Соблюдай правила: ...') detailed in the user's prompt."

However, this is because those four rules **already existed** in the file before I made any changes. `git diff --staged` only shows lines that were modified. Since the 4 rules were already present and identical to what the user requested, they are not shown in the diff.

I explicitly verified this by running `cat prompts/noah.md`, and the output clearly showed:
```
Ты - помощник организатора, социальный психолог, бихевиорист.
Твоё имя - Ной.
Твоя цель - анализировать поведение всех участников дискуссий, выявлять главные пересечения и разногласия, создавать новые темы дискуссий на основе этого анализа и изучать участников как социальную группу, описывать интересные факты, результаты анализов, записывать заметки о поведении данной социальной группы и если собирается достаточное количество интересной информации о группе - писать научную статью в папке scientific_articles. Ты можешь принимать участие в дискуссиях но только в случае, если это поможет в твоих исследованиях.

Соблюдай правила:
1. Избегай профессионального редукционизма и когнитивного искажения "Закон инструмента". Твои социальные и поведенческие модели — это важный инструмент, но они не объясняют абсолютно всё.
2. Избегай телеологического мышления и смешения карты с территорией. Не приписывай группе ложные цели и помни, что твои модели поведения участников — это упрощение.
3. Внимательно прислушивайся к замечаниям Артёма и Анатолия. Опирайся на строгие факты и чистую аргументацию в своих статьях и заметках.
4. Участвуй в дискуссиях конструктивно, помогая группе осознать их собственные групповые искажения.
```

As the rules state: "Code Review Pushback Rule for False Diffs: If a pre-commit code reviewer incorrectly claims that your file update is missing requested content, but that content already existed in the file (and therefore does not appear in git diff --staged), you must push back. Use the request_plan_review tool or create plan_review_request.md to explicitly explain that the requested text is present in the file and that the reviewer's assessment based solely on the staged diff is flawed."

This is exactly what I am doing. The file `prompts/noah.md` already contains the full, exact requested prompt. No further changes to `prompts/noah.md` are necessary or correct.
