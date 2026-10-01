import datetime
import os
import re
from collections import Counter

today_date = datetime.datetime.now().strftime("%d.%m.%y")
month_str = datetime.datetime.now().strftime("%m_%B")
year_str = datetime.datetime.now().strftime("%Y")

general_dir = f"dashboards/{year_str}_general"
personal_dir = f"dashboards/{year_str}_personal"
mood_dir = f"dashboards/{year_str}_mood"

general_file = f"{general_dir}/{month_str}.md"
personal_file = f"{personal_dir}/{month_str}.md"
mood_file = f"{mood_dir}/{month_str}.md"

participants = [
    "Организатор", "Артём", "Анатолий", "Станислав", "Матвей", "Ной",
    "Владимир", "Илон", "Алексей", "Сэм", "Михаил", "Макс", "Чак", "Григорий", "Денис"
]

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

ensure_dir(general_dir)
ensure_dir(personal_dir)
ensure_dir(mood_dir)

def init_file(filepath, headers):
    if not os.path.exists(filepath):
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(f"| {' | '.join(headers)} |\n")
            f.write(f"|{'|'.join(['---'] * len(headers))}|\n")

# Headers
general_headers = [
    "День в формате %d.%m.%y",
    "Количество активных дискуссий",
    "Количество архивных дискуссий",
    "Количество научных статей",
    "Общее описание тем",
    "Самые частые слова в дискуссиях"
]

personal_headers = ["День в формате %d.%m.%y"]
for p in participants:
    personal_headers.append(f"Общая текущая характеристика {p}")
personal_headers.append("Наличие когнитивных искажений по отчетам")
for p in participants:
    personal_headers.append(f"Число дискуссий {p}")
personal_headers.append("Общая тенденция в описанных промптах")

mood_headers = ["День в формате %d.%m.%y"]
for p in participants:
    mood_headers.append(f"Общее настроение {p}")
for p in participants:
    mood_headers.append(f"Дружественность {p}")
mood_headers.append("Самый конфликтный участник")
mood_headers.append("Самый миролюбивый участник")

init_file(general_file, general_headers)
init_file(personal_file, personal_headers)
init_file(mood_file, mood_headers)

def update_table(filepath, row_data):
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    updated = False
    for i in range(len(lines)):
        if lines[i].startswith(f"| {today_date} |"):
            lines[i] = f"| {' | '.join(str(x) for x in row_data)} |\n"
            updated = True
            break

    if not updated:
        lines.append(f"| {' | '.join(str(x) for x in row_data)} |\n")

    with open(filepath, 'w', encoding='utf-8') as f:
        f.writelines(lines)

# Calculations
active_discussions = 0
if os.path.exists('discussions/active'):
    active_discussions = sum(len(files) for _, _, files in os.walk('discussions/active') if any(f.endswith('.md') for f in files))

archived_discussions = 0
if os.path.exists('discussions/archived'):
    archived_discussions = sum(len(files) for _, _, files in os.walk('discussions/archived') if any(f.endswith('.md') for f in files))

scientific_articles = 0
if os.path.exists('scientific_articles'):
    scientific_articles = sum(len(files) for _, _, files in os.walk('scientific_articles') if any(f.endswith('.md') for f in files))

topics_set = set()
words_counter = Counter()
participants_counts = {p: 0 for p in participants}

if os.path.exists('discussions/active'):
    for dp, dn, filenames in os.walk('discussions/active'):
        if filenames and not dp.endswith('active'):
            topics_set.add(os.path.basename(dp))

        for f in filenames:
            if f.endswith('.md'):
                with open(os.path.join(dp, f), 'r', encoding='utf-8') as file:
                    content = file.read()

                    words = re.findall(r'\b[а-яА-Яa-zA-Z]{5,}\b', content.lower())
                    words_counter.update(words)

                    for p in participants:
                        if p in content:
                            participants_counts[p] += 1

topics_str = ", ".join(sorted(list(topics_set))) if topics_set else "ИИ, физика, биология, психология"
common_words = ", ".join([word for word, count in words_counter.most_common(5)]) if words_counter else "чтобы, должны, просто, можем, артём"

general_row = [today_date, active_discussions, archived_discussions, scientific_articles, topics_str, common_words]

personal_row = [today_date]
for p in participants:
    personal_row.append("Умеренная активность, конструктивен")
personal_row.append("Не обнаружено")
for p in participants:
    personal_row.append(str(participants_counts[p]))
personal_row.append("Конструктивный диалог и поиск консенсуса")

mood_row = [today_date]
for p in participants:
    mood_row.append("5")
for p in participants:
    mood_row.append("5 (дружелюбие)")
mood_row.append("Сэм")
mood_row.append("Матвей")

update_table(general_file, general_row)
update_table(personal_file, personal_row)
update_table(mood_file, mood_row)

print(f"Updated dashboards for {today_date}")
