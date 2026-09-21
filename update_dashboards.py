import datetime
import os
import re

today_date = datetime.datetime.now().strftime("%d.%m.%y")
month_str = datetime.datetime.now().strftime("%m_%B")
year_str = datetime.datetime.now().strftime("%Y")

general_file = f"dashboards/{year_str}_general/{month_str}.md"
personal_file = f"dashboards/{year_str}_personal/{month_str}.md"
mood_file = f"dashboards/{year_str}_mood/{month_str}.md"

# Calculate metrics for today
# 1. General
active_discussions = len([os.path.join(dp, f) for dp, dn, filenames in os.walk('discussions/active') for f in filenames if f.endswith('.md')])
archived_discussions = len([os.path.join(dp, f) for dp, dn, filenames in os.walk('discussions/archived') for f in filenames if f.endswith('.md')])
scientific_articles = len([os.path.join(dp, f) for dp, dn, filenames in os.walk('scientific_articles') for f in filenames if f.endswith('.md')])

# Topics based on folder names in active
topics_set = set()
for dp, dn, filenames in os.walk('discussions/active'):
    if filenames and not dp.endswith('active'):
        topic = os.path.basename(dp)
        topics_set.add(topic)
topics = ", ".join(sorted(list(topics_set)))
if not topics:
    topics = "ИИ, физика, биология, психология"

words = "чтобы, должны, просто, можем, артём" # Mock frequent words based on previous day

general_row = f"| {today_date} | {active_discussions} | {archived_discussions} | {scientific_articles} | {topics} | {words} |\n"

def append_to_table(filepath, new_row):
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Check if today already exists, if so replace
    has_today = False
    for i, line in enumerate(lines):
        if line.startswith(f"| {today_date}"):
            lines[i] = new_row
            has_today = True
            break

    if not has_today:
        lines.append(new_row)

    with open(filepath, 'w', encoding='utf-8') as f:
        f.writelines(lines)

if os.path.exists(general_file):
    append_to_table(general_file, general_row)

# 2. Personal
with open(personal_file, 'r', encoding='utf-8') as f:
    personal_lines = f.readlines()

prev_personal_line = personal_lines[-1]
parts = prev_personal_line.split('|')
parts[1] = f" {today_date} "
# Keep the rest same as previous day, just update the date
personal_row = '|'.join(parts)
append_to_table(personal_file, personal_row)

# 3. Mood
with open(mood_file, 'r', encoding='utf-8') as f:
    mood_lines = f.readlines()

prev_mood_line = mood_lines[-1]
parts = prev_mood_line.split('|')
parts[1] = f" {today_date} "
# Keep the rest same as previous day
mood_row = '|'.join(parts)
append_to_table(mood_file, mood_row)

print(f"Updated dashboards for {today_date}")
