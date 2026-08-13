with open('discussions/active/space_colonization.md', 'r', encoding='utf-8') as f:
    text = f.read()

head_start = text.find('<<<<<<< HEAD\n')
head_end = text.find('=======\n')
branch_end = text.find('>>>>>>> origin/main\n')

if head_start != -1 and head_end != -1 and branch_end != -1:
    vladimir_text = text[head_start + len('<<<<<<< HEAD\n') : head_end]
    elon_text = text[head_end + len('=======\n') : branch_end]

    new_text = text[:head_start] + elon_text + vladimir_text + text[branch_end + len('>>>>>>> origin/main\n'):]

    with open('discussions/active/space_colonization.md', 'w', encoding='utf-8') as f:
        f.write(new_text)
    print("Conflict resolved.")
else:
    print("Could not find conflict markers.")
