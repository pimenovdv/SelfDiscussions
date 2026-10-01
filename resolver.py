with open("discussions/active/ai/ai_limits_of_learning.md", "r") as f:
    lines = f.readlines()

new_lines = []
in_conflict = False
for line in lines:
    if line.startswith("<<<<<<< HEAD"):
        in_conflict = True
    elif line.startswith("======="):
        pass
    elif line.startswith(">>>>>>> origin/main"):
        in_conflict = False
    else:
        new_lines.append(line)

with open("discussions/active/ai/ai_limits_of_learning.md", "w") as f:
    f.writelines(new_lines)


with open("discussions/active/ai/ai_stagnation_crisis.md", "r") as f:
    lines = f.readlines()

new_lines = []
in_conflict = False
head_lines = []
main_lines = []

state = 0 # 0=normal, 1=head, 2=main

for line in lines:
    if line.startswith("<<<<<<< HEAD"):
        state = 1
    elif line.startswith("======="):
        state = 2
    elif line.startswith(">>>>>>> origin/main"):
        state = 0
        new_lines.extend(main_lines)
        new_lines.extend(head_lines)
        head_lines = []
        main_lines = []
    else:
        if state == 0:
            new_lines.append(line)
        elif state == 1:
            head_lines.append(line)
        elif state == 2:
            main_lines.append(line)

with open("discussions/active/ai/ai_stagnation_crisis.md", "w") as f:
    f.writelines(new_lines)
