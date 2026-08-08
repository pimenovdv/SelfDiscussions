import re
import sys

filename = sys.argv[1]
with open(filename, 'r', encoding='utf-8') as f:
    text = f.read()

print(f"--- CONTENT OF {filename} (last 2000 chars) ---")
print(text[-2000:])

speakers = re.findall(r'\*\*(.+?)\s*\(.*?\):\*\*', text)
if speakers:
    print("\n--- LAST SPEAKER ---")
    print(speakers[-1])
