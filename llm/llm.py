from pathlib import Path

script_dir = Path(__file__).resolve().parent
file_path = script_dir / "The_Verdict.txt"

with file_path.open("r", encoding="utf-8") as file:
     raw_text = file.read()
print("Total number of characters: ", len(raw_text))
print(raw_text[:100])  # Display the first 100 characters to check the content