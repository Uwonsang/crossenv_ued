import re
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
LOG_PATH = BASE_DIR / "layout_log.txt"

# 결과 파일명: 원본파일명_Analyzed.txt
OUTPUT_PATH = LOG_PATH.with_name(f"{LOG_PATH.stem}_Analyzed.txt")

pattern = re.compile(
    r"\[layout check\] pot=(\d+), onion=(\d+), plate=(\d+), goal=(\d+), valid=(True|False)"
)

total_count = 0
valid_count = 0
invalid_count = 0

records = []

with open(LOG_PATH, "r", encoding="utf-8") as f:
    for line in f:
        match = pattern.search(line)
        if not match:
            continue

        pot = int(match.group(1))
        onion = int(match.group(2))
        plate = int(match.group(3))
        goal = int(match.group(4))
        valid = (match.group(5) == "True")

        total_count += 1
        if valid:
            valid_count += 1
        else:
            invalid_count += 1

        records.append({
            "pot": pot,
            "onion": onion,
            "plate": plate,
            "goal": goal,
            "valid": valid,
        })

yield_rate = valid_count / total_count if total_count > 0 else 0.0

result_lines = []
result_lines.append("===== Yield Summary =====")
result_lines.append(f"source_file   : {LOG_PATH.name}")
result_lines.append(f"total_count   : {total_count}")
result_lines.append(f"valid_count   : {valid_count}")
result_lines.append(f"invalid_count : {invalid_count}")
result_lines.append(f"yield_rate    : {yield_rate:.4f}")
result_lines.append("")
result_lines.append("===== Invalid Samples =====")

for i, r in enumerate(records):
    if not r["valid"]:
        result_lines.append(
            f"[{i}] pot={r['pot']}, onion={r['onion']}, "
            f"plate={r['plate']}, goal={r['goal']}, valid={r['valid']}"
        )

result_text = "\n".join(result_lines)

# 터미널에도 출력
print(result_text)

# 분석 결과 파일로 저장
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    f.write(result_text)

print(f"\nSaved analyzed result to: {OUTPUT_PATH}")