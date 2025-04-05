import re
def format_result_markdown(gpt_result: str) -> str:
    lines = gpt_result.strip().split("\n")

    header = "### 🎓 졸업 진단 결과\n\n"
    status = ""
    issues = []
    action = ""

    for line in lines:
        line = line.strip()

        if "졸업 가능" in line and "불가능" not in line:
            status = "**✅ 졸업 가능**\n"
        elif "졸업 불가능" in line or "불가" in line:
            status = "**❌ 졸업 불가**\n"
        elif line.startswith("-") or re.match(r"\d+\.", line):
            issues.append(line)
        elif line.startswith("👉") or "조치" in line or "수강" in line:
            action += f"\n{line}"  # 누적

    body = ""
    if issues:
        body += "\n**부족 항목**\n" + "\n".join(issues)

    if action:
        body += f"\n\n**📌 권장 조치**{action}"

    return header + status + body