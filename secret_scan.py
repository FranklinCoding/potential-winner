"""Self-review: scan all .py files for hardcoded secrets."""
import re
import sys
from pathlib import Path

PATTERN = re.compile(
    r'(api_key|api_secret|passphrase|private_key|password)'
    r'\s*=\s*["\'][^"\']{8,}["\']',
    re.IGNORECASE,
)
SKIP_TERMS = ["os.getenv", "os.environ", "your_", "placeholder",
              "test_", "example", "paper", "fake", "none", "empty"]

issues = []
for f in Path(".").rglob("*.py"):
    if ".pytest_cache" in str(f) or "secret_scan" in str(f):
        continue
    try:
        content = f.read_text(errors="ignore")
        for m in PATTERN.finditer(content):
            val = m.group(0).lower()
            if not any(t in val for t in SKIP_TERMS):
                issues.append(f"{f.name}: {m.group(0)[:80]}")
    except Exception:
        pass

if issues:
    print("POTENTIAL HARDCODED SECRETS:")
    for i in issues:
        print(f"  {i}")
    sys.exit(1)
else:
    print("Secret scan: CLEAN — no hardcoded secrets found.")
