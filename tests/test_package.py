"""The package as a whole: what importing it brings in."""

import subprocess
import sys

# Prints the top-level packages outside the standard library that `import hyoshigi` loads.
IMPORT_REPORT = """
import sys
loaded_before = set(sys.modules)
import hyoshigi
loaded = {name.partition(".")[0] for name in set(sys.modules) - loaded_before}
print(sorted(loaded - sys.stdlib_module_names))
"""


def test_importing_the_package_loads_only_the_standard_library() -> None:
    command = [sys.executable, "-c", IMPORT_REPORT]

    report = subprocess.run(command, capture_output=True, text=True, check=True)

    assert report.stdout.strip() == "['hyoshigi']"
