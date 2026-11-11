"""Protect GitHub math rendering while keeping code and link checks usable."""

import importlib.util
from pathlib import Path
import subprocess
import sys


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check_docs.py"
SPEC = importlib.util.spec_from_file_location("genai_check_docs", SCRIPT)
DOCS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DOCS)


def test_detects_unrendered_inline_math_at_chinese_boundaries():
    text = r"系数$\sqrt{\tilde\beta_1}=0$，通过判据$\bar\alpha_t$。" + "\n"
    text += "A spaced formula $x_{<i}$ follows the same repository convention."
    issues = DOCS.math_format_issues(text)
    assert [line for line, _ in issues] == [1, 2]
    assert all("$`TeX`$" in message for _, message in issues)


def test_detects_display_math_inside_details_at_its_opening_line():
    text = "<details><summary>答案</summary>\n\n$$\n\\hat x_0=(1,-2).\n$$\n\n</details>"
    issues = DOCS.math_format_issues(text)
    assert len(issues) == 1
    assert issues[0][0] == 3 and "```math" in issues[0][1]
    assert DOCS.math_format_issues("$$x=y$$")[0][0] == 1


def test_accepts_explicit_math_with_markdown_special_characters():
    text = r"条件$`x_{<i}`$与$`\sqrt{\tilde\beta_1}=0`$，无需空格。" + "\n"
    text += "<details><summary>答案</summary>\n\n$`\\hat x_0=(1,-2).`$\n\n</details>\n\n"
    text += "```math\n\\hat x_0=(1,-2).\n```\n"
    assert DOCS.math_format_issues(text) == []


def test_details_math_fences_report_the_opening_lines_and_respect_nesting():
    text = """<details>
<summary>答案</summary>

$`x`$

```math
x=1
```
<details><summary>进一步说明</summary>
~~~math
y=2
~~~
</details>
</details>

```math
z=3
```
"""
    issues = DOCS.math_format_issues(text)
    assert [line for line, _ in issues] == [6, 10]
    assert all("standalone $`TeX`$ paragraph" in message for _, message in issues)
    assert all("outside details" in message for _, message in issues)
    # HTML details is not a void element: the slash still opens a section.
    self_closing = "<details/>\n```math\nx=1\n```\n"
    assert [line for line, _ in DOCS.math_format_issues(self_closing)] == [2]


def test_literal_details_in_code_comments_and_attributes_do_not_open_sections():
    text = """`<details>` and a multiline `<details>
literal span` are examples.
<span title="<details>">attribute</span>
<!--
<details>
```math
x=1
```
-->
````markdown
<details>
```math
x=2
```
````
\\<details>

```math
x=3
```
"""
    assert DOCS.math_format_issues(text) == []
    assert DOCS.math_format_issues("<!--\n<details>\n$$hidden to EOF$$") == []


def test_ignores_code_fences_spans_and_literal_dollars():
    text = """Use `$x$` or ``$`x`$`` as literal syntax examples.
An inline span can continue `with $x$
and $$ on the next line`.
Escaped \\$x\\$ and one price $5 remain literal.
````markdown
```math
$$invalid only when used as prose$$
```
still $inside$ the outer fence
````
~~~python
shell_pid = '$$'
~~~
"""
    assert DOCS.math_format_issues(text) == []


def test_ignores_html_attributes_but_checks_visible_html_content():
    text = """<a title="cost $5 or $7" href="https://example.org/$x$">link</a>
<span data-math='
$$ raw attribute $$'>条件$x$</span>
<!-- $ignored$ and
$$ ignored comment $$ -->
"""
    issues = DOCS.math_format_issues(text)
    assert len(issues) == 1 and issues[0][0] == 3


def test_table_norm_delimiters_preserve_their_tex_payload():
    broken = r"| 范数 | $`\|x\|_2`$ |" + "\n"
    issues = DOCS.math_format_issues(broken)
    assert len(issues) == 1 and issues[0][0] == 1
    assert r"\Vert instead of \|" in issues[0][1]
    repaired = r"| 范数 | $`\Vert x\Vert_2`$ |" + "\n"
    repaired += r"普通段落中的$`\|x\|_2`$允许保留。" + "\n"
    repaired += r"| 语法示例 | ``$`\|x\|_2`$`` |" + "\n"
    repaired += r"| 转义美元示例 | \$`\|x\|_2`$ |" + "\n"
    assert DOCS.math_format_issues(repaired) == []


def test_math_row_separator_stays_on_the_same_physical_line():
    broken = "```math\na &= b\\\\\nc &= d\n```\n"
    issues = DOCS.math_format_issues(broken)
    assert len(issues) == 1 and issues[0][0] == 2
    assert "same physical line" in issues[0][1]
    repaired = "```math\na &= b\\\\ c &= d\n```\n"
    repaired += "```python\nliteral_backslashes = \\\\\n```\n"
    repaired += "<!--\n```math\na &= b\\\\\n```\n-->\n"
    assert DOCS.math_format_issues(repaired) == []


def test_math_error_preserves_link_validation_and_source_line(tmp_path):
    (tmp_path / "target.md").write_text("# Target\n", encoding="utf-8")
    page = tmp_path / "lesson.md"
    page.write_text("# Lesson\n\n中文$x$\n[go](target.md#target)\n[broken](missing.md)\n", encoding="utf-8")
    errors, count = DOCS.check(tmp_path, require_paths=False)
    assert count == 2
    assert len(errors) == 2
    assert errors[0].startswith("lesson.md:3: inline math:")
    assert any("lesson.md:5:" in error and "target does not exist" in error for error in errors)
    page.write_text("# Lesson\n\n中文$`x`$\n[go](target.md#target)\n", encoding="utf-8")
    assert DOCS.check(tmp_path, require_paths=False) == ([], 1)


def test_cli_reports_real_rendering_fault_and_accepts_repaired_page(tmp_path):
    page = tmp_path / "lesson.md"
    page.write_text("# Lesson\n\n原始$x$\n", encoding="utf-8")
    command = [sys.executable, str(SCRIPT), "--root", str(tmp_path), "--no-required-paths"]
    failed = subprocess.run(command, capture_output=True, text=True, timeout=10)
    assert failed.returncode == 1
    assert "lesson.md:3: inline math:" in failed.stderr
    assert "0 local links; 1 errors" in failed.stdout
    page.write_text("# Lesson\n\n修复$`x`$\n", encoding="utf-8")
    passed = subprocess.run(command, capture_output=True, text=True, timeout=10)
    assert passed.returncode == 0 and passed.stderr == ""
    assert "0 local links; 0 errors" in passed.stdout
