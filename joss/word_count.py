import re
import shutil
from pathlib import Path

from mrkdwn_analysis import MarkdownAnalyzer

if __name__ == "__main__":
    markdown_file = Path(__file__).parent / "paper.md"
    analyzer = MarkdownAnalyzer(markdown_file)
    word_count = analyzer.count_words()
    print(f"Total words: {word_count}")

    tmp_folder = Path(__file__).parent / ".tmp"
    tmp_folder.mkdir(exist_ok=True, parents=True)
    tmp_markdown_file = tmp_folder / "tmp_paper.md"
    markdown_text = markdown_file.read_text(encoding="utf-8", errors="ignore")

    # remove code blocks
    markdown_file_text_only = re.sub(r"```.*?```", "", markdown_text, flags=re.DOTALL)
    # remove metadata section
    markdown_file_text_only = re.sub(
        r"^---.*?---", "", markdown_file_text_only, flags=re.DOTALL
    )
    # remove figures inside [](), ![](), [](){}, ![](){}
    markdown_file_text_only = re.sub(
        r"!?\[[^\]]*\]\((?:[^()\\]|\\.|(?:\([^()]*\)))*\)(?:\{[^}]*\})?",
        "",
        markdown_file_text_only,
    )
    # remove citations
    markdown_file_text_only = re.sub(
        r"\[\s*@[\w:-]+(?:\s*[,;]\s*@[\w:-]+)*\s*\]", "", markdown_file_text_only
    )

    tmp_markdown_file.write_text(
        markdown_file_text_only, encoding="utf-8", errors="ignore"
    )
    analyzer_text = MarkdownAnalyzer(tmp_markdown_file)
    word_count_text = analyzer_text.count_words()
    print(f"Total words in the text only: {word_count_text}")
    shutil.rmtree(tmp_folder, ignore_errors=True)
