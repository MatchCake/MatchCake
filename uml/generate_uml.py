import os
from pylint.pyreverse.main import Run


if __name__ == '__main__':
    output_directory = os.path.join(os.path.dirname(__file__), "uml")
    source_root = os.path.join(os.path.dirname(__file__), "..", "src", "matchcake")
    python_venv = os.path.join(os.path.dirname(__file__), "..", "venv", "Scripts", "python")
    args = [
        "--module-names y",
        "--colorized",
        "--project MatchCake",
        f"--output-directory {output_directory}",
        f"--source-roots {source_root}",
        "--output pdf",
        source_root,
    ]
    Run(' '.join(args).split(' '))
    # os.system(f"{python_venv} -m pyreverse {' '.join(args)}")
