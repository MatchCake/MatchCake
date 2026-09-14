import os
import sys


def generate_doc(path_to_root_dir: str = "."):
    commands = [
        # -d 3 stops the generated toctrees one level above the members of a class. Sphinx puts every
        # documented object in the table of contents, so the default depth of 4 lists every method of
        # every class on the package page, burying the submodules the page is there to show.
        rf"sphinx-apidoc -f -d 3 -o {path_to_root_dir}/sphinx/source {path_to_root_dir}/src/matchcake",
        rf"{path_to_root_dir}\sphinx\make clean html",
        rf"{path_to_root_dir}\sphinx\make html",
    ]
    for command in commands:
        print(f"Executing: {command}")
        os.system(command)


if __name__ == "__main__":
    root_dir = sys.argv[1] if len(sys.argv) > 1 else ".."
    generate_doc(root_dir)
