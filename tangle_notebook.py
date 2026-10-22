import sys
from pathlib import Path

import nbformat

def tangle(notebook_path, tag="tangle", output_path=None):
    notebook_path = Path(notebook_path)

    with notebook_path.open(encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)

    code = [
        cell.source
        for cell in nb.cells
        if cell.cell_type == "code"
        and tag in cell.metadata.get("tags", [])
    ]

    output_path = Path(output_path) if output_path else notebook_path.with_suffix(".py")

    output_path.write_text(
        "\n\n".join(code) + "\n",
        encoding="utf-8",
    )

    print(f"Extracted {len(code)} cells to {output_path}")

if __name__ == "__main__":
    tangle(sys.argv[1])
