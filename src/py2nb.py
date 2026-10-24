"""Convierte scripts en formato 'percent' (# %% / # %% [markdown]) en cuadernos .ipynb
y, opcionalmente, los ejecuta.

Uso (kernel "ml-e2" registrado con ipykernel): python src/py2nb.py cuadernos/01_datos.py book/01_datos.ipynb [--ejecutar]
"""
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient


def convertir(src: Path) -> nbformat.NotebookNode:
    nb = nbformat.v4.new_notebook()
    nb.metadata["kernelspec"] = {"name": "ml-e2", "display_name": "Python (ml-e2)", "language": "python"}
    nb.metadata["language_info"] = {"name": "python"}
    celdas, tipo, buf = [], None, []

    def cerrar():
        if tipo is None:
            return
        texto = "\n".join(buf).strip("\n")
        if tipo == "md":
            texto = "\n".join(l[2:] if l.startswith("# ") else l.lstrip("#") for l in texto.splitlines())
            celdas.append(nbformat.v4.new_markdown_cell(texto))
        elif texto:
            celdas.append(nbformat.v4.new_code_cell(texto))

    for linea in src.read_text(encoding="utf-8").splitlines():
        if linea.startswith("# %%"):
            cerrar()
            tipo, buf = ("md" if "[markdown]" in linea else "code"), []
        else:
            buf.append(linea)
    cerrar()
    nb.cells = celdas
    return nb


if __name__ == "__main__":
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    nb = convertir(src)
    if "--ejecutar" in sys.argv:
        try:
            NotebookClient(nb, timeout=7200, kernel_name="ml-e2",
                           resources={"metadata": {"path": str(dst.parent)}}).execute()
        finally:
            nbformat.write(nb, dst)
    nbformat.write(nb, dst)
    print("OK", dst)
