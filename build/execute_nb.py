"""Execute a notebook in-process with nbclient, reporting any failing cell."""
import sys, time, traceback
import nbformat
from nbclient import NotebookClient

src, dst = sys.argv[1], sys.argv[2]
nb = nbformat.read(src, as_version=4)
client = NotebookClient(nb, timeout=-1, resources={'metadata': {'path': '.'}},
                        kernel_name='python3', allow_errors=False)
t0 = time.time()
try:
    client.execute()
    nbformat.write(nb, dst)
    print(f'OK  {dst}  ({time.time()-t0:.0f}s)', flush=True)
except Exception as e:
    print(f'FAILED: {type(e).__name__}: {e!r}', file=sys.stderr, flush=True)
    for cell in nb.cells:
        if cell.cell_type == 'code':
            for out in cell.get('outputs', []):
                if out.get('output_type') == 'error':
                    print(f"---- error in cell: {cell.source[:90]!r}", file=sys.stderr)
                    print('\n'.join(out.get('traceback', [])), file=sys.stderr, flush=True)
    try:
        nbformat.write(nb, dst)
    except Exception:
        pass
    sys.exit(1)