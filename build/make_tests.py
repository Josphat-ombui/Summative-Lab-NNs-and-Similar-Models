# Creates incremental test notebooks for staged validation of the final notebook.
import os, sys
import nbformat as nbf

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, 'nb_cells'))
OUT = os.path.join(HERE, '..')

from c00_common import TITLE_MD, CONFIG_CODE, UTILS_CODE
from c01_part1 import (P1_MD_IMAGE, P1_CODE_IMG_DATA, P1_CODE_IMG_EDA,
                       P1_CODE_IMG_SAMPLES, P1_MD_TEXT, P1_CODE_TEXT_EDA,
                       P1_CODE_TEXT_COMMON, P1_MD_POLICY, P1_CODE_POLICY_EDA,
                       P1_MD_CHALLENGES, P1_MD_PIPELINES, P1_CODE_PIPELINES)
from c03_part3 import (P3_MD, P3_CODE_PREP, P3_CODE_TUNE, P3_CODE_NN,
                       P3_CODE_EVAL, P3_CODE_ERRANAL, P3_CODE_FN)
from c04_part4 import (P4_MD, P4_CODE_CORPUS, P4_CODE_RETRIEVE,
                       P4_CODE_GENDATA, P4_CODE_FINETUNE, P4_CODE_SAMPLING,
                       P4_CODE_RAGEVAL, P4_MD_SUMMARY, P4_CODE_FN)

md = lambda s: nbf.v4.new_markdown_cell(s)
cd = lambda s: nbf.v4.new_code_cell(s)

def write_nb(name, cells):
    nb = nbf.v4.new_notebook(cells=cells, metadata={
        'kernelspec': {'display_name': 'Python 3 (ipykernel)', 'language': 'python', 'name': 'python3'},
        'language_info': {'name': 'python', 'version': '3.12'}})
    path = os.path.join(OUT, name)
    with open(path, 'w', encoding='utf-8') as f:
        nbf.write(nb, f)
    print('wrote', path, '-', len(cells), 'cells')

P1 = [md(TITLE_MD), cd(CONFIG_CODE), cd(UTILS_CODE),
      md(P1_MD_IMAGE), cd(P1_CODE_IMG_DATA), cd(P1_CODE_IMG_EDA),
      cd(P1_CODE_IMG_SAMPLES), md(P1_MD_TEXT), cd(P1_CODE_TEXT_EDA),
      cd(P1_CODE_TEXT_COMMON), md(P1_MD_POLICY), cd(P1_CODE_POLICY_EDA),
      md(P1_MD_CHALLENGES), md(P1_MD_PIPELINES), cd(P1_CODE_PIPELINES)]
write_nb('_dev_t1_part1.ipynb', P1)

P3 = P1 + [md(P3_MD), cd(P3_CODE_PREP), cd(P3_CODE_TUNE), cd(P3_CODE_NN),
           cd(P3_CODE_EVAL), cd(P3_CODE_ERRANAL), cd(P3_CODE_FN)]
write_nb('_dev_t3_text.ipynb', P3)

P4 = P1 + [md(P4_MD), cd(P4_CODE_CORPUS), cd(P4_CODE_RETRIEVE),
           cd(P4_CODE_GENDATA), cd(P4_CODE_FINETUNE), cd(P4_CODE_SAMPLING),
           cd(P4_CODE_RAGEVAL), md(P4_MD_SUMMARY), cd(P4_CODE_FN)]
write_nb('_dev_t4_rag.ipynb', P4)