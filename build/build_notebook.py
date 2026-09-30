# Assembles the final EcoSort summative notebook from the cell modules.
import nbformat as nbf

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'nb_cells'))

from c00_common import TITLE_MD, CONFIG_CODE, UTILS_CODE
from c01_part1 import (P1_MD_IMAGE, P1_CODE_IMG_DATA, P1_CODE_IMG_EDA,
                       P1_CODE_IMG_SAMPLES, P1_MD_TEXT, P1_CODE_TEXT_EDA,
                       P1_CODE_TEXT_COMMON, P1_MD_POLICY, P1_CODE_POLICY_EDA,
                       P1_MD_CHALLENGES, P1_MD_PIPELINES, P1_CODE_PIPELINES)
from c02_part2 import (P2_MD, P2_CODE_FEATURES, P2_CODE_HEAD_EXP,
                       P2_CODE_BUILD_MODEL, P2_CODE_CNN_EVAL, P2_CODE_CNN_ERRORS)
from c03_part3 import (P3_MD, P3_CODE_PREP, P3_CODE_TUNE, P3_CODE_NN,
                       P3_CODE_EVAL, P3_CODE_ERRANAL, P3_CODE_FN)
from c04_part4 import (P4_MD, P4_CODE_CORPUS, P4_CODE_RETRIEVE,
                       P4_CODE_GENDATA, P4_CODE_FINETUNE, P4_CODE_SAMPLING,
                       P4_CODE_RAGEVAL, P4_MD_SUMMARY, P4_CODE_FN)
from c05_part5 import (P5_MD, P5_CODE_CLASS, P5_CODE_TEST, P5_CODE_TESTTXT,
                       P5_CODE_EDGE, P5_MD_SUMMARY, P5_CODE_SUMMARY,
                       P5_MD_REFLECT)

cells = []
md = lambda s: nbf.v4.new_markdown_cell(s)
cd = lambda s: nbf.v4.new_code_cell(s)

cells += [md(TITLE_MD), cd(CONFIG_CODE), cd(UTILS_CODE)]

# Part 1
cells += [md(P1_MD_IMAGE), cd(P1_CODE_IMG_DATA), cd(P1_CODE_IMG_EDA),
          cd(P1_CODE_IMG_SAMPLES), md(P1_MD_TEXT), cd(P1_CODE_TEXT_EDA),
          cd(P1_CODE_TEXT_COMMON), md(P1_MD_POLICY), cd(P1_CODE_POLICY_EDA),
          md(P1_MD_CHALLENGES), md(P1_MD_PIPELINES), cd(P1_CODE_PIPELINES)]

# Part 2
cells += [md(P2_MD), cd(P2_CODE_FEATURES), cd(P2_CODE_HEAD_EXP),
          cd(P2_CODE_BUILD_MODEL), cd(P2_CODE_CNN_EVAL), cd(P2_CODE_CNN_ERRORS)]

# Part 3
cells += [md(P3_MD), cd(P3_CODE_PREP), cd(P3_CODE_TUNE), cd(P3_CODE_NN),
          cd(P3_CODE_EVAL), cd(P3_CODE_ERRANAL), cd(P3_CODE_FN)]

# Part 4
cells += [md(P4_MD), cd(P4_CODE_CORPUS), cd(P4_CODE_RETRIEVE),
          cd(P4_CODE_GENDATA), cd(P4_CODE_FINETUNE), cd(P4_CODE_SAMPLING),
          cd(P4_CODE_RAGEVAL), md(P4_MD_SUMMARY), cd(P4_CODE_FN)]

# Part 5
cells += [md(P5_MD), cd(P5_CODE_CLASS), cd(P5_CODE_TEST), cd(P5_CODE_TESTTXT),
          cd(P5_CODE_EDGE), md(P5_MD_SUMMARY), cd(P5_CODE_SUMMARY),
          md(P5_MD_REFLECT)]

nb = nbf.v4.new_notebook(cells=cells, metadata={
    'kernelspec': {
        'display_name': 'Python 3 (ipykernel)',
        'language': 'python',
        'name': 'python3',
    },
    'language_info': {'name': 'python', 'version': '3.12'},
})

out = os.path.join(os.path.dirname(__file__), '..', 'EcoSort_Summative_Lab.ipynb')
out = os.path.abspath(out)
with open(out, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)
print('Notebook written to', out)
print('Total cells:', len(cells))