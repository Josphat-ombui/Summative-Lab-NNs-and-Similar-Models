import numpy as np, pandas as pd, torch, os
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
from rouge_score import rouge_scorer

DATA, MODELS = 'data', 'models'
corpus = pd.read_csv(f'{DATA}/rag_corpus.csv')
emb = np.load(f'{DATA}/rag_embeddings.npz')['emb']
encoder = SentenceTransformer('all-MiniLM-L6-v2')
tok = AutoTokenizer.from_pretrained('distilgpt2'); tok.pad_token = tok.eos_token
gen_model = AutoModelForCausalLM.from_pretrained(f'{MODELS}/rag_gpt2_finetuned')

CLASSES = ['Cardboard','Food Organics','Glass','Metal','Miscellaneous Trash','Paper','Plastic','Textile Trash','Vegetation']

def valid_for(r, category):
    vc = str(r['valid_categories'])
    return category in [x.strip() for x in vc.strip('[]').replace("'", '').split(',')]

def retrieve_hybrid(query, category, k=3, boost=0.6):
    q = encoder.encode([query], normalize_embeddings=True)[0]
    sim = emb @ q
    for i, r in corpus.iterrows():
        if valid_for(r, category):
            sim[i] *= (1 + boost)
    top = np.argsort(sim)[::-1][:k]
    return [(sim[i], int(i)) for i in top]

GEN_CFG = dict(do_sample=True)

def gen(category, style, max_new, k_docs=3):
    ctx = []
    for s, i in retrieve_hybrid(f'How to recycle {category}?', category, k=k_docs):
        ctx.append(corpus.iloc[i]['text'])
    joined = '\n\n'.join(ctx)
    if style == 'A':
        prompt = (f'You are a recycling assistant for Metro City. Answer using only the reference policy.\n'
                  f'Question: How to recycle {category}?\nReference:\n{joined}\nAnswer:')
    else:
        prompt = (f'You are a recycling assistant for Metro City. Answer using only the reference policy.\n'
                  f'Question: How to recycle {category}?\nReference:\n{joined}\n'
                  f'Answer: Give the preparation and collection steps exactly as stated in the reference.\nSteps:')
    ids = tok.encode(prompt, return_tensors='pt')
    with torch.no_grad():
        out = gen_model.generate(ids, max_new_tokens=max_new, **GEN_CFG,
                                 no_repeat_ngram_size=3, repetition_penalty=1.1,
                                 eos_token_id=tok.eos_token_id, pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip()

rs = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
rows = []
for style, mn in [('A', 110), ('B', 70), ('A2', 70)]:
    s_ = 'A' if style == 'A2' else style
    for cat in CLASSES:
        top1 = retrieve_hybrid(f'How to recycle {cat}?', cat, k=1)[0][1]
        ref = corpus.iloc[top1]['text']
        t = gen(cat, s_, mn)
        r = rs.score(ref, t)
        rows.append({'variant': style, 'cat': cat,
                     'R1': r['rouge1'].fmeasure, 'R2': r['rouge2'].fmeasure,
                     'RL': r['rougeL'].fmeasure, 'words': len(t.split())})
df = pd.DataFrame(rows)
print(df.groupby('variant')[['R1', 'R2', 'RL', 'words']].mean().round(3))
print()
print(rows[-12:])