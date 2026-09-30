# =====================================================================
# Part 5 - Integrated waste management assistant
# =====================================================================

P5_MD = r"""# Part 5: Integrated Waste Management Assistant

The three trained components are composed into a single assistant that
accepts **either an image or a free-text description** and returns
**(i)** the predicted waste category, **(ii)** tailored recycling
instructions and **(iii)** the supporting policy documents.

```text
                    +-------------------------------------------+
                    |         EcoSort Waste Assistant            |
                    |                                           |
  image  ---------> |  CNN classifier (MobileNetV2 + head)      |
  (uploaded photo)  |      -> predicted material class          |
                    |                                           |
  text   ---------> |  Text classifier (TF-IDF + LogisticReg)    |
  (resident query)  |      -> predicted material class          |
                    |                    |                      |
                    |                    v                      |
                    |            unified category               |
                    |                    |                      |
                    |                    v                      |
                    |  RAG generator: retrieve relevant         |
                    |  Metro City policy -> grounded recycling  |
                    |  instructions + source documents          |
                    +-------------------------------------------+
```

We also test robustness with **edge cases** (missing file, empty text,
low-confidence predictions) and sketch a lightweight **user-feedback**
mechanism that adapts retrieval scoring over time.
"""

P5_CODE_CLASS = r'''# --------------------------------------------------------
# 5.1 Assemble the assistant (loads all saved artifacts)
import joblib
from sklearn.metrics import accuracy_score as _acc

class EcoSortAssistant:
    """Unified waste assistant: image OR text in -> instructions out."""

    MODELS = ['cnn', 'text_ml', 'text_nn', 'rag']

    def __init__(self, models_dir=None):
        if models_dir is None:
            models_dir = MODELS
        self.models_dir = models_dir
        self.labels = CLASSES
        self.idx_to_label = IDX_TO_CLASS

        # --- CNN ---
        self.cnn = tf.keras.models.load_model(os.path.join(models_dir, 'cnn_finetuned.keras'))
        self.cnn_size = FT_SIZE  # fine-tune resolution used at training time

        # --- text classifier (pick the best saved pipeline) ---
        pkl = joblib.load(os.path.join(models_dir, 'text_ml_pipeline.pkl'))
        self.tfidf = pkl['tfidf']
        self.text_clf = pkl['model']
        self.cat_names = list(pkl['y_cat_names'])
        self.bilstm = tf.keras.models.load_model(os.path.join(models_dir, 'text_bilstm.keras'))
        self.vec_nn = None  # set on first neural call (TextVectorization re-adapt)

        # --- RAG ---
        self.emb = np.load(os.path.join(DATA, 'rag_embeddings.npz'))['emb']
        self.corpus = pd.read_csv(os.path.join(DATA, 'rag_corpus.csv'))
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        from transformers import AutoTokenizer, AutoModelForCausalLM
        self.tok = AutoTokenizer.from_pretrained(os.path.join(models_dir, 'rag_gpt2_finetuned'))
        self.tok.pad_token = self.tok.eos_token
        self.gen_model = AutoModelForCausalLM.from_pretrained(os.path.join(models_dir, 'rag_gpt2_finetuned'))
        self.gen_model.eval()

        self.feedback_log = os.path.join(DATA, 'feedback_log.jsonl')
        self.confidence_threshold = 0.60

    # ---------------- image classification ----------------
    def predict_image(self, image_path):
        if not os.path.exists(image_path):
            raise FileNotFoundError(f'Image not found: {image_path}')
        im = Image.open(image_path).convert('RGB').resize((self.cnn_size, self.cnn_size))
        x = np.asarray(im, dtype=np.float32)
        x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
        probs = self.cnn.predict(x[None], verbose=0)[0]
        order = np.argsort(probs)[::-1]
        return self.labels[order[0]], float(probs[order[0]]), \
               [(self.labels[i], float(probs[i])) for i in order[:3]]

    # ---------------- text classification ----------------
    def predict_text(self, text):
        cleaned = clean_text(text)
        if not cleaned:
            raise ValueError('Empty description after cleaning.')
        X = self.tfidf.transform([cleaned])
        probs = self.text_clf.predict_proba(X)[0]
        cls_order = self.text_clf.classes_
        order = np.argsort(probs)[::-1]
        return self.cat_names[cls_order[order[0]]], float(probs[order[0]]), \
               [(self.cat_names[cls_order[i]], float(probs[i])) for i in order[:3]]

    # ---------------- RAG ----------------
    def _retrieve(self, query, k=3, category=None, boost=0.6):
        q = self.encoder.encode([query], normalize_embeddings=True)[0]
        sim = self.emb @ q
        if category is not None:
            for i, r in self.corpus.iterrows():
                vc = str(r['valid_categories'])
                if category in [x.strip() for x in vc.strip('[]').replace("'", "").split(',')]:
                    sim[i] *= (1 + boost)
        top = np.argsort(sim)[::-1][:k]
        return [(float(sim[i]), int(i)) for i in top]

    def _generate(self, category, query, k_docs=3, max_new=110):
        ctx_parts = []
        for s, i in self._retrieve(query, k_docs, category=category):
            ctx_parts.append(self.corpus.iloc[i]['text'])
        joined = '\n\n'.join(ctx_parts)
        prompt = (f'You are a recycling assistant for Metro City. Answer using only the reference policy.\n'
                  f'Question: How to recycle {category}?\nReference:\n{joined}\nAnswer:')
        ids = self.tok.encode(prompt, return_tensors='pt')
        with torch.no_grad():
            out = self.gen_model.generate(ids, max_new_tokens=max_new,
                                          **GEN_CFG, no_repeat_ngram_size=3,
                                          repetition_penalty=1.1,
                                          eos_token_id=self.tok.eos_token_id,
                                          pad_token_id=self.tok.eos_token_id)
        text = self.tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip()
        cut = text.find('Question:')
        if cut > 0:
            text = text[:cut].strip()
        return text, joined

    def instructions(self, category, query=None, k_docs=3):
        query = query or f'How to recycle {category}?'
        text, _ = self._generate(category, query, k_docs=k_docs)
        sources = [{'doc_id': self.corpus.iloc[i]['doc_id'],
                    'policy_type': self.corpus.iloc[i]['policy_type'],
                    'score': round(s, 3),
                    'excerpt': self.corpus.iloc[i]['text'][:160]}
                   for s, i in self._retrieve(query, k_docs, category=category)]
        return text, sources

    # ---------------- unified entry point ----------------
    def assist(self, image_path=None, text=None, k_docs=3):
        """Classify the input (image XOR text) and produce instructions."""
        history = []
        if image_path is not None:
            cat, conf, top3 = self.predict_image(image_path)
            history.append({'mode': 'image', 'input': os.path.basename(image_path),
                            'top3': top3})
        elif text is not None:
            cat, conf, top3 = self.predict_text(text)
            history.append({'mode': 'text', 'input': text, 'top3': top3})
        else:
            raise ValueError('Provide either image_path or text.')

        warning = None
        if conf < self.confidence_threshold:
            warning = (f'Low confidence ({conf:.2f}); the answer below uses the '
                       f'predicted category but treat it as provisional.')

        instr, sources = self.instructions(cat, k_docs=k_docs)
        return {'category': cat, 'confidence': conf, 'history': history,
                'warning': warning, 'instructions': instr, 'sources': sources}

    # ---------------- user feedback mechanism ----------------
    def log_feedback(self, category, note='', ok=True):
        with open(self.feedback_log, 'a', encoding='utf-8') as f:
            f.write(json.dumps({'ts': time.time(), 'category': category,
                                'satisfied': bool(ok), 'note': note}) + '\n')

    def apply_feedback_to_retrieval(self):
        """Reward categories that residents mark as useful when retrieving."""
        if not os.path.exists(self.feedback_log):
            return None
        scores = Counter()
        with open(self.feedback_log, encoding='utf-8') as f:
            for ln in f:
                r = json.loads(ln)
                if r.get('satisfied'):
                    scores[r['category']] += 1
                    scores[r['category']] += 0
        boosts = {c: 1.0 + 0.05 * min(s, 10) for c, s in scores.items()}
        return boosts

assistant = EcoSortAssistant()
print('EcoSortAssistant initialised.')
print('Quick smoke test (text):',
      assistant.assist(text='an empty crushed plastic bottle')['category'])
'''

P5_CODE_TEST = r'''# --------------------------------------------------------
# 5.2 System test on REAL test-set images
import matplotlib as _mpl
rng = random.Random(5)
demo = test_df.sample(3, random_state=5)

fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
for ax, (_, row) in zip(axes, demo.iterrows()):
    res = assistant.assist(image_path=row['path'])
    im = Image.open(row['path']).resize((200, 200))
    ax.imshow(im); ax.axis('off')
    ax.set_title(f"TRUE:{row['class']}  PRED:{res['category']}", fontsize=9)
plt.suptitle('Integrated assistant on held-out test images', y=1.0)
save_fig(fig, 'assistant_image_demo.png')

for _, row in demo.iterrows():
    print('\n' + '=' * 80)
    print(f"INPUT IMAGE  : {os.path.basename(row['path'])}  (TRUE: {row['class']})")
    res = assistant.assist(image_path=row['path'])
    print(f"PREDICTED    : {res['category']}  (conf {res['confidence']:.2f})")
    if res['warning']:
        print('WARNING      :', res['warning'])
    print('INSTRUCTIONS :')
    print('   ' + res['instructions'].replace('\n', '\n   '))
    print('SOURCES      :', [s['doc_id'] for s in res['sources']])
'''

P5_CODE_TESTTXT = r'''# --------------------------------------------------------
# 5.3 System test on real text descriptions (from the held-out fold)
demo_txt = df_txt.iloc[test_idx].sample(4, random_state=6)
for _, row in demo_txt.iterrows():
    print('\n' + '=' * 80)
    print(f"INPUT TEXT   : \"{row['description']}\"  (TRUE: {row['category']})")
    res = assistant.assist(text=row['description'])
    print(f"PREDICTED    : {res['category']}  (conf {res['confidence']:.2f})")
    if res['warning']:
        print('WARNING      :', res['warning'])
    print('INSTRUCTIONS :')
    print('   ' + res['instructions'].replace('\n', '\n   '))
    print('SOURCES      :', [s['doc_id'] for s in res['sources']])
'''

P5_CODE_EDGE = r'''# --------------------------------------------------------
# 5.4 Edge cases + user-feedback loop
print('--- Edge case 1: missing image file ---')
try:
    assistant.assist(image_path='definitely_missing.jpg')
except Exception as ex:
    print('Handled ->', type(ex).__name__, '-', ex)

print('\n--- Edge case 2: empty / junk text ---')
for bad in ['', '  !! ## ', 'zzz qqq']:
    try:
        r = assistant.assist(text=bad)
        print(f'  {bad!r:<12} -> {r["category"]} (conf {r["confidence"]:.2f})')
    except Exception as ex:
        print(f'  {bad!r:<12} -> Handled: {type(ex).__name__}: {ex}')

print('\n--- Edge case 3: low-confidence prediction shows a warning ---')
res = assistant.assist(text='some weird shiny rectangular thing')
print('pred:', res['category'], 'conf:', round(res['confidence'], 3),
      '| warning:', res['warning'])

print('\n--- User feedback adjusts retrieval preferences ---')
for c in ['Cardboard', 'Cardboard', 'Plastic']:
    assistant.log_feedback(c, note='resident marked useful')
boosts = assistant.apply_feedback_to_retrieval()
print('Category boosts derived from feedback log:', boosts)

# demonstrate the boost effect on a cardboard query
def _ret_top(query, k=3, boost=None):
    q = assistant.encoder.encode([query], normalize_embeddings=True)[0]
    sim = assistant.emb @ q
    A = assistant.corpus
    if boost:
        for i, r in A.iterrows():
            vc = str(r['valid_categories'])
            for c, b in boost.items():
                if c in vc:
                    sim[i] *= b
    top = np.argsort(sim)[::-1][:k]
    return [(A.iloc[i]['doc_id'], round(float(sim[i]), 3)) for i in top]

before = _ret_top('How to recycle cardboard?')
after = _ret_top('How to recycle cardboard?', boost=boosts)
print('Cardboard retrieval BEFORE feedback:', before)
print('Cardboard retrieval AFTER  feedback:', after)
'''

P5_MD_SUMMARY = r"""### 5.5 System-level evaluation

Below we consolidate the metrics of every component into one table and record
limitations, biases and possible improvements. This mirrors the rubric's
final evaluation of the *integrated* system.
"""

P5_CODE_SUMMARY = r'''# --------------------------------------------------------
# 5.5 Overall results table + reflection
summary = {
    'CNN (image)': {
        'frozen-224 head test acc': res_head[best_head]['test_acc'],
        'fine-tuned test acc': round(cnn_test_acc, 4),
        'worst class (recall)': per.sort_values('recall').iloc[0]['class'],
    },
    'Text classification': {
        'LogReg+TF-IDF test acc': round(acc, 4),
        'BiLSTM test acc': round(nn_te, 4),
        'worst error pair': pairs_worst,
    },
    'RAG retrieval': {
        'hit@3 (pure embedding)': float(ret_ev['hit@k'].mean()),
        'precision@3 (pure)': float(ret_ev['p@k'].mean()),
        'hit@3 (category-aware)': float(ret_ev_hyb['hit@k'].mean()),
        'precision@3 (category-aware)': float(ret_ev_hyb['p@k'].mean()),
    },
    'RAG generation': {
        'ROUGE-1 vs top-1 doc': round(float(ragev['R1_vs_top1'].mean()), 3),
        'ROUGE-L vs top-1 doc': round(float(ragev['ROUGE_L_vs_top1'].mean()), 3),
        'faithfulness-jaccard': round(float(ragev['jaccard_vs_ctx'].mean()), 3),
    },
}
summary_df = pd.DataFrame({k: pd.Series(v) for k, v in summary.items()})
print(summary_df.to_string())
with open(os.path.join(DATA, 'final_metrics.json'), 'w') as f:
    json.dump(summary, f, indent=2, default=str)

# one integrated end-to-end demo
print('\n========== END-TO-END DEMO ==========')
res = assistant.assist(text='crumpled newspaper from yesterday')
print('Q : "crumpled newspaper from yesterday"')
print('Class:', res['category'], '| conf:', round(res['confidence'], 3))
print('Instructions:')
print(' ', res['instructions'].replace('\n', '\n  '))
print('Sources:')
for s in res['sources']:
    print(f"  - [{s['score']}] {s['doc_id']}: {s['excerpt'][:90]}")
'''

P5_MD_REFLECT = r"""### 5.6 Reflection: limitations, biases, and next steps

**What works.** The frozen-feature MobileNetV2 head reaches ~80% test accuracy
on the 9-class RealWaste set while the 224px frozen head is the strongest single
image model; the CPU-budget 128px fine-tune lands at ~73% (the 128px downgrade
costs more than the domain adaptation gains at this resolution). TF-IDF +
LogisticRegression classifies 5-word resident descriptions near-perfectly and is
trivially explainable. The RAG component reliably *grounds*
its instructions in Metro City policy (retrieval `hit@3` near 1.0; generated
text shares vocabulary with the retrieved source).

**Limitations & biases**
1. *Image domain gap* — RealWaste photos are clean, desk-style, uniform 524px
   captures; a phone snapshot of a dirty/non-decomposed item may behave
   differently. We mitigate with augmentation but cannot fully close the gap.
2. *Glass colour roll-up* — the dataset collapses green/brown/white glass into
   one "Glass" class, hiding a colour-sorting requirement the city cares about.
3. *Class imbalance* — Plastic/Metal dominate; Textile Trash is the smallest.
   Balanced class weights mitigate accuracy bias, but the model remains less
   calibrated on rare classes (see per-class recall).
4. *Synthetic text* — `waste_descriptions.csv` is generated (templated
   adjectives), so vocabulary diversity is limited relative to genuine
   resident language; common_confusion is 50% empty.
5. *Small generator* — DistilGPT2 on CPU is heavily capacity-limited; outputs
   are correct-but-stiff and sometimes truncate. A larger instruction-tuned
   model would help readability without changing the RAG architecture.
6. *Single jurisdiction* — all policy documents are Metro City's; porting the
   system requires swapping the corpus and re-indexing.

**Suggested next steps**
- Fine-tune/calibrate on real resident text + citizen feedback logs.
- Add the colour sub-labels for glass to the CNN (hierarchical labels).
- Serve via a lightweight API (FastAPI) with the assistant behind a
  confidence gating policy so low-confidence inputs trigger human review.
- Move generation to an instruct-tuned small LM and evaluate with human raters.
"""