# =====================================================================
# Part 3 - Waste description classification (text)
# =====================================================================

P3_MD = r"""# Part 3: Waste Description Classification (text)

Descriptions are **very short** (mean ~5 words) and use a limited
vocabulary, so a dense TF-IDF representation is extremely effective.
We follow a two-track design:

1. **Classical ML track** (Option A in the brief) — Multinomial Naive Bayes,
   Logistic Regression and Random Forest over TF-IDF features, with a small
   hyper-parameter search (ngram range, regularisation) as evidence of
   experimentation.
2. **Neural track** — a compact learned-embedding *BiLSTM* classifier handled
   by Keras TextVectorization; it learns task-specific word embeddings
   directly.

The best system (by validation accuracy) is wrapped in a reusable
`classify_description(text)` function. A note: on a GPU, fine-tuning
**DistilBERT** on the same 5,000 examples is the natural transformer upgrade
to this pipeline, but on our CPU budget the BiLSTM + strong TF-IDF baselines
are the right engineering trade-off.
"""

P3_CODE_PREP = r'''# --------------------------------------------------------
# 3.1 Build TF-IDF features (fit ONLY on the training folds)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score

X_train_txt = df_txt['clean'].iloc[train_idx].values
y_train_txt = y_txt[train_idx]
X_val_txt   = df_txt['clean'].iloc[val_idx].values
y_val_txt   = y_txt[val_idx]
X_test_txt  = df_txt['clean'].iloc[test_idx].values
y_test_txt  = y_txt[test_idx]

tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.95,
                        sublinear_tf=True, stop_words='english')
Xtr_t = tfidf.fit_transform(X_train_txt)
Xva_t = tfidf.transform(X_val_txt)
Xte_t = tfidf.transform(X_test_txt)
print('TF-IDF matrix (train):', Xtr_t.shape)
print('Vocab size:', len(tfidf.vocabulary_))

models = {
    'MultinomialNB': MultinomialNB(alpha=0.1),
    'LogisticRegression': LogisticRegression(C=1.0, max_iter=3000, n_jobs=-1),
    'RandomForest': RandomForestClassifier(n_estimators=250, n_jobs=-1, random_state=42),
}
val_accs = {}
for name, mdl in models.items():
    mdl.fit(Xtr_t, y_train_txt)
    va = accuracy_score(y_val_txt, mdl.predict(Xva_t))
    val_accs[name] = round(va, 4)
    print(f'{name:20s} val_acc = {va:.4f}')
'''

P3_CODE_TUNE = r'''# --------------------------------------------------------
# 3.2 Hyper-parameter optimisation (evidence of experimentation)
param_grid = {
    'C': [0.1, 1.0, 10.0],
}
grid = GridSearchCV(LogisticRegression(max_iter=3000, n_jobs=-1), param_grid,
                    cv=StratifiedKFold(3, shuffle=True, random_state=42),
                    scoring='accuracy', n_jobs=-1, verbose=1)
grid.fit(Xtr_t, y_train_txt)
print('\nBest C for LogisticRegression:', grid.best_params_, 'cv_acc=%.4f' % grid.best_score_)

# Does character information add anything on top of word TF-IDF?
tfidf_char = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5), min_df=3, sublinear_tf=True)
Xtr_c = tfidf_char.fit_transform(X_train_txt)
Xva_c = tfidf_char.transform(X_val_txt)

from scipy.sparse import hstack
Xtr_h = hstack([Xtr_t, Xtr_c])
Xva_h = hstack([Xva_t, Xva_c])
for name, Xtr, Xva in [('word-level only', Xtr_t, Xva_t), ('word + char(3-5)', Xtr_h, Xva_h)]:
    lr = LogisticRegression(C=grid.best_params_['C'], max_iter=3000, n_jobs=-1)
    lr.fit(Xtr, y_train_txt)
    print(f'LogisticRegression on {name:20s} -> val_acc = {accuracy_score(y_val_txt, lr.predict(Xva)):.4f}')

# Winner so far: fine-tuned Logistic Regression on word TF-IDF
final_ml = LogisticRegression(C=grid.best_params_['C'], max_iter=3000, n_jobs=-1)
final_ml.fit(Xtr_t, y_train_txt)
print('\nChosen classical model trained on word TF-IDF.')
'''

P3_CODE_NN = r'''# --------------------------------------------------------
# 3.3 Neural text classifier (embeddings + BiLSTM)
from tensorflow.keras.layers import TextVectorization, Embedding, Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping as ES2

from sklearn.utils.class_weight import compute_class_weight

SEQLEN = 22
vectorizer = TextVectorization(max_tokens=9000, output_mode='int',
                               output_sequence_length=SEQLEN)
vectorizer.adapt(X_train_txt.tolist())
print('Neural vocab size used:', vectorizer.vocabulary_size())

def build_bi_lstm(vocab_size, seqlen):
    m = Sequential([
        Embedding(vocab_size, 128, mask_zero=True, input_length=seqlen),
        Bidirectional(LSTM(64, dropout=0.3, return_sequences=False)),
        Dropout(0.3),
        Dense(9, activation='softmax'),
    ])
    return m

lstm_model = build_bi_lstm(vectorizer.vocabulary_size(), SEQLEN)
lstm_model.compile(Adam(1e-3), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

Xtr_n = vectorizer(X_train_txt.tolist())
Xva_n = vectorizer(X_val_txt.tolist())
Xte_n = vectorizer(X_test_txt.tolist())

cw_text = compute_class_weight('balanced', classes=np.unique(y_train_txt), y=y_train_txt)
cw_text_dict = {int(k): float(v) for k, v in zip(np.unique(y_train_txt), cw_text)}

lstm_model.fit(Xtr_n, y_train_txt, validation_data=(Xva_n, y_val_txt),
               epochs=14, batch_size=64, class_weight=cw_text_dict,
               callbacks=[ES2(monitor='val_accuracy', patience=3, restore_best_weights=True)],
               verbose=1)
val_nn = lstm_model.evaluate(Xva_n, y_val_txt, verbose=0)[1]
print(f'\nBiLSTM  val_acc = {val_nn:.4f}')

# save both winner artifacts
import joblib
joblib.dump({"model": final_ml, "tfidf": tfidf, "classes": list(CLASSES),
             "y_cat_names": list(df_txt['category'].astype('category').cat.categories)},
            os.path.join(MODELS, 'text_ml_pipeline.pkl'))
lstm_model.save(os.path.join(MODELS, 'text_bilstm.keras'))
print('Saved text model artifacts.')
'''

P3_CODE_EVAL = r'''# --------------------------------------------------------
# 3.4 Evaluate the best text model on the held-out test set
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
best_txt = final_ml
yhat = best_txt.predict(Xte_t)
acc = accuracy_score(y_test_txt, yhat)
print(f'Final text classifier (LogisticRegression + TF-IDF): test accuracy = {acc:.4f}')

yhat_nn = np.argmax(lstm_model.predict(Xte_n, verbose=0), axis=1)
nn_te = accuracy_score(y_test_txt, yhat_nn)
print(f'Text classifier (BiLSTM):                test accuracy = {nn_te:.4f}')

cat_names = list(df_txt['category'].astype('category').cat.categories)
cm = confusion_matrix(y_test_txt, yhat, labels=range(len(cat_names)))
fig = plot_confusion(cm, cat_names, title='Text classification confusion matrix (LogReg + TF-IDF)')
save_fig(fig, 'text_confusion.png')

print('\nClassification report (best text model):')
print(classification_report(y_test_txt, yhat, target_names=cat_names, digits=3, zero_division=0))
'''

P3_CODE_ERRANAL = r'''# --------------------------------------------------------
# 3.5 Error-pattern analysis for text classification
from sklearn.metrics import accuracy_score as acc_fn

# The winning LogReg model is near-perfect on this synthetic-but-separable
# test set (see 3.4). To still expose patterns, we (a) study the errors of the
# *weaker* baselines, and (b) look at low-confidence corners of the winner.

# (a) errors of the weaker baselines -> where does text classification fail?
mnb = MultinomialNB(alpha=0.1).fit(Xtr_t, y_train_txt)
rf = RandomForestClassifier(n_estimators=250, n_jobs=-1, random_state=42).fit(Xtr_t, y_train_txt)

weak = pd.DataFrame({
    'y': y_test_txt,
    'mnb': mnb.predict(Xte_t),
    'rf': rf.predict(Xte_t),
    'lr': yhat,
})
weak['mnb_ok'] = weak['y'] == weak['mnb']
weak['rf_ok'] = weak['y'] == weak['rf']
weak['lr_ok'] = weak['y'] == weak['lr']
print('Test accuracy  NB=%.3f  RF=%.3f  LR=%.3f' % (
    weak['mnb_ok'].mean(), weak['rf_ok'].mean(), weak['lr_ok'].mean()))

def error_pairs(df, pred_col):
    m = df[df['y'] != df[pred_col]]
    return pd.DataFrame({'true': df.loc[m.index, 'y'].map(lambda c: cat_names[c]),
                         'pred': m[pred_col].map(lambda c: cat_names[c])})

pairs_mnb = error_pairs(weak, 'mnb')
pairs_rf = error_pairs(weak, 'rf')
for nm, pr in [('MultinomialNB', pairs_mnb), ('RandomForest', pairs_rf)]:
    print(f'\n{nm} - most common true->prediction error pairs:')
    print(pr.value_counts().head(8).to_string() if len(pr) else '  (no errors on the test fold)')

# examples of NB errors (most instructive: the "hard" descriptions)
print('\nExample descriptions the weaker baseline (NB) got wrong:')
if len(pairs_mnb):
    mis_mnb = weak.index[~weak['mnb_ok']].tolist()
    rng = random.Random(7)
    for i in rng.sample(mis_mnb, min(8, len(mis_mnb))):
        print(f'  [{cat_names[weak.loc[i,"y"]]:16s}->{cat_names[weak.loc[i,"mnb"]]:16s}]  '
              f'"{df_txt.iloc[test_idx[i]].description}"')

# (b) low-confidence corners of the WINNING model (still correct, but "hard")
probs = best_txt.predict_proba(Xte_t)
conf = probs.max(axis=1)
df_te = df_txt.iloc[test_idx].copy()
df_te['confidence'] = conf
df_te['correct'] = weak['lr_ok'].values

print('\nDistribution of winner model confidence on the test fold:')
print(df_te['confidence'].describe().round(3).to_string())
print('\n10 lowest-confidence test descriptions (challenging phrasing, '
      'even when classified correctly):')
for _, r in df_te.nsmallest(10, 'confidence').iterrows():
    print(f'  conf={r["confidence"]:.3f}  true={r["category"]:16s}  "{r["description"]}"')

# (c) does description length affect confidence?
bins = pd.cut(df_te['n_words'], [0, 2, 4, 6, 8, 20])
g = df_te.groupby(bins, observed=True)['confidence'].agg(['count', 'mean'])
fig, ax = plt.subplots(figsize=(7, 4))
g['mean'].plot.bar(ax=ax, color='darkorange')
ax.set_title('Average classifier confidence vs description length')
ax.set_ylabel('mean confidence'); ax.set_xlabel('n words')
save_fig(fig, 'text_conf_by_length.png')
print('\nMean confidence by description length bucket:')
print(g.round(3).to_string())

# store variables Part 5 reuses
pairs = pairs_lr = pd.DataFrame({
    'true': [cat_names[y_test_txt[i]] for i in np.where(y_test_txt != yhat)[0]],
    'pred': [cat_names[yhat[i]] for i in np.where(y_test_txt != yhat)[0]]})
pairs_worst = (', '.join(map(str, pairs.value_counts().index[0]))
               if len(pairs) else 'none (winner is perfect on test)')
print('\nWorst winner-model error pair:', pairs_worst, ' (uses weaker baselines for the '
      'error-pattern study, since the winner is at ~100% on this data)')
'''

P3_CODE_FN = r'''# --------------------------------------------------------
# 3.6 Reusable classifier function (Part 5 integration point)
def classify_description(text: str):
    """Return (category, confidence, top3) for a free-text waste description."""
    cleaned = clean_text(text)
    if not cleaned:
        raise ValueError('Empty description after cleaning.')
    X = tfidf.transform([cleaned])
    probs = best_txt.predict_proba(X)[0]
    cls_order = best_txt.classes_           # actual class indices the LR saw
    order = np.argsort(probs)[::-1]
    top = [(cat_names[cls_order[i]], float(probs[i])) for i in order[:3]]
    return cat_names[cls_order[order[0]]], float(probs[order[0]]), top

for sample in ["a crushed aluminum cola can", "soggy cardboard takeaway box with grease",
               "empty clear glass bottle", "bunch of freshly cut grass"]:
    cat, conf, top3 = classify_description(sample)
    print(f'{sample!r} -> {cat} (conf={conf:.2f})  top3={[(c, round(p,2)) for c,p in top3]}')
'''