# =====================================================================
# Part 1 - Dataset exploration and preparation
# =====================================================================

P1_MD_IMAGE = r"""# Part 1: Dataset Exploration & Preparation

## 1.1 RealWaste image dataset

We start by building a tidy DataFrame over the **RealWaste** images
(9 material classes), scanning basic metadata (size, colour mode, file
size, brightness / saturation) so we can reason about the visual
characteristics and the challenges the CNN will face.
"""

P1_CODE_IMG_DATA = r'''# -------------------------------------------------
# 1.1.1 Build the image catalogue (cached on disk)
meta_path = os.path.join(DATA, 'image_metadata.csv')
if os.path.exists(meta_path):
    df_img = pd.read_csv(meta_path)
    print('Loaded cached image metadata.')
else:
    rows = []
    for cls in CLASSES:
        cdir = os.path.join(IMAGE_ROOT, cls)
        for fn in sorted(os.listdir(cdir)):
            p = os.path.join(cdir, fn)
            rows.append({'filename': fn, 'class': cls, 'path': p})
    df_img = pd.DataFrame(rows)

    def scan_row(r):
        try:
            im = Image.open(r['path'])
            w, h = im.size
            mode = im.mode
            fs = os.path.getsize(r['path'])
            small = im.convert('RGB').resize((48, 48))
            arr = np.asarray(small).astype(np.float32)
            hsv = np.asarray(small.convert('HSV')).astype(np.float32)
            im.close()
            return pd.Series({
                'width': w, 'height': h, 'mode': mode,
                'aspect_ratio': w / h, 'file_kb': round(fs / 1024, 1),
                'mean_brightness': arr.mean(),
                'std_brightness': arr.std(),
                'mean_saturation': hsv[..., 1].mean(),
            })
        except Exception as ex:
            return pd.Series({'width': np.nan, 'height': np.nan, 'mode': 'ERR',
                              'aspect_ratio': np.nan, 'file_kb': np.nan,
                              'mean_brightness': np.nan, 'std_brightness': np.nan,
                              'mean_saturation': np.nan})

    meta = df_img.apply(scan_row, axis=1)
    df_img = pd.concat([df_img, meta], axis=1)
    df_img.to_csv(meta_path, index=False)
    print('Catalogue written to', meta_path)

print(df_img.shape)
print('Classes:', df_img.groupby('class').size().to_dict())
print('NaNs in metadata:', df_img[['width', 'height', 'mean_brightness']].isna().sum().sum())
'''

P1_CODE_IMG_EDA = r'''# -------------------------------------------------
# 1.1.2 Class distribution
class_counts = df_img['class'].value_counts().sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(10, 4.5))
bars = ax.bar(class_counts.index, class_counts.values, color=sns.color_palette('viridis', len(CLASSES)))
ax.set_title('RealWaste - number of images per material class')
ax.set_ylabel('Images')
for b, v in zip(bars, class_counts.values):
    ax.text(b.get_x() + b.get_width() / 2, v + 4, str(v), ha='center', fontsize=9)
plt.xticks(rotation=45, ha='right')
save_fig(fig, 'image_class_distribution.png')

imbalance = class_counts.max() / class_counts.min()
print(f'Most frequent class: {class_counts.index[0]} ({class_counts.iloc[0]})')
print(f'Least frequent class: {class_counts.index[-1]} ({class_counts.iloc[-1]})')
print(f'Class imbalance ratio (max/min) = {imbalance:.2f}x  -> class weights will be used.')

# -------------------------------------------------
# 1.1.3 Image geometry / quality summary
print('\n--- Image geometry  (all sizes are square) ---')
print(df_img[['width', 'height', 'aspect_ratio', 'file_kb']].describe().round(1).to_string())
print('\nColour modes:', df_img['mode'].value_counts().to_dict())

print('\n--- Per-class visual summaries (brightness 0-255, saturation 0-255) ---')
g = df_img.groupby('class')[['mean_brightness', 'mean_saturation', 'std_brightness', 'file_kb']].mean().round(1).sort_values('mean_brightness')
print(g.to_string())

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
sns.boxplot(data=df_img, x='class', y='mean_brightness', ax=axes[0], palette='viridis')
axes[0].tick_params(axis='x', rotation=45)
axes[0].set_title('Mean brightness per class')
sns.boxplot(data=df_img, x='class', y='mean_saturation', ax=axes[1], palette='magma')
axes[1].tick_params(axis='x', rotation=45)
axes[1].set_title('Mean saturation per class')
save_fig(fig, 'image_brightness_saturation.png')
'''

P1_CODE_IMG_SAMPLES = r'''# -------------------------------------------------
# 1.1.4 Look at real samples per class
fig, axes = plt.subplots(3, 3, figsize=(10, 10))
for ax, cls in zip(axes.ravel(), CLASSES):
    sub = df_img[df_img['class'] == cls]
    sample = sub.iloc[random.randrange(len(sub))]['path']
    im = Image.open(sample).resize((180, 180))
    ax.imshow(im); ax.set_title(cls, fontsize=10); ax.axis('off')
plt.suptitle('Random sample from each RealWaste class', y=1.01)
save_fig(fig, 'image_class_samples.png')

# And one richer grid for the "Plastic vs Glass vs Metal" family which we
# expect to be confusable.
def grid_for(cls_subset, title, n=6):
    rng = random.Random(7)
    rows = []
    for cls in cls_subset:
        sub = df_img[df_img['class'] == cls]
        rows.append(sub.sample(n, random_state=7))
    panel = pd.concat(rows)
    fig, axes = plt.subplots(n, len(cls_subset), figsize=(3 * len(cls_subset), 3 * n))
    for i, cls in enumerate(cls_subset):
        for j, (_, r) in enumerate(panel[panel['class'] == cls].iterrows()):
            im = Image.open(r['path']).resize((120, 120))
            axes[j, i].imshow(im); axes[j, i].axis('off')
            if j == 0:
                axes[j, i].set_title(cls, fontsize=10)
    plt.suptitle(title, y=1.01)
    return fig

fig = grid_for(['Plastic', 'Glass', 'Metal', 'Miscellaneous Trash'],
               'Potentially confusable classes: Plastic vs Glass vs Metal vs Miscellaneous Trash')
save_fig(fig, 'image_confusable_grid.png')
'''

P1_MD_TEXT = r"""## 1.2 Waste description text dataset (`waste_descriptions.csv`)

5,000 resident-style descriptions of waste items with:
`description`, `category`, `disposal_instruction`, `common_confusion`,
`material_composition`.
"""

P1_CODE_TEXT_EDA = r'''# -------------------------------------------------
# 1.2.1 Load and inspect the description data
df_txt = pd.read_csv(TEXT_CSV)
print('Shape:', df_txt.shape)
print('\nDtypes:\n', df_txt.dtypes)
print('\nHead:')
print(df_txt.head(8).to_string())
print('\nMissing values:')
print(df_txt.isna().sum().to_string())

print('\nCategory distribution:')
ct = df_txt['category'].value_counts()
print(ct.to_string())

fig, ax = plt.subplots(figsize=(10, 4.5))
ct.sort_values().plot.barh(ax=ax, color=sns.color_palette('viridis', len(CLASSES)))
ax.set_title('waste_descriptions.csv - category distribution')
ax.set_xlabel('Count')
plt.tight_layout()
save_fig(fig, 'text_class_distribution.png')

# -------------------------------------------------
# 1.2.2 Vocabulary & structure analysis
lengths = df_txt['description'].str.split().str.len()
print('\nDescription word-count: mean %.2f, std %.2f, min %d, max %d' %
      (lengths.mean(), lengths.std(), lengths.min(), lengths.max()))

fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(lengths, bins=range(0, 13), edgecolor='k', color='steelblue')
ax.set_title('Description length (words)')
ax.set_xlabel('words'); ax.set_ylabel('count')
save_fig(fig, 'text_description_length.png')

vocab = set()
df_txt['description'].str.lower().str.findall(r"[a-z']+").apply(lambda t: vocab.update(t))
print(f'Unique lowercase tokens across all descriptions: {len(vocab):,}')

# -------------------------------------------------
from sklearn.feature_extraction.text import CountVectorizer
STOP = set(__import__('sklearn.feature_extraction.text', fromlist=['ENGLISH_STOP_WORDS']).ENGLISH_STOP_WORDS)
top_words = {}
for cls in CLASSES:
    corpus = df_txt[df_txt['category'] == cls]['description'].tolist()
    cv = CountVectorizer(ngram_range=(1, 1), stop_words='english')
    try:
        X = cv.fit_transform(corpus)
    except ValueError:
        continue
    sums = np.asarray(X.sum(axis=0)).ravel()
    idx = np.argsort(sums)[::-1][:8]
    top_words[cls] = [(cv.get_feature_names_out()[i], int(sums[i])) for i in idx]

print('\nTop-8 characteristic tokens per class:')
for cls, toks in top_words.items():
    print(f'{cls:20s} -> ' + ', '.join(f'{w}({c})' for w, c in toks))
'''

P1_CODE_TEXT_COMMON = r'''# -------------------------------------------------
# 1.2.3 Auxiliary text fields (why the extra columns matter for RAG)
print('common_confusion - filled in', df_txt['common_confusion'].notna().sum(), 'of', len(df_txt), 'rows')
print('\nMost frequent confusion notes:')
cc = df_txt['common_confusion'].dropna()
print(Counter(cc).most_common(8))

print('\nUnique disposal instructions:', df_txt['disposal_instruction'].nunique())
print('Most frequent disposal instructions:')
dc = df_txt['disposal_instruction'].value_counts()
print(dc.head(8).to_string())

print('\nUnique material-composition strings:', df_txt['material_composition'].nunique())
mc = df_txt['material_composition'].value_counts()
print(mc.head(8).to_string())

print('\nSample rows for the ambiguous class "Miscellaneous Trash":')
print(df_txt[df_txt['category'] == 'Miscellaneous Trash'].sample(5, random_state=1)[
    ['description', 'disposal_instruction', 'common_confusion']].to_string())
'''

P1_MD_POLICY = r"""## 1.3 Waste policy documents (`waste_policy_documents.json`)

Municipal regulations from *Metro City*. Each record has a `policy_type`,
`categories_covered`, `effective_date`, `jurisdiction` and the raw
`document_text` block. These documents become both the **retrieval corpus**
and the **supervision signal** for the RAG generator.
"""

P1_CODE_POLICY_EDA = r'''# -------------------------------------------------
# 1.3.1 Load and organise the policy documents
with open(POLICY_JSON, 'r', encoding='utf-8') as f:
    policy_raw = json.load(f)
print('Number of policy documents:', len(policy_raw))
pf = pd.DataFrame(policy_raw)
print('\nColumns:', list(pf.columns))
print('\nHead of tabular fields:')
print(pf[['policy_id', 'policy_type', 'categories_covered', 'effective_date',
          'jurisdiction']].head(14).to_string())

pf['n_words'] = pf['document_text'].str.split().str.len()
pf['n_lines'] = pf['document_text'].str.count('\n') + 1
print('\nDocument length stats (words):')
print(pf['n_words'].describe().round(1).to_string())

fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
pf['n_words'].plot.hist(bins=15, ax=axes[0], edgecolor='k', color='seagreen')
axes[0].set_title('Policy document length (words)')
pf['policy_type'].value_counts().sort_values()[:12].plot.barh(ax=axes[1], color='teal')
axes[1].set_title('Policy types')
plt.tight_layout()
save_fig(fig, 'policy_docs_summary.png')

cov = Counter()
for cl in pf['categories_covered']:
    for c in cl:
        cov[c] += 1
print('\nPolicies covering each category:')
print(pd.Series(cov).sort_values(ascending=False).to_string())

# -------------------------------------------------
# 1.3.2 Structure of a document (they use a recurring SKELETON)
print('\n--- Structure of a typical policy document ---')
print(policy_raw[3]['document_text'][:900])
'''

P1_MD_CHALLENGES = r"""### 1.4 Preliminary data-quality findings & challenges

**Images (RealWaste)**
1. **Uniform geometry** — every image is an RGB JPEG, exactly `524 x 524`.
   There is no resolution variation to handle, but real resident photos will
   differ; we therefore still use augmentation (flips, rotations, zoom) so the
   CNN does not overfit to the dataset's particular capture style.
2. **Class imbalance** — `Plastic` (921) is ~2.9x more frequent than
   `Textile Trash` (318). Vanilla training will bias toward frequent classes,
   so we balance the loss with *class weights*.
3. **Confusable material families** — transparent bottles mean *Plastic vs
   Glass* and *Paper vs Cardboard* look similar; *Miscellaneous Trash* is a
   catch-all with internally inconsistent visuals. Expect confusion along
   these pairs; we analyse this explicitly.
4. **Background clutter** — capture-style differences (napkins, hands,
   tables) act as domain noise; brightness/saturation vary strongly which is
   why colour augmentation matters.

**Text (`waste_descriptions.csv`)**
1. **Very short inputs** (mean ~5 words) — almost the whole signal lives in a
   handful of tokens (e.g. "bottle", "crushed", "grounds").
2. **Half of `common_confusion` is empty** (2,504 blanks) — we treat it as an
   optional pre-retrieval hint, never a required field.
3. **Adjective-led phrasing** ("soiled", "wrinkled", "intact") is irrelevant
   to material class and adds noise; we keep noun vocabulary but watch that
   TF-IDF does not latch onto these modifiers.
4. Regional/term variation (see `material_composition`) needs normalisation so
   the classifier and retriever agree.

**Policy documents (JSON)**
1. A small corpus (14 docs) but with a highly *regular skeleton*
   (ACCEPTABLE / NON-ACCEPTABLE / COLLECTION METHOD / PREPARATION /
   BENEFITS) which lets us chunk reliably.
2. Some documents cover **multiple categories** (e.g. Municipal Waste) and a
   few categories are *not* covered by their own dedicated document
   (e.g. Glass colours) — retrieval must therefore handle multi-category
   sections.
3. `categories_covered` gives us *ground-truth tags* to score retrieval.

These findings drive the preprocessing decisions made in the rest of Part 1.
"""

P1_MD_PIPELINES = r"""### 1.5 Data pipelines for every modality

Now we create tidy, reproducible artifacts used by Parts 2-4:

* **Image split** — stratified 70 / 15 / 15 into `data/img_*.csv`, plus a
  `tf.data` pipeline (resize + augment + normalise).
* **Text split** — stratified 80 / 10 / 10 into `data/txt_*.csv` together with
  a light cleaning function used by both classical and neural models.
* **Policy chunking** — the 14 documents are split into per-category sections
  stored in `data/policy_chunks.csv`, ready to become RAG retrieval units.
"""

P1_CODE_PIPELINES = r'''# ------------------------------------------------------
# 1.5.1 Image split (stratified) + tf.data pipeline
from sklearn.model_selection import train_test_split

train_df, tmp = train_test_split(df_img, test_size=0.30, stratify=df_img['class'],
                                 random_state=42)
val_df, test_df = train_test_split(tmp, test_size=0.50, stratify=tmp['class'],
                                   random_state=42)
for name, d in [('img_train', train_df), ('img_val', val_df), ('img_test', test_df)]:
    d[['path', 'class', 'filename']].to_csv(os.path.join(DATA, name + '.csv'), index=False)
print('Image split sizes:', len(train_df), len(val_df), len(test_df))
print('Stratification check (class balances in each split):')
print(pd.DataFrame({
    'train': train_df['class'].value_counts(normalize=True),
    'val':   val_df['class'].value_counts(normalize=True),
    'test':  test_df['class'].value_counts(normalize=True),
}).round(3).to_string())

y_train_img = train_df['class'].map(CLASS_TO_IDX).values
class_weight_dict = {}
for cur in np.unique(y_train_img):
    n = (y_train_img == cur).sum()
    class_weight_dict[cur] = len(y_train_img) / (len(np.unique(y_train_img)) * n)
print('\nComputed image class weights (inverse frequency, normalised):')
print({IDX_TO_CLASS[k]: round(v, 3) for k, v in class_weight_dict.items()})

IMG_SIZE_DEFAULT = 224

def build_img_pipeline(df, img_size, batch_size, augment=False, cache_name=None):
    paths = df['path'].values
    labels = df['class'].map(CLASS_TO_IDX).values.astype(np.int64)
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    def _load(p, y):
        raw = tf.io.read_file(p)
        x = tf.image.decode_jpeg(raw, channels=3)
        x = tf.image.resize(x, (img_size, img_size))
        return x, y
    if augment:
        aug = tf.keras.Sequential([
            tf.keras.layers.RandomFlip('horizontal'),
            tf.keras.layers.RandomRotation(0.15),
            tf.keras.layers.RandomZoom(0.15),
            tf.keras.layers.RandomBrightness(0.15),
            tf.keras.layers.RandomContrast(0.15),
        ], name='aug')
        def _a(x, y):
            x = tf.cast(x, tf.float32)
            return aug(x, training=True), y
    else:
        def _a(x, y):
            return tf.cast(x, tf.float32), y
    def _norm(x, y):
        x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
        return tf.cast(x, tf.float32), y
    ds = ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.map(_a, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.map(_norm, num_parallel_calls=tf.data.AUTOTUNE)
    if cache_name:
        ds = ds.cache(os.path.join(DATA, cache_name))
    ds = ds.shuffle(2048, reshuffle_each_iteration=True) if augment else ds
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds

train_ds224 = build_img_pipeline(train_df, IMG_SIZE_DEFAULT, 32, augment=True, cache_name='cache_train_224')
val_ds224   = build_img_pipeline(val_df,   IMG_SIZE_DEFAULT, 32, augment=False, cache_name='cache_val_224')
test_ds224  = build_img_pipeline(test_df,  IMG_SIZE_DEFAULT, 32, augment=False, cache_name='cache_test_224')
print('Sample batch shapes (train):',
      [t.shape for t in next(iter(train_ds224.unbatch().batch(8)))])

# ------------------------------------------------------
# 1.5.2 Text cleaning + split
def clean_text(s):
    """Lightweight cleaning tuned for resident-style descriptions."""
    if not isinstance(s, str):
        return ''
    s = s.lower()
    s = re.sub(r'https?://\S+|www\.\S+', ' ', s)      # urls
    s = re.sub(r'[\w.+-]+@[\w-]+\.[\w.-]+', ' ', s)   # emails
    s = re.sub(r'[^a-z\s]', ' ', s)                   # punctuation/numbers
    s = re.sub(r'\s+', ' ', s).strip()
    return s

df_txt['clean'] = df_txt['description'].apply(clean_text)
df_txt['n_words'] = df_txt['clean'].str.split().str.len()
print('Clean examples:')
print(df_txt[['description', 'clean']].head(5).to_string())

X_txt = df_txt['clean'].values.astype(str)
y_txt = df_txt['category'].astype('category').cat.codes.values
cat_names = list(df_txt['category'].astype('category').cat.categories)
for i, c in enumerate(cat_names):
    print(i, c)

Xtr, tmpX = train_test_split(np.arange(len(df_txt)), test_size=0.20, stratify=y_txt, random_state=42)
Xv, Xte = train_test_split(tmpX, test_size=0.50, stratify=y_txt[tmpX], random_state=42)
train_idx, val_idx, test_idx = Xtr, Xv, Xte
print('\nText split sizes:', len(train_idx), len(val_idx), len(test_idx))
for name, idx in [('txt_train', train_idx), ('txt_val', val_idx), ('txt_test', test_idx)]:
    pd.DataFrame({'index': idx, 'text': df_txt['clean'].iloc[idx].values,
                  'label': df_txt['category'].iloc[idx].values}).to_csv(
        os.path.join(DATA, name + '.csv'), index=False)

# ------------------------------------------------------
# 1.5.3 Policy documents -> per-category chunks (improved splitter)
def split_sections(doc_text):
    """Split a policy document on its recurring section headers."""
    lines = doc_text.split('\n')
    header_re = re.compile(r'^[A-Za-z][A-Za-z0-9 /\-]*:$')
    sections = []
    cur_head, cur = None, []
    for ln in lines:
        s = ln.strip()
        if s and len(s) <= 45 and header_re.match(s) and not s.startswith('-'):
            if cur_head is not None:
                sections.append((cur_head, '\n'.join(cur).strip()))
            cur_head, cur = s.rstrip(':').strip(), []
        else:
            cur.append(ln)
    if cur_head is not None:
        sections.append((cur_head, '\n'.join(cur).strip()))
    return sections

def assign_category(header, categories):
    h = header.lower()
    if len(categories) == 1:
        return categories[0]
    for c in categories:
        if c.lower() in h:
            return c
    return None  # e.g. GENERAL REQUIREMENTS section

CHUNKS = []
for doc in policy_raw:
    for header, body in split_sections(doc['document_text']):
        tag = assign_category(header, doc['categories_covered'])
        CHUNKS.append({
            'section': header, 'category': tag, 'text': body,
            'policy_id': doc['policy_id'], 'policy_type': doc['policy_type'],
            'categories_covered': ','.join(doc['categories_covered']),
            'effective_date': doc['effective_date'], 'jurisdiction': doc['jurisdiction'],
        })
chunk_df = pd.DataFrame(CHUNKS)
print('Policy chunks created:', chunk_df.shape)
print('\nTagged vs untagged chunks:')
print(chunk_df['category'].fillna('GENERAL').value_counts().to_string())
print('\nA few sample chunks:')
print(chunk_df[['section', 'category', 'text']].head(3).to_string())
print('\nChunk word-count stats:')
print(chunk_df['text'].str.split().str.len().describe().round(1).to_string())
chunk_df.to_csv(os.path.join(DATA, 'policy_chunks.csv'), index=False)
'''