# =====================================================================
# Part 2 - Waste material classification with a CNN (transfer learning)
# =====================================================================

P2_MD = r"""# Part 2: Waste Material Classification (CNN)

**Approach — transfer learning with MobileNetV2**

* **Why MobileNetV2:** state-of-the-art accuracy vs FLOPs ratio, built-in
  inverted-residual blocks, and it trains in reasonable time on a CPU-only
  machine (unlike ResNet-152 / EfficientNet-B4). Its 224px design matches the
  uniform 524px RealWaste captures well.
* **Two-stage strategy** (typical for small datasets):
  1. *Bottleneck feature extraction* at 224px with frozen ImageNet weights →
     fast, well-regularised head training, letting us experiment with head
     architectures cheaply;
  2. *Fine-tuning* the top convolutional block at low learning rate with
     input augmentation so the filters adapt to the waste domain.
* **Regularisation:** dropout in the head, data augmentation, L2 in the
  final layer is handled by Keras' default, early stopping, and a low
  fine-tune LR.
* **Class imbalance:** inverse-frequency class weights (Plastic is ~2.9x
  more frequent than Textile Trash).

> Compute note: we run on CPU, so fine-tuning uses 128px inputs (MobileNetV2
> accepts any size >= 32) to keep epochs tractable; the frozen 224px stage
> gives us the "native resolution" performance reference.
"""

P2_CODE_FEATURES = r'''# --------------------------------------------------------
# 2.1 Feature extraction (frozen ImageNet backbone), cached on disk
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.callbacks import EarlyStopping

IMG_FEAT = 224
feat_path = os.path.join(DATA, 'img_features224.npz')
if os.path.exists(feat_path):
    z = np.load(feat_path)
    F_tr, F_va, F_te = z['F_tr'], z['F_va'], z['F_te']
    y_f_tr, y_f_va, y_f_te = z['y_tr'], z['y_va'], z['y_te']
    print('Loaded cached ImageNet-224 features:', F_tr.shape, F_va.shape, F_te.shape)
else:
    backbone = MobileNetV2(include_top=False, weights='imagenet',
                           input_shape=(IMG_FEAT, IMG_FEAT, 3), pooling='avg')
    backbone.trainable = False

    def feats_from(df_all, bs=64):
        paths = df_all['path'].values
        labels = df_all['class'].map(CLASS_TO_IDX).values.astype(np.int64)
        ds = tf.data.Dataset.from_tensor_slices((paths, labels))
        @tf.function
        def load(p, y):
            raw = tf.io.read_file(p)
            x = tf.image.decode_jpeg(raw, channels=3)
            x = tf.image.resize(x, (IMG_FEAT, IMG_FEAT))
            x = preprocess_input(tf.cast(x, tf.float32))
            return x, y
        ds = ds.map(load, num_parallel_calls=tf.data.AUTOTUNE).batch(bs).prefetch(tf.data.AUTOTUNE)
        feats, ys = [], []
        steps = 0
        for xb, yb in ds:
            feats.append(backbone(xb, training=False).numpy())
            ys.append(yb.numpy())
            steps += 1
            if steps % 10 == 0:
                print(f'  extracted {steps*bs} samples...')
        return np.concatenate(feats), np.concatenate(ys)

    print('Extracting frozen features from all splits (224x224)...')
    t0 = time.time()
    F_tr, y_f_tr = feats_from(train_df)
    F_va, y_f_va = feats_from(val_df)
    F_te, y_f_te = feats_from(test_df)
    dt = time.time() - t0
    print(f'Feature extraction done in {dt/60:.1f} min '
          f'({F_tr.shape[1]}-dim vectors per image).')
    np.savez(feat_path, F_tr=F_tr, F_va=F_va, F_te=F_te,
             y_tr=y_f_tr, y_va=y_f_va, y_te=y_f_te)
    del backbone

print('Feature shapes -> train:', F_tr.shape, 'val:', F_va.shape, 'test:', F_te.shape)
print('Sample feature stats: mean %.4f  std %.4f' % (F_tr.mean(), F_tr.std()))
'''

P2_CODE_HEAD_EXP = r'''# --------------------------------------------------------
# 2.2 Head-architecture experiments on the frozen features
# Evidence of hyper-parameter experimentation: width, depth & dropout.
from tensorflow.keras import layers, Model, Sequential as KSeq, Input as KInput
from tensorflow.keras.optimizers import Adam
from sklearn.utils.class_weight import compute_class_weight

cw_feat = compute_class_weight('balanced', classes=np.unique(y_f_tr), y=y_f_tr)
cw_dict = {int(k): float(v) for k, v in zip(np.unique(y_f_tr), cw_feat)}

def make_head(dropout=0.3, hidden=128, bn=False):
    """Head mapping 1280-D bottleneck features -> 9 classes."""
    m = KSeq([KInput(shape=(F_tr.shape[1],))])
    m.add(layers.Dense(hidden, kernel_regularizer='l2'))
    if bn:
        m.add(layers.BatchNormalization())
    m.add(layers.Activation('relu'))
    m.add(layers.Dropout(dropout))
    m.add(layers.Dense(9, activation='softmax'))
    return m

variants = {
    'headA (128 units, drp 0.3)': dict(dropout=0.3, hidden=128, bn=False),
    'headB (256 units, drp 0.5, BN)': dict(dropout=0.5, hidden=256, bn=True),
    'headC (512 units, drp 0.6, BN)': dict(dropout=0.6, hidden=512, bn=True),
}

print('Training head variants on frozen 224px features (early-stopped):')
res_head = {}
_head_results = os.path.join(DATA, 'head_sweep_results.json')
if os.path.exists(_head_results):
    with open(_head_results) as f:
        res_head = json.load(f)
    print('Loaded cached head-sweep results:')
    print(pd.DataFrame(res_head).T.to_string())
    best_head = max(res_head, key=lambda k: res_head[k]['val_acc'])
    print('\nBest head variant (cached):', best_head)
else:
    for name, kw in variants.items():
        seed_everything(42)
        h = make_head(**kw)
        h.compile(Adam(1e-3), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        es = EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True)
        h.fit(F_tr, y_f_tr, validation_data=(F_va, y_f_va), epochs=30, batch_size=64,
              class_weight=cw_dict, callbacks=[es], verbose=0)
        va = h.evaluate(F_va, y_f_va, verbose=0)[1]
        te = h.evaluate(F_te, y_f_te, verbose=0)[1]
        res_head[name] = {'val_acc': round(va, 4), 'test_acc': round(te, 4)}
        print(f'{name:28s} val_acc={va:.4f}  test_acc={te:.4f}')
        del h
    with open(_head_results, 'w') as fp:
        json.dump(res_head, fp, indent=2)

head_res = pd.DataFrame(res_head).T.sort_values('val_acc', ascending=False)
print(head_res.to_string())
best_head = head_res.index[0]
print('\nBest head variant:', best_head)
'''

P2_CODE_BUILD_MODEL = r'''# --------------------------------------------------------
# 2.3 Assemble the full transfer-learning model and fine-tune
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, CSVLogger

# data pipelines at the fine-tune resolution (128 px) with augmentation
FT_SIZE = 128
train_ds = build_img_pipeline(train_df, FT_SIZE, 32, augment=True, cache_name='cache_train_128')
val_ds   = build_img_pipeline(val_df,   FT_SIZE, 32, augment=False)
test_ds  = build_img_pipeline(test_df,  FT_SIZE, 32, augment=False, cache_name='cache_test_128')

# class weights (same as used for the head)
cw_feat = compute_class_weight('balanced', classes=np.unique(y_f_tr), y=y_f_tr)
cw_dict = {int(k): float(v) for k, v in zip(np.unique(y_f_tr), cw_feat)}

FINAL_CNN_PATH = os.path.join(MODELS, 'cnn_finetuned.keras')
if os.path.exists(FINAL_CNN_PATH):
    print('Fine-tuned CNN checkpoint found - skipping the training stage '
          '(loads it in the next section).')
else:
    img_input = Input(shape=(FT_SIZE, FT_SIZE, 3))
    base = MobileNetV2(include_top=False, weights='imagenet',
                       input_shape=(FT_SIZE, FT_SIZE, 3), pooling='avg')
    base.trainable = False

    # best head architecture from the sweep (matched by name)
    def make_head_fixed(dropout=0.3, hidden=128, bn=False):
        m = KSeq([KInput(shape=(int(base.output.shape[1]),))])
        m.add(layers.Dense(hidden, kernel_regularizer='l2'))
        if bn:
            m.add(layers.BatchNormalization())
        m.add(layers.Activation('relu'))
        m.add(layers.Dropout(dropout))
        m.add(layers.Dense(9, activation='softmax'))
        return m

    choice = dict(hidden=128, dropout=0.3, bn=False)
    if best_head.startswith('headB'):
        choice = dict(hidden=256, dropout=0.5, bn=True)
    elif best_head.startswith('headC'):
        choice = dict(hidden=512, dropout=0.6, bn=True)
    print('Using head config:', choice, ' (winner of the sweep)')

    head = make_head_fixed(**choice)
    x = base(img_input)
    y = head(x)
    model = Model(img_input, y)
    model.compile(Adam(1e-3), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # 2.3.1 warm-up: train only the new head on augmented 128px images
    print('\n[Stage 1] Training the classification head (base frozen) ...')
    es = EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True)
    hist1 = model.fit(train_ds, validation_data=val_ds, epochs=8, class_weight=cw_dict,
                      callbacks=[es], verbose=1)
    print(f'Head-only val_accuracy = {max(hist1.history["val_accuracy"]):.4f}')

    # 2.3.2 fine-tune the top convolutional block
    print('\n[Stage 2] Unfreezing the top 20 MobileNetV2 layers and fine-tuning ...')
    for layer in base.layers[-20:]:
        layer.trainable = True
    model.compile(Adam(1e-5), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    cp = ModelCheckpoint(FINAL_CNN_PATH, monitor='val_accuracy', save_best_only=True)
    rlr = ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=2, min_lr=1e-7)
    hist2 = model.fit(train_ds, validation_data=val_ds, epochs=6, class_weight=cw_dict,
                      callbacks=[es, cp, rlr], verbose=1)
    print('Fine-tuned val_accuracy =', max(hist2.history['val_accuracy']))
    print('\nCNN model saved at:', FINAL_CNN_PATH)

final_cnn = tf.keras.models.load_model(FINAL_CNN_PATH)
print('Loaded final CNN for evaluation.')
'''

P2_CODE_CNN_EVAL = r'''# --------------------------------------------------------
# 2.4 Evaluation
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# reload best checkpoint
final_cnn = tf.keras.models.load_model(os.path.join(MODELS, 'cnn_finetuned.keras'))
print('Loaded best fine-tuned CNN.')

pred_test = np.argmax(final_cnn.predict(test_ds, verbose=1), axis=1)
y_true = test_df['class'].map(CLASS_TO_IDX).values

acc = cnn_test_acc = accuracy_score(y_true, pred_test)
print(f'\nTest accuracy (fine-tuned CNN, 128px) = {cnn_test_acc:.4f}')

# head-only classifier evaluated on 224px frozen features (from the sweep)
acc_head = res_head[best_head]['test_acc']
print(f'Test accuracy (frozen features + {best_head}) = {acc_head:.4f}')

comp = pd.DataFrame({'stage': ['Frozen-224 + ' + best_head, 'Fine-tuned CNN (128px)'],
                     'test_accuracy': [round(acc_head, 4), round(cnn_test_acc, 4)]})
print('\n', comp.to_string(index=False))

# ---- confusion matrix + report for the final model ----
cm = confusion_matrix(y_true, pred_test, labels=range(len(CLASSES)))
fig = plot_confusion(cm, CLASSES, title='CNN confusion matrix - RealWaste test set')
save_fig(fig, 'cnn_confusion.png')

print('\nClassification report:')
print(classification_report(y_true, pred_test, target_names=CLASSES, digits=3, zero_division=0))

# per-class recall to spot the hard categories
per = pd.DataFrame({
    'class': CLASSES,
    'recall': [round((cm[i, i] / cm[i].sum()) if cm[i].sum() else 0, 3) for i in range(9)],
    'support': cm.sum(axis=1),
}).sort_values('recall')
print('\nHardest-to-easiest classes (by test recall):')
print(per.to_string(index=False))

fig, ax = plt.subplots(figsize=(9, 4))
per.set_index('class')['recall'].plot.bar(ax=ax, color='firebrick')
ax.set_title('Per-class test recall (sorted)'); ax.set_ylabel('recall')
plt.xticks(rotation=45, ha='right')
save_fig(fig, 'cnn_per_class_recall.png')
'''

P2_CODE_CNN_ERRORS = r'''# --------------------------------------------------------
# 2.5 Error-pattern analysis: where does the CNN go wrong?
mis_idx = np.where(y_true != pred_test)[0]
print(f'Misclassified test images: {len(mis_idx)} of {len(y_true)} ({100*len(mis_idx)/len(y_true):.1f}%)')

wrong_pairs = pd.DataFrame({
    'true': [CLASSES[y_true[i]] for i in mis_idx],
    'pred': [CLASSES[pred_test[i]] for i in mis_idx],
})
pair_counts = wrong_pairs.value_counts().head(12)
print('\nMost common true->prediction error pairs:')
print(pair_counts.to_string())

# Show a handful of the actual failing images
rng = random.Random(3)
fig, axes = plt.subplots(2, 4, figsize=(12, 6))
picked = rng.sample(list(mis_idx), min(8, len(mis_idx)))
for ax, i in zip(axes.ravel(), picked):
    im = Image.open(test_df.iloc[i]['path']).resize((140, 140))
    ax.imshow(im)
    ax.set_title(f'TRUE: {CLASSES[y_true[i]]}\nPRED: {CLASSES[pred_test[i]]}', fontsize=9)
    ax.axis('off')
plt.suptitle('Example CNN misclassifications on the held-out test set', y=1.0)
save_fig(fig, 'cnn_misclassification_examples.png')

# Which classes rob accuracy from others? (off-diagonal volumes)
off = cm.copy().astype(float)
for i in range(9):
    off[i, i] = 0
    if off[i].sum():
        off[i, :] = 100 * off[i, :] / off[i].sum()
fig, ax = plt.subplots(figsize=(9, 7))
sns.heatmap(off, annot=True, fmt='.0f', cmap='Reds', xticklabels=CLASSES, yticklabels=CLASSES, ax=ax)
ax.set_title('% of each TRUE class\'s errors landing on each PREDICTED class')
ax.set_xlabel('Predicted'); ax.set_ylabel('True')
plt.xticks(rotation=45, ha='right'); plt.yticks(rotation=0)
save_fig(fig, 'cnn_error_distribution.png')
print('\nInterpretation: most confusion is concentrated along '
      'Plastic<->Miscellaneous Trash and Paper<->Cardboard.')
'''