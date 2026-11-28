import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from preprocessing import preprocess_image
from dataset import sample_image

def train_model(df, method, epochs, batch_size, img_size, limit_per_class=0):
    # TensorFlow is imported only when training is requested.
    import tensorflow as tf
    labels = ["MEL", "NV", "BKL"]
    work = df.copy()
    if limit_per_class and limit_per_class > 0:
        work = work.groupby("label", group_keys=False).apply(
            lambda x: x.sample(min(len(x), limit_per_class), random_state=42)
        ).reset_index(drop=True)
    x, y = [], []
    for _, row in work.iterrows():
        image = sample_image(row["path"])
        if image is None:
            continue
        image = preprocess_image(image, method)
        image = tf.image.resize(image, (img_size, img_size)).numpy()
        x.append(image.astype("float32") / 255.0
        )
        y.append(labels.index(row["label"]))
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)
    if len(x) < 15:
        raise ValueError("Data terlalu sedikit untuk split train/validation/test.")
    x_train, x_temp, y_train, y_temp = train_test_split(
        x, y, test_size=0.30, random_state=42, stratify=y
    )
    x_val, x_test, y_val, y_test = train_test_split(
        x_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(img_size, img_size, 3)),
        tf.keras.layers.Conv2D(32, 3, activation="relu", padding="same"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, 3, activation="relu", padding="same"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(128, 3, activation="relu", padding="same"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.35),
        tf.keras.layers.Dense(3, activation="softmax")
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    history = model.fit(
        x_train, y_train, validation_data=(x_val, y_val),
        epochs=epochs, batch_size=batch_size, verbose=0
    )
    probs = model.predict(x_test, verbose=0)
    pred = probs.argmax(axis=1)
    report = classification_report(y_test, pred, labels=[0,1,2], target_names=labels, zero_division=0)
    return {
        "model": model, "history": history.history, "test_accuracy": accuracy_score(y_test, pred),
        "y_test": y_test, "pred": pred, "report": report, "class_codes": labels
    }
