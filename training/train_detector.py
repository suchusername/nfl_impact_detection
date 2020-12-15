import os, sys
import yaml
import json
import numpy as np
import secrets  # for generating random hex code
import importlib.util  # for importing a model

import tensorflow as tf

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from data_processing.Processor import load_detection_ds

CONFIG_PATH = "training/detector_config.yaml"


def numpy_json_converter(obj):
    """
    Allows to dump numpy arrays as json.
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, datetime.datetime):
        return obj.__str__()


def main():

    config_path = os.path.join(ROOT_DIR, CONFIG_PATH)
    with open(config_path, "r") as fd:
        config = yaml.load(fd, Loader=yaml.FullLoader)

    # Keyword arguments for building a model
    kwargs = config["model"]["args"]

    # importing
    module_path = os.path.join(
        ROOT_DIR, "models", config["model"]["name"], config["model"]["build"] + ".py"
    )
    spec = importlib.util.spec_from_file_location(config["model"]["build"], module_path)
    build = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(build)

    # preprocessing
    train_data_path = os.path.join(ROOT_DIR, config["data"]["train"])
    val_data_path = os.path.join(ROOT_DIR, config["data"]["val"])

    train_raw_ds = load_detection_ds(train_data_path)
    val_raw_ds = load_detection_ds(val_data_path)

    processor = build.build_processor(**kwargs)
    train_processed_ds = (
        train_raw_ds.map(processor).batch(config["batch_size"]).prefetch(4)
    )
    val_processed_ds = val_raw_ds.map(processor).batch(config["batch_size"]).prefetch(4)

    # building a model
    model = build.build_model(**kwargs)
    loss = build.build_loss(**kwargs)
    metrics = build.build_metrics(**kwargs)

    # compiling
    optimizer = getattr(tf.keras.optimizers, config["optimizer"]["name"])
    model.compile(
        optimizer=optimizer(config["optimizer"]["lr"]), loss=loss, metrics=metrics,
    )

    # generating model code
    model_code = secrets.token_hex(16)

    # callbacks
    model_save_path = os.path.join(
        ROOT_DIR, "models", config["model"]["name"], "snapshots", model_code
    )
    os.makedirs(model_save_path, exist_ok=True)

    lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=config["callbacks"]["lr_scheduler"]["factor"],
        patience=config["callbacks"]["lr_scheduler"]["patience"],
        cooldown=config["callbacks"]["lr_scheduler"]["cooldown"],
        min_lr=1e-7,
        verbose=True,
    )

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=config["callbacks"]["early_stop"]["patience"]
    )

    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        os.path.join(model_save_path, "weights.hdf5"),
        monitor="val_loss",
        verbose=1,
        save_best_only=True,
        mode="min",
    )

    callbacks = [lr_scheduler, early_stop, checkpoint]

    # training
    history = model.fit(
        train_processed_ds,
        epochs=config["epochs"],
        validation_data=val_processed_ds,
        callbacks=callbacks,
    )

    # saving history
    with open(os.path.join(model_save_path, "history.json"), "w") as fd:
        history.history["train_config"] = config
        json.dump(history.history, fd, indent=4, default=numpy_json_converter)


if __name__ == "__main__":
    main()
