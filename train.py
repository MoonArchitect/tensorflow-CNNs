import os
import sys
import hashlib
import argparse
from datetime import datetime
from tensorboard.plugins.hparams.api import KerasCallback

import tensorflow as tf
import utils
import data


# TODO try to recover if loss is NaN


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, )
    parser.add_argument("--optimizer", type=str, required=True, )
    parser.add_argument("--lr_schedule", type=str, required=True, )

    parser.add_argument("--batch_size", type=int, required=True, )
    parser.add_argument("--epochs", type=int, required=True, )
    parser.add_argument("--data", type=str, required=True, )

    parser.add_argument("--name", type=str, default=None, help="" )
    parser.add_argument("--id", type=str, default=None, help="" )

    parser.add_argument("--checkpoint", type=str, )

    parser.add_argument("--xla", dest="xla", action="store_true")
    # parser.add_argument("--no-xla", dest="xla", action="store_false")

    parser.add_argument("--fp16", dest="fp16", action="store_true")
    # parser.add_argument("--no-fp16", dest="fp16", action="store_false")

    parser.add_argument("--augment", choices=["cutmix", "cutout", "mixup"], default=False)
    
    parser.add_argument("--dryrun", action="store_true")
    parser.add_argument("--full_info", action="store_true")

    return parser.parse_args()



def parse_code(code):
    try:
        if "(" in code:
            name, params = code.split("(", 1)
            name = name.split(".")[-1]
            kwargs = eval("dict(" + params)
        else:
            name = code.split(".")[-1]
            kwargs = {}

    except Exception as e:
        print(f"\nERROR     -     {e.__class__} exception occured during an attempt to parse {code}")
        print("Expected syntax: \n")
        raise e

    return name, kwargs



def dry_run():
    print("\ndryrun\n")



def train(model,
          optimzer,
          lr_schedule,
          data_dir: str,
          batch_size: int,
          epochs: int,
          hparams: dict,
          name: str,
          model_id: str,
          adv_augment: str = None,
          checkpoint: str = None ):

    log_dir = os.path.join("logs", name)
    callbacks = [
        lr_schedule,

        KerasCallback( os.path.join(log_dir, "validation"), hparams = hparams ),

        tf.keras.callbacks.TensorBoard(log_dir = log_dir, profile_batch = '300, 310'),

        tf.keras.callbacks.experimental.BackupAndRestore(f"backup/{name}"),

        tf.keras.callbacks.ModelCheckpoint(
            f"Checkpoints/{name}/best_{model.name}",
            save_weights_only=True, save_best_only=True, monitor='val_accuracy', mode='max',
        ),

        tf.keras.callbacks.TerminateOnNaN()
    ]

    if checkpoint is not None:
        callbacks.append(
            utils.Callbacks.ProgressCheckpoint(checkpoint, model_id,  hparams)
        )

    train_ds, val_ds = data.read_dataset(path=data_dir)

    train_ds, val_ds = data.prepare_cifar10(
        train_ds, val_ds,
        batch_size=batch_size,
        adv_augment=adv_augment
    )

    model.compile(
        loss=tf.keras.losses.CategoricalCrossentropy(True),
        optimizer=optimzer,
        metrics=['accuracy']
    )

    # log hparams
    for k, v in hparams.items():
        print(f"{k}: {v}")


    model.fit(
        train_ds,
        epochs = epochs,
        steps_per_epoch = 50000 // batch_size,
        validation_data = val_ds,
        callbacks = callbacks
    )


def main():
    args = parse_args()

    if not args.full_info:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    if args.xla:
        os.environ['TF_XLA_FLAGS'] = "--tf_xla_auto_jit=3"

    if args.fp16:
        tf.keras.mixed_precision.set_global_policy('mixed_float16')

    model_name, model_kwargs = parse_code(args.model)
    optimizer_name, optimizer_kwargs = parse_code(args.optimizer)
    lr_schedule_name, lr_schedule_kwargs = parse_code(args.lr_schedule)
    
    model = utils.creator.create_model(model_name, **model_kwargs)
    optimizer = utils.creator.create_optimzer(optimizer_name, **optimizer_kwargs)
    lr_schedule = utils.creator.create_lr_schedule(lr_schedule_name, **lr_schedule_kwargs)

    if args.name is None:
        args.name = model.name + datetime.now().strftime("@%m%d%H%M")
    
    if args.id is None:
        args.id = hashlib.md5(" ".join(sys.argv).encode()).hexdigest()

    if args.dryrun:
        dry_run()
    else:
        train(
            model = model,
            lr_schedule = lr_schedule,
            optimzer = optimizer,
            data_dir = args.data,
            batch_size = args.batch_size,
            epochs = args.epochs,
            adv_augment = args.augment,
            checkpoint = args.checkpoint,
            name = args.name,
            model_id = args.id,
            hparams = {
                "name": args.name,
                "model": args.model,
                "batch_size": args.batch_size,
                "xla": args.xla,
                "fp16": args.fp16,
                "augment": args.augment if args.augment else "None",  # TODO Fix this. hparams can't mix str/bool values, does
                                                                      # not allow None values, but these are used in prepareDatasets
                "optimzer": args.optimizer,
                "lr_schedule": args.lr_schedule,
            }
        )


if __name__ == "__main__":
    main()
