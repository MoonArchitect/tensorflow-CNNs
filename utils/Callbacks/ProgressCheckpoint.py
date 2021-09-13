import os
import sys
from datetime import datetime

import yaml
from tensorflow.keras.callbacks import Callback


__all__ = ['ProgressCheckpoint']


class ProgressCheckpoint(Callback):
    """
    ProgressCheckpoint
    """
    def __init__(self, path, model_id, hparams):
        super().__init__()
        
        if path[-5:] != ".yaml":
            path += ".yaml"

        open(path, "a+").close()  # TODO check if file exists first
        self.file = open(path, "r+")
        self.yaml = yaml.load(self.file, Loader=yaml.FullLoader)

        command = " ".join(sys.argv[0:])
        self.id = model_id

        if self.yaml is None:
            self.yaml = {}

        if self.id not in self.yaml:
            self.yaml[self.id] = {
                "cfg": {
                    "command": command,
                    **hparams
                },
                "status": "Initialized",
                "epoch": -1,
                "best": {
                    "val-accuracy": -1,
                    "val-loss": -1
                },
                "start": [],
                "end": [],
                "error": None  # TODO finish error functionality
            }
            self.dump_yaml()
        
        self.status = self.yaml[self.id]
        
    
    def on_train_begin(self, logs=None):
        self.status["start"].append( datetime.now().strftime("%m/%d|%H:%M:%S") )
        self.status["end"].append( datetime.now().strftime("%m/%d|%H:%M:%S") )
        self.status["status"] = "Training"
        

    def on_train_end(self, logs=None):  # TODO called even after stopping due to NaN loss
        self.remove_backup()

        self.status["end"][-1] = datetime.now().strftime("%m/%d|%H:%M:%S")
        self.status["status"] = "Finished"
        
        self.dump_yaml()
        self.file.close()


    def remove_backup(self):
        backup_path = f"backup/{self.status['cfg']['name']}"
        if os.path.exists(backup_path):
            if len(os.listdir(backup_path)) == 0:
                os.rmdir(backup_path)
            else:
                print(f"Couldn't delete '{backup_path}''. Folder is not empty")
        else:
            print(f"Couldn't find '{backup_path}'")


    def on_epoch_end(self, epoch, logs=None):
        if epoch == 0:
            self.status["start"][-1] = datetime.now().strftime("%m/%d|%H:%M:%S")

        if self.status["best"]["val-accuracy"] < round( logs["val_accuracy"] * 100, 2 ):
            self.status["best"]["val-accuracy"] = round( logs["val_accuracy"] * 100, 2 )
            self.status["best"]["val-loss"] = round( logs["val_loss"], 4 )

        self.status["end"][-1] = datetime.now().strftime("%m/%d|%H:%M:%S")

        self.status["epoch"] = epoch + 1

        self.dump_yaml()


    def dump_yaml(self):
        self.file.truncate(0)
        self.file.seek(0)
        yaml.dump(self.yaml, self.file, sort_keys=False, default_flow_style=False, width=1000)
