import sys
import hashlib
from datetime import datetime

import yaml
from tensorflow.keras.callbacks import Callback


__all__ = ['ProgressCheckpoint']


class ProgressCheckpoint(Callback):
    """
    ProgressCheckpoint
    """
    def __init__(self, path, hparams):
        super().__init__()
        
        open(path, "a+").close()
        self.file = open(path, "r+")
        self.yaml = yaml.load(self.file, Loader=yaml.FullLoader)

        command = " ".join(sys.argv[0:])
        self.id = hashlib.md5(command.encode()).hexdigest()

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
        

    def on_train_end(self, logs=None):
        # TODO remove backup/model_name_id folder

        self.status["end"][-1] = datetime.now().strftime("%m/%d|%H:%M:%S")
        self.status["status"] = "Finished"
        
        self.dump_yaml()
        self.file.close()


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
