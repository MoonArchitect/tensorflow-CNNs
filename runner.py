import os
import hashlib
import argparse
import yaml


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("cfg", help="yaml config file")
    parser.add_argument("--checkpoint", help="yaml progress (checkpoint) file", default="checkpoint.yaml")

    return parser.parse_args()


def is_flag_valid(cfg, flag, overrides = False):
    flag_types = {
        "models": list,
        "model": str,
        "name": str,
        "data": str,
        "batch_size": int,
        "epochs": int,
        "optimizer": str,
        "lr_schedule": str,
        "xla": bool,
        "fp16": bool,
        "augment": str,
    }

    assert flag in flag_types, f"Encountered unknown flag - '{flag}'"
    
    assert not overrides or flag != "models", \
        f"Encountered an unexpected flag '{flag}:{cfg[flag]}' in override dictionary"
    
    assert overrides or flag != "model", \
        f"Encountered an unexpected flag '{flag}:{cfg[flag]}' in base dictionary"

    assert cfg[flag] is None or isinstance(cfg[flag], flag_types[flag]), \
        f"Flag '{flag}' is of incorrect type '{type(cfg[flag])}, expected '{flag_types[flag]}'"


def parse_config(path):
    # read cfg
    with open(path) as file:
        cfg = yaml.load(file, Loader=yaml.FullLoader)

    # build all model configurations
    configs = []
    if "models" not in cfg:
        raise ValueError(f"models flag was not found in {path} -> cfg: {cfg}")
    
    base = {}
    for flag in cfg:
        if flag == "models":
            continue
        is_flag_valid(cfg, flag)
        base[flag] = cfg[flag]

    assert isinstance(cfg["models"], list), "'models' should be a list of configurations"
    
    for config in cfg["models"]:
        overrides = base.copy()
        override_info = ""

        for flag in config:
            if flag == "model":
                continue

            is_flag_valid(config, flag, True)
            
            if flag in overrides:
                override_info += f"\tOverriding {flag} from {overrides[flag]} to {config[flag]}\n"
            overrides[flag] = config[flag]
        
        if override_info != "":
            print(config["model"])
            print(override_info)
            
        for name in list(config["model"]):
            configs.append( overrides.copy() )
            configs[-1]["model"] = name
    
    return configs


def parse_flags(configs, quotes = True):  # TODO swap double quotes for single quotes inside arguments
    serialized = []

    if not isinstance(configs, list):
        configs = [ configs ]

    for model in configs:
        command = ""

        for flag, value in model.items():
            if isinstance(value, bool):
                if value:
                    command += f" --{flag}"
            elif quotes and isinstance(value, str):
                command += f" --{flag}=\"{value}\""
            elif value is None:
                pass
            else:
                command += f" --{flag}={value}"

        serialized.append(command)
        
    return serialized


def parse_commands(configs, checkpoint_dir):
    commands = []
    arguments = parse_flags(configs)
    
    clean_arguments = parse_flags(configs, False)
    ids = [ hashlib.md5(("train.py" + clean + f" --checkpoint={checkpoint_dir}").encode()).hexdigest() for clean in clean_arguments ]

    open(checkpoint_dir, "a+").close()  # TODO check if yaml file
    with open(checkpoint_dir, "r+") as file:
        checkpoint = yaml.load(file, Loader=yaml.FullLoader)

    for flags, id in zip(arguments, ids):
        command = f'python train.py {flags} --checkpoint="{checkpoint_dir}"'
        
        if checkpoint is not None and id in checkpoint:
            if checkpoint[id]["status"] == "Finished":  # TODO check if "status" is present
                continue
            
            command += f" --id={id} --name={checkpoint[id]['cfg']['name']}"  # TODO check if fields exist
        
        commands.append(command)

    return [ c + " --dryrun --full_info" for c in commands ] + commands


def main():
    args = parse_args()

    configurations = parse_config(args.cfg)
    
    commands = parse_commands(configurations, args.checkpoint)
    
    for command in commands:
        print(f"Running:   {command}\n\n")
        
        os.system(command)

        print(f"\n{'-'*135}\n")



if __name__ == "__main__":
    main()
    

