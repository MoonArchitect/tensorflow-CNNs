import hashlib
import os
import yaml
import argparse


def parse_args():
	parser = argparse.ArgumentParser()

	parser.add_argument("cfg", help="yaml config file")
	parser.add_argument("--checkpoint", help="yaml progress (checkpoint) file", default="checkpoint.yaml")

	return parser.parse_args()


def parse_flags(models, quotes = True): # TODO swap double quotes for single quotes inside arguments
	serialized = []

	if not type(models) == list:
		models = [ models ]

	for model in models:
		command = ""

		for flag, value in model.items():
			if type(value) == bool:
				if value:
					command += f" --{flag}"
			elif quotes and type(value) == str:
				command += f" --{flag}=\"{value}\""
			elif value == None:
				pass
			else:
				command += f" --{flag}={value}"

		serialized.append(command)
		
	return serialized


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
	
	assert not overrides or flag != "models", f"Encountered an unexpected flag '{flag}:{cfg[flag]}' in override dictionary"
	
	assert overrides or flag != "model", f"Encountered an unexpected flag '{flag}:{cfg[flag]}' in base dictionary"

	assert cfg[flag] == None or type(cfg[flag]) == flag_types[flag], f"Flag '{flag}' is of incorrect type '{type(cfg[flag])}, expected '{flag_types[flag]}'"
	
	return True 


def main():
	# get file
	args = parse_args()

	# read cfg
	with open(args.cfg) as file:
		cfg = yaml.load(file, Loader=yaml.FullLoader)

	# build all model configurations
	models = []
	if not "models" in cfg:
		raise ValueError(f"models flag was not found in {args.cfg} -> cfg: {cfg}")
	
	base = {}
	for flag in cfg:
		if flag == "models" or not is_flag_valid(cfg, flag): continue
		base[flag] = cfg[flag]

	assert type(cfg["models"]) == list, "Models should be a list of configurations"

	for config in cfg["models"]:
		overrides = base.copy()

		override_info = ""
		for flag in config:
			if flag == "model" or not is_flag_valid(config, flag, True): continue
			if flag in overrides:
				override_info += f"\tOverriding {flag} from {overrides[flag]} to {config[flag]}\n"
			overrides[flag] = config[flag]
		
		if override_info != "":
			print(config["model"])
			print(override_info)
			
		for name in list(config["model"]):
			models.append( overrides.copy() )
			models[-1]["model"] = name

	
	arguments = parse_flags(models)
	clean_arguments = parse_flags(models, False)

	# train -> if checkpoint is given, read, 
	# read progress file
	# TODO check if yaml file
	open(args.checkpoint, "a+").close()
	with open(args.checkpoint, "r+") as file:
		checkpoint = yaml.load(file, Loader=yaml.FullLoader)

	
	ids = [hashlib.md5(("train.py" + clean + f" --checkpoint={args.checkpoint}").encode()).hexdigest() for clean in clean_arguments]
	
	for model, id in zip(arguments, ids):
		if not checkpoint is None and id in checkpoint and checkpoint[id]["status"] == "Finished":
			continue
		
		command = "python train.py" + model + " --dryrun" + " --full_info"
		print(f"\n{'-'*100}  Dry run: {id}  {'-'*100}")
		print(f"Running:   {command}\n\n")
		
		os.system(command)

		print(f"\n{'-'*300}\n")

	for model, id in zip(arguments, ids):
		if not checkpoint is None and id in checkpoint and checkpoint[id]["status"] == "Finished":
			continue
		
		command = "python train.py" + model + f" --checkpoint=\"{args.checkpoint}\""
		print(f"\n{'-'*300}")
		print(f"Id:   {id}")
		print(f"Running:   {command}\n\n")
		
		os.system(command)

		print(f"\n{'-'*300}\n")

	# print(cfg)

if __name__ == "__main__":
	main()
	

