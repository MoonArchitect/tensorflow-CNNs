import sys
from tensorflow.keras import optimizers

models = {}

lr_schedules = {}

optimzers = {
	"sgd": optimizers.SGD
}


def register_model(fn):
	name = fn.__name__
	module = sys.modules[fn.__module__]

	if hasattr(module, '__all__'):
		module.__all__.append(name)
	else:
		module.__all__ = [ name ]
	
	name = name.lower()
	if name in models:
		raise ValueError(f"Model '{name}' is already registered in '{module}'")
	models[name] = fn

	return fn


def get_all_models():
	return list(models.keys())


def is_model(name):
	return name in models


def create_model(name, **kwargs):
	if not "Models" in sys.modules:
		import Models

	name = name.lower()
	if is_model(name):
		return models[name](**kwargs)
	else:
		raise ValueError(f"Model '{name}' is not registered. Available: {list(models.keys())}")



def create_optimzer(name, **kwargs):	
	name = name.lower()
	if name in optimzers:
		return optimzers[name](**kwargs)
	else:
		raise ValueError(f"Optimzer '{name}' is not registered. Available: {list(optimzers.keys())}")




def register_schedule(fn):
	name = fn.__name__.lower()
	module = sys.modules[fn.__module__]
	
	if name in lr_schedules:
		raise ValueError(f"lr_schedule '{name}' is already registered in '{module}'")
	lr_schedules[name] = fn

	return fn

def create_lr_schedule(name, **kwargs):
	if not "LearningRate" in sys.modules:
		import utils.LearningRate

	name = name.lower()
	if name in lr_schedules:
		return lr_schedules[name](**kwargs)
	else:
		raise ValueError(f"lr_schedule '{name}' is not registered. Available: {list(lr_schedules.keys())}")



