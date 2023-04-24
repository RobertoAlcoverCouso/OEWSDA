from trainer import Trainer
import argparse
import yaml
import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description="Code for On exploring weakly supervised domain adaptation strategies")
parser.add_argument("-c", "--config", default="", type=str)
parser.add_argument("-m", "--model_root", default="", type=str)
parser.add_argument("-a", "--architecture", default="", type=str)
parser.add_argument("-r", "--restore_file", default="", type=str)
args = parser.parse_args()
arg_dict = args.__dict__
data = yaml.safe_load(open(args.config,'r'))
for key, value in data.items():
    if key in arg_dict:
        if arg_dict[key]:
            continue
    arg_dict[key] = value

trainer = Trainer(train_set=args.train_set, validation_set=args.validation_set, validate=args.doval, train_args=args)
train = getattr(trainer, args.train_mode)
train(epochs=args.epochs, args=args)


