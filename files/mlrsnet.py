import os
import pickle
import math
import random
from collections import defaultdict

from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from dassl.utils import read_json, write_json, mkdir_if_missing, listdir_nohidden


def read_split(filepath, path_prefix):
    def _convert(items):
        out = []
        for impath, label, classname in items:
            impath = os.path.join(path_prefix, impath)
            item = Datum(impath=impath, label=int(label), classname=classname)
            out.append(item)
        return out

    print(f"Reading split from {filepath}")
    split = read_json(filepath)
    train = _convert(split["train"])
    val = _convert(split["val"])
    test = _convert(split["test"])

    return train, val, test


def read_and_split_data(image_dir, p_trn=0.5, p_val=0.2, ignored=[], new_cnames=None):
    categories = listdir_nohidden(image_dir)
    categories = [c for c in categories if c not in ignored]
    categories.sort()

    p_tst = 1 - p_trn - p_val
    print(f"Splitting into {p_trn:.0%} train, {p_val:.0%} val, and {p_tst:.0%} test")



def save_split(train, val, test, filepath, path_prefix):
    def _extract(items):
        out = []
        for item in items:
            impath = item.impath
            label = item.label
            classname = item.classname
            impath = impath.replace(path_prefix, "")
            if impath.startswith("/"):
                impath = impath[1:]
            out.append((impath, label, classname))
        return out

    train = _extract(train)
    val = _extract(val)
    test = _extract(test)

    split = {"train": train, "val": val, "test": test}

    write_json(split, filepath)
    print(f"Saved split to {filepath}")



def subsample_classes(*args, subsample="all"):
    """Divide classes into two groups. The first group
    represents base classes while the second group represents
    new classes.

    Args:
        args: a list of datasets, e.g. train, val and test.
        subsample (str): what classes to subsample.
        """
    assert subsample in ["all", "base", "new"]

    if subsample == "all":
        return args
        
    dataset = args[0]
    labels = set()
    for item in dataset:
        labels.add(item.label)
    labels = list(labels)
    labels.sort()
    n = len(labels)
    # Divide classes into two halves
    m = math.ceil(n / 2)

    print(f"SUBSAMPLE {subsample.upper()} CLASSES!")
    if subsample == "base":
        selected = labels[:m]  # take the first half
    else:
        selected = labels[m:]  # take the second half
    relabeler = {y: y_new for y_new, y in enumerate(selected)}
        
    output = []
    for dataset in args:
        dataset_new = []
        for item in dataset:
            if item.label not in selected:
                continue
            item_new = Datum(
                impath=item.impath,
                label=relabeler[item.label],
                classname=item.classname
            )
            dataset_new.append(item_new)
        output.append(dataset_new)
        
    return output



@DATASET_REGISTRY.register()
class MLRSNet(DatasetBase):

    dataset_dir = "MLRSNet"

    def __init__(self, cfg):
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        print(root)
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, "images")
        self.split_path = os.path.join(
            self.dataset_dir, "mlrsnet.json")
        self.shots_dir = os.path.join(
            self.dataset_dir, "shots") 
        # mkdir_if_missing(self.shots_dir)

        if os.path.exists(self.split_path):
            train, val, test = read_split(
                self.split_path, self.image_dir)
        else:
            train, val, test = read_and_split_data(
                self.image_dir, ignored=None)
            save_split(
                train, val, test, self.split_path, self.image_dir)

        num_shots = cfg.DATASET.NUM_SHOTS
        if num_shots >= 1:
            seed = cfg.SEED
            preprocessed = os.path.join(
                self.shots_dir, f"shot_{num_shots}-seed_{seed}.pkl")

            if os.path.exists(preprocessed):
                print(
                    f"Loading preprocessed few-shot data from {preprocessed}")
                with open(preprocessed, "rb") as file:
                    data = pickle.load(file)
                    train, val = data["train"], data["val"]
            else:
                train = self.generate_fewshot_dataset(
                    train, num_shots=num_shots)
                val = self.generate_fewshot_dataset(
                    val, num_shots=min(num_shots, 4))
                data = {"train": train, "val": val}
                print(f"Saving preprocessed few-shot data to {preprocessed}")
                with open(preprocessed, "wb") as file:
                    pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)

        subsample = cfg.DATASET.SUBSAMPLE_CLASSES
        train, val, test = subsample_classes(
            train, val, test, subsample=subsample)

        super().__init__(train_x=train, val=val, test=test)
