import torch
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np

from config import *

import mido
import itertools
import matplotlib.pyplot as plt

import kagglehub
import os
from glob import glob

os.environ["KAGGLEHUB_CACHE"] = "F:/.cache/kagglehub"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_default_device(DEVICE)


def discreteTokenizerFactory(config):
    minimum = config.minTimeResolution / 4

    labels = {
        "note_on": ["velocity", "note"],
        "note_off": ["velocity", "note"],
        "polytouch": ["value", "note"],
        "control_change": ["value", "control"],
        "program_change": ["program"],
        "aftertouch": ["value"],
        "pitchwheel": ["pitch"]
    }

    discretization = {
        "channel": lambda x: int(x),

        "velocity": lambda x: int(x),
        "value": lambda x: int(x + 128),
        "program": lambda x: int(x + 256),
        "pitch": lambda x: int(x + 8192 + 384),

        "note": lambda x: int(x),
        "control": lambda x: int(x + 128)
    }

    maximums = {0: 16768, 1: 256}

    def tokenizeDiscrete(midiPath):
        channels = {"messageType": [], "channel": [], "param0": [], "param1": [], "time": []}

        mid = mido.MidiFile(midiPath, clip=True)
        ppq = mid.ticks_per_beat
        resolution = ppq / minimum

        for msg in mid:
            messageType = msg.type
            if messageType not in labels:
                # print(msg)
                continue

            messageParams = labels[messageType]
            params = []
            for param in messageParams:
                if hasattr(msg, param):
                    value = getattr(msg, param)
                    discretized = discretization[param](value)
                    params.append(discretized)
                else:
                    print("huh")

            channels["messageType"].append(list(labels.keys()).index(messageType))
            channels["channel"].append(msg.channel)
            channels["time"].append(round(msg.time / resolution))

            for i in range(2):
                if len(params) > i:
                    channels[f"param{i}"].append(params[i])
                else:
                    channels[f"param{i}"].append(maximums[i])

        return channels

    return tokenizeDiscrete


class LakhData(Dataset):
    truncate = None

    def __init__(self, config):
        self.config = config
        LakhData.truncate = self.config.truncate

        path = kagglehub.dataset_download("imsparsh/lakh-midi-clean")
        self.filePaths = glob(os.path.join(path, "**", "*.mid"))

        self.tokenizer = discreteTokenizerFactory(config.tokenizer)

    def __len__(self):
        return len(self.filePaths)

    def __getitem__(self, item):
        return self.tokenizer(self.filePaths[item])

    @staticmethod
    def collate(samples):
        data = {}

        for key in samples[0]:
            lists = [sample[key] for sample in samples]
            padded = list(zip(*itertools.zip_longest(*lists, fillvalue=0)))
            tensor = torch.tensor(np.array(padded, dtype=np.uint16), dtype=torch.long)
            tensor = tensor[:, :LakhData.truncate]
            data[key] = tensor

        maxLength = max([len(sample["messageType"]) for sample in samples])
        mask = torch.zeros_like(data["messageType"], dtype=torch.bool)
        for s in range(len(samples)):
            length = len(samples[s]["messageType"])
            if maxLength != length:
                mask[s, -(maxLength - length):] = 1

        mask = mask[:, :LakhData.truncate]

        collated = {"sequences": data, "mask": mask}

        return collated


class VGData(Dataset):
    def __init__(self):
        pass

    def __len__(self):
        pass

    def __getitem__(self, item):
        pass


if __name__ == "__main__":
    dataset = LakhData(Config().load(os.path.join("..", "configs", "config.json")).dataset)
    batch = LakhData.collate([dataset[i] for i in range(128)])
    lengths = (batch["mask"].shape[1] - torch.sum(batch["mask"], dim=1)).cpu().numpy()

    plt.hist(lengths)
    plt.show()

    plt.hist(batch["sequences"]["messageType"].flatten().cpu().numpy())
    plt.show()
