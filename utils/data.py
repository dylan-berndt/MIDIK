import torch
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np

from config import *

import mido
from mido import second2tick
import itertools
import matplotlib.pyplot as plt

import kagglehub
import os
from glob import glob

os.environ["KAGGLEHUB_CACHE"] = "F:/.cache/kagglehub"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_default_device(DEVICE)

ratios = []


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
        channels = {"messageType": [], "channel": [], "param0": [], "param1": [], "time": [], "duration": []}

        mid = mido.MidiFile(midiPath, clip=True)
        ppq = mid.ticks_per_beat
        resolution = ppq / minimum

        openNotes = {}
        currentTime = 0
        accumulatedTime = 0
        tempo = 500000

        for msg in mid:
            ticks = second2tick(msg.time, mid.ticks_per_beat, tempo)
            currentTime += ticks
            messageType = msg.type
            if messageType not in labels:
                if msg.time != 0:
                    accumulatedTime += ticks
                # print(msg)
                continue

            if msg.type == 'set_tempo':
                tempo = msg.tempo

            messageParams = labels[messageType]
            params = []
            for param in messageParams:
                if hasattr(msg, param):
                    value = getattr(msg, param)
                    discretized = discretization[param](value)
                    params.append(discretized)
                else:
                    print("huh")

            if messageType == "note_off" or (messageType == "note_on" and msg.velocity == 0):
                if (msg.channel, msg.note) in openNotes and openNotes[(msg.channel, msg.note)] != []:
                    index, start = openNotes[(msg.channel, msg.note)][-1]
                    duration = currentTime - start
                    channels["duration"][index] = round(duration / resolution)
                    del openNotes[(msg.channel, msg.note)][-1]
                    # print("huh found", msg.type)
                continue

            channels["messageType"].append(list(labels.keys()).index(messageType))
            channels["channel"].append(msg.channel)
            channels["time"].append(round((ticks + accumulatedTime) / resolution))
            channels["duration"].append(0)
            accumulatedTime = 0

            if messageType == "note_on" and msg.velocity != 0:
                note = (len(channels["messageType"]) - 1, currentTime)
                if (msg.channel, msg.note) in openNotes:
                    openNotes[(msg.channel, msg.note)].append(note)
                else:
                    openNotes[(msg.channel, msg.note)] = [note]

            for i in range(2):
                if len(params) > i:
                    channels[f"param{i}"].append(params[i])
                else:
                    channels[f"param{i}"].append(maximums[i])

        # Fix any overhung notes
        if len({key: value for key, value in openNotes.items() if value != []}) > 0:
            for key in openNotes:
                for note in openNotes[key]:
                    index, start = note
                    duration = currentTime - start
                    channels["duration"][index] = round(duration / resolution)

        ratios.append(len(channels["messageType"]) / mid.length)

        return channels

    return tokenizeDiscrete


class LakhData(Dataset):
    def __init__(self, config, location="lakh"):
        self.config = config

        self.tokenizer = discreteTokenizerFactory(config.tokenizer)

        path = kagglehub.dataset_download("imsparsh/lakh-midi-clean")
        self.midiPaths = glob(os.path.join(path, "**", "*.mid"))

        existing = [os.path.exists(os.path.join(location, os.path.basename(path).removesuffix(".mid") + ".json")) for path in self.midiPaths]

        missing = np.array(self.midiPaths)[np.array(existing)]
        for f, filePath in enumerate(missing):
            data = self.tokenizer(filePath)
            savePath = os.path.join(location, os.path.basename(filePath).removesuffix(".mid") + ".json")
            with open(savePath, "w") as file:
                json.dump(file, data)
            print(f"{f + 1}/{len(missing)} MIDIs tokenized")

        self.jsonPaths = glob(os.path.join(location, "*.json"))
        self.tokens = []
        self.lengths = []

        for t, tokenPath in enumerate(self.jsonPaths):
            with open(tokenPath, "r") as file:
                data = json.load(file)
            if len(data["messageType"]) < self.config.sequenceLength:
                continue
            self.tokens.append(data)
            self.lengths.append(len(data["messageType"]))

        self.lengths = np.array(self.lengths)

    def __len__(self):
        return np.sum(self.lengths - self.config.sequenceLength)

    def __getitem__(self, item):
        song = 0
        offset = item
        while self.lengths[song] - self.config.sequenceLength < offset:
            offset -= self.lengths[song] - self.config.sequenceLength
            song += 1

        data = self.tokens[song]
        if self.config.sequenceLength != None:
            for key in data:
                channel = np.array(data[key])
                data[key] = channel[offset: offset + self.config.sequenceLength]

        return data

    @staticmethod
    def collate(samples):
        data = {}

        # TODO: Change collate for fixed length sequences
        for key in samples[0]:
            # Gross
            lists = [sample[key].tolist() for sample in samples]
            padded = list(zip(*itertools.zip_longest(*lists, fillvalue=0)))
            tensor = torch.tensor(np.array(padded, dtype=np.uint16), dtype=torch.long)
            data[key] = tensor

        maxLength = max([len(sample["messageType"]) for sample in samples])
        mask = torch.zeros_like(data["messageType"], dtype=torch.bool)
        for s in range(len(samples)):
            length = len(samples[s]["messageType"])
            if maxLength != length:
                mask[s, -(maxLength - length):] = 1

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
    root = ".." if os.getcwd().endswith("utils") else ""
    dataset = LakhData(Config().load(os.path.join(root, "configs", "config.json")).dataset)
    batch = LakhData.collate([dataset[i] for i in range(128)])
    lengths = (batch["mask"].shape[1] - torch.sum(batch["mask"], dim=1)).cpu().numpy()

    plt.title("Song Lengths (Tokens)")
    plt.hist(lengths)
    plt.show()

    plt.title("Message Types")
    plt.hist(batch["sequences"]["messageType"].flatten().cpu().numpy())
    plt.show()

    plt.hist(batch["sequences"]["param0"][batch["sequences"]["messageType"] == 0 & batch["mask"]].cpu().numpy())
    plt.show()

    time = (batch["sequences"]["time"][batch["sequences"]["messageType"] == 0 & batch["mask"]]).cpu().numpy()
    plt.hist(time)
    plt.show()
    print(np.mean(time > 1024))

    duration = (batch["sequences"]["duration"][batch["sequences"]["messageType"] == 0 & batch["mask"]]).cpu().numpy()
    plt.hist(duration)
    plt.show()
    print(np.mean(duration > 1024))

    print((batch["sequences"]["param0"][batch["sequences"]["messageType"] == 0 & batch["mask"]] == 0).cpu().numpy().sum())

    plt.hist(dataset.config.truncate / np.array(ratios))
    plt.show()

    print((dataset.config.truncate / np.array(ratios)).mean())