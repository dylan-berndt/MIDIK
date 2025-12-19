import torch
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np
import json
import pandas as pd

import mido
from mido import second2tick
import itertools
import matplotlib.pyplot as plt

import kagglehub
import os
from glob import glob

from concurrent.futures import ThreadPoolExecutor, as_completed

os.environ["KAGGLEHUB_CACHE"] = "F:/.cache/kagglehub"


def discreteTokenizerFactory(config):
    minimum = config.minTimeResolution / 4
    ranges = config.ranges
    padding = config.padding

    labels = {
        "note_on": ["velocity", "note"],
        "note_off": ["velocity", "note"],
        "polytouch": ["value_1", "note"],
        "control_change": ["value_2", "control"],
        "program_change": ["program"],
        "aftertouch": ["value_3"],
        "pitchwheel": ["pitch"]
    }

    def tokenizeDiscrete(midiPath):
        channels = {key: [] for key in ranges.keys()}

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

            if messageType == "note_off" or (messageType == "note_on" and msg.velocity == 0):
                if (msg.channel, msg.note) in openNotes and openNotes[(msg.channel, msg.note)] != []:
                    index, start = openNotes[(msg.channel, msg.note)][-1]
                    duration = currentTime - start
                    channels["duration"][index] = round(duration / resolution)
                    del openNotes[(msg.channel, msg.note)][-1]
                    # print("huh found", msg.type)
                continue

            messageParams = labels[messageType]
            for param in messageParams:
                messageParam = param
                if "_" in param:
                    messageParam = param.split("_")[0]
                if hasattr(msg, messageParam):
                    value = getattr(msg, messageParam)
                    if messageParam != "pitch":
                        discretized = int(value)
                    else:
                        discretized = (value + 8192) // 8
                    channels[param].append(discretized)
                else:
                    print("huh")

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

            for channel in channels.keys():
                if len(channels[channel]) < len(channels["messageType"]):
                    channels[channel].append(ranges[channel])

        # Fix any overhung notes
        if len({key: value for key, value in openNotes.items() if value != []}) > 0:
            for key in openNotes:
                for note in openNotes[key]:
                    index, start = note
                    duration = currentTime - start
                    channels["duration"][index] = round(duration / resolution)

        for key in channels:
            for i in range(len(channels[key])):
                if channels[key][i] >= ranges[key]:
                    if padding[key] is None:
                        channels[key][i] = ranges[key] - 1
                    else:
                        channels[key][i] = padding[key]
                if channels[key][i] < 0:
                    channels[key][i] = 0

        context = {"length": [mid.length], "tokens": [len(channels["messageType"])]}

        return channels, context

    def tokenizeProtected(midiPath):
        try:
            return tokenizeDiscrete(midiPath)
        # Why do you gotta do me like that
        except (OSError, ValueError, EOFError, mido.KeySignatureError):
            return None

    return tokenizeProtected


def untokenizeFactory(config):
    minimum = config.minTimeResolution / 4
    ranges = config.ranges
    padding = config.padding

    labels = {
        "note_on": ["velocity", "note"],
        "note_off": ["velocity", "note"],
        "polytouch": ["value_1", "note"],
        "control_change": ["value_2", "control"],
        "program_change": ["program"],
        "aftertouch": ["value_3"],
        "pitchwheel": ["pitch"]
    }

    def detokenize(channels):
        openNotes = {}
        for i in range(len(channels["messageType"])):
            pass

    return detokenize


class LakhData(Dataset):
    def __init__(self, config, location="lakh", fixMissing=True):
        location = os.path.join("..", location) if os.getcwd().endswith("utils") else location
        self.config = config
        self.seqLength = self.config.sequenceLength

        self.tokenizer = discreteTokenizerFactory(config.tokenizer)

        path = kagglehub.dataset_download("imsparsh/lakh-midi-clean")
        self.midiPaths = glob(os.path.join(path, "**", "*.mid"))

        existing = [os.path.exists(os.path.join(location, os.path.basename(path).removesuffix(".mid") + ".json")) for path in self.midiPaths]

        missing = np.array(self.midiPaths)[~np.array(existing)]
        if not os.path.exists(location):
            os.mkdir(location)

        context = pd.DataFrame({"length": [], "tokens": []})
        try:
            oldContext = pd.read_csv(os.path.join(location, "context.csv"))
            context = pd.concat([oldContext, context])
        except Exception:
            pass

        if fixMissing:
            for p, path in enumerate(missing):
                try:
                    data, newContext = self.tokenizer(path)
                    if data is None:
                        continue
                    savePath = os.path.join(location, os.path.basename(path).removesuffix(".mid") + ".json")
                    if len(context) == 0:
                        context = pd.DataFrame(newContext)
                    else:
                        context = pd.concat([context, pd.DataFrame(newContext)], ignore_index=True)
                    with open(savePath, "w+") as file:
                        json.dump(data, file)
                except KeyboardInterrupt:
                    context.to_csv(os.path.join(location, "context.csv"))
                    break
                except TypeError:
                    continue

                print(f"\r{p + 1}/{len(missing)} MIDIs tokenized", end="")

        print()

        self.context = context
        context.to_csv(os.path.join(location, "context.csv"))

        self.jsonPaths = glob(os.path.join(location, "*.json"))
        self.tokens = []
        self.lengths = []

        for t, tokenPath in enumerate(self.jsonPaths):
            with open(tokenPath, "r") as file:
                try:
                    data = json.load(file)
                except json.JSONDecodeError:
                    continue
            if len(data["messageType"]) < self.config.sequenceLength:
                continue
            self.tokens.append(data)
            self.lengths.append(len(data["messageType"]))

        self.lengths = np.array(self.lengths)

        print(np.sum(self.lengths), "tokens in dataset")
        # print(f"{(np.array(ratios) / self.seqLength).mean():.2f} average seconds per {self.seqLength} token sequence")

    def __len__(self):
        return np.sum(self.lengths - self.seqLength)

    def __getitem__(self, item):
        song = 0
        offset = item
        while self.lengths[song] - self.seqLength < offset:
            offset -= self.lengths[song] - self.seqLength
            song += 1

        data = self.tokens[song]
        sampled = {}
        for key in data:
            channel = np.array(data[key])
            sampled[key] = channel[offset: offset + self.seqLength]

        return sampled

    @staticmethod
    def collate(samples):
        data = {}

        for key in samples[0]:
            lists = [sample[key] for sample in samples]
            tensor = torch.tensor(np.array(lists), dtype=torch.long)
            data[key] = tensor

        collated = {"sequences": data}

        return collated

    @staticmethod
    def split(dataset, seed=1234, split=0.8):
        np.random.seed(seed)
        songIndices = np.arange(dataset.lengths.shape[0])
        trainMask = np.random.uniform(size=[songIndices.shape[0]]) < split

        samplesPerSong = np.clip(dataset.lengths - dataset.config.sequenceLength, 0, np.inf)
        songSampleIndices = np.concat([[0], np.cumsum(samplesPerSong, axis=0)], axis=0).astype(int)

        trainSongs = songIndices[trainMask]
        testSongs = songIndices[~trainMask]

        index = np.arange(len(dataset))
        trainIndices = np.concat([index[songSampleIndices[song]:songSampleIndices[song + 1]]
                                  for song in trainSongs], axis=0)
        testIndices = np.concat([index[songSampleIndices[song]:songSampleIndices[song + 1]]
                                for song in testSongs], axis=0)

        # TODO: Test this
        train = torch.utils.data.Subset(dataset, trainIndices)
        test = torch.utils.data.Subset(dataset, testIndices)

        return train, test


class VGData(LakhData):
    def __init__(self, config, location="vg", fixMissing=True):
        super().__init__(config, location, fixMissing)

    def __getitem__(self, item):
        pass


if __name__ == "__main__":
    from config import *

    root = ".." if os.getcwd().endswith("utils") else ""
    dataset = LakhData(Config().load(os.path.join(root, "configs", "config.json")).dataset, fixMissing=False)
    loader = DataLoader(dataset, batch_size=128, collate_fn=LakhData.collate, shuffle=True)

    batch = next(iter(loader))

    plt.title("Message Types")
    plt.hist(batch["sequences"]["messageType"].flatten().cpu().numpy())
    plt.show()

    plt.title("Velocity")
    plt.hist(batch["sequences"]["velocity"][batch["sequences"]["messageType"] == 0].cpu().numpy())
    plt.show()

    plt.title("Time")
    time = (batch["sequences"]["time"][batch["sequences"]["messageType"] == 0]).cpu().numpy()
    plt.hist(time)
    plt.show()
    print(np.mean(time > 1024))
    print(np.unique_counts(time))

    plt.title("Duration")
    duration = (batch["sequences"]["duration"][batch["sequences"]["messageType"] == 0]).cpu().numpy()
    plt.hist(duration)
    plt.show()
    print(np.mean(duration > 1024))
    print(np.unique_counts(duration))

    print((batch["sequences"]["velocity"][batch["sequences"]["messageType"] == 0] == 0).cpu().numpy().sum())

    plt.title("Song Length Encoding (s)")
    plt.hist(dataset.config.sequenceLength / (dataset.context.tokens / dataset.context.length))
    plt.show()

    plt.title("Song Length (tokens)")
    plt.hist(dataset.context.length)
    plt.show()

    print((dataset.config.sequenceLength / (dataset.context.tokens / dataset.context.length)).mean())