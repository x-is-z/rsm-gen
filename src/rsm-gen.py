#!/usr/bin/env python3

import keras
import wave
import numpy
import math

def importFile(path):
  with wave.open(path, 'rb') as f:
    nchannels = f.getnchannels()
    nframes = f.getnframes()
    clip = f.readframes(nframes)
    rate = f.getframerate()

  clip = numpy.frombuffer(clip, dtype=numpy.int16)
  clip = clip.astype(numpy.float32)
  clip = clip.reshape((nchannels, -1))

  mono = clip[0]

  for i in range(1, len(clip)):
    mono += channel

  return mono, rate

def exportFile(path, clip):
  with wave.open(path, 'wb') as f:
    f.setnchannels(1)
    f.setframerate(44100)
    f.setnframes(len(clip))
    f.writeframes(clip.astype(numpy.int16).tobytes())

def compress(clip, rate):
  notes = []

  for i in range(0, len(clip) - int(rate / 20), int(rate / 10)):
    transform = abs(numpy.fft.fft(clip[i : i + (rate / 10)]))

    for i in range(88):
      a4 = 440 * (rate / 44100)
      freq = (a4 / 16) * (2 ** (i / 12))

      notes.append(transform[int(note / 10)])

  notes = numpy.array(notes)
  notes *= 255 / max(notes)
  
  return notes.astype(numpy.uint8).astype(numpy.float32)

def extract(notes):
  ret = numpy.array([])

  cache = []

  for i in range(88):
    a4 = 440 * (rate / 44100)
    freq = (a4 / 16) * (2 ** (i / 12))
    
    cache.append(numpy.array([math.sin((t * 2 * math.pi * freq) / 44100) for t in range(4410)]))

  for i in range(0, len(notes), 88):
    chunk = numpy.zeros(4410)

    for j in range(88):
        chunk += cache[j] * (notes[i + j] / 2205)

    ret = numpy.append(ret, chunk)

  return ret * (1 / max(ret))

def createModel():
  model = keras.models.Sequential([
    Dense(100, activation='relu', input_dim=(880 * 5)),
    Dense(100, activation='relu'),
    Dense(100, activation='relu'),
    Dense(880 * 5, activation='relu'),
  ])

  model.compile(loss='mean_squared_error', optimizer='adam')
  return model

def getDataset(paths, numEntries):
  xtrain = []
  ytrain = []

  audioClips = {}

  for i in range(numEntries):
    path = random.choice(paths)

    if path in audioClips:
      audio = audioClips[path]
    else:
      audio, sr = importFile(path)
      audio = compress(audio, sr)
      audioClips[path] = audio

    index = int(random.random() * (len(audio) - (880 * 10)))
    xtrain.append(audio[index : index + (880 * 5)])
    ytrain.append(audio[index + (880 * 5) : index + (880 * 10)])

  xtrain = numpy.array(xtrain)
  ytrain = numpy.array(ytrain)

  return xtrain, ytrain

def fitModel(model, xtrain, ytrain, epochs=10, batchSize=32):
  model.fit(xtrain, ytrain, epochs=epochs, batch_size=batchSize)

def getRandomSeed(path):
  audio, sr = importFile(path)
  audio = compress(audio, sr)

  index = int(random.random() * (len(audio) - (880 * 5)))
  
  return audio[index : index + (880 * 5)]

def predict(model, seed, numContinuations):
  audio = numpy.array([])
  
  for i in range(numContinuations):
    audio = numpy.append(audio, extract(seed))
    seed = model.predict(numpy.array([seed]))[0]

  return audio

def main():
  pass

if __name__ == '__main__':
  main()
  
