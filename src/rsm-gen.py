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

def main():
  pass

if __name__ == '__main__':
  main()
