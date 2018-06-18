from painNetwork import PainNet
import sys
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

from painInputGen import InputGenerator

input_gen = InputGenerator()

network = PainNet()

for step in range(1000):
  images,labels = input_gen.get_next_batch(4)
  print(network.run(images,labels))  
