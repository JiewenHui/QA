import sys
import os
import subprocess
import time

batch_size = [32,60,64]
dropout_keep_probs=[0.4,0.5]
learning_rate = [0.01,0.001,0.0001]
num_filters = [80,100]
filter_size = [40,43,45]
num_epoch = 10
count = 0
model = 'QA_quantum'
for batch in batch_size:
	for dropout_keep_prob in dropout_keep_probs:
		for rate in learning_rate:
			for num_filter in num_filters:
				for fi in filter_size:
					print ('The ', count, 'excue\n')
					count += 1
					subprocess.call('python train.py --batch_size %d --dropout_keep_prob %f --learning_rate %f --num_epochs %d --num_filters %d --filter_sizes %d --model %s' % (batch,dropout_keep_prob,rate,num_epoch,num_filter,fi,model), shell = True)
