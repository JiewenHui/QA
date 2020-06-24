import sys
import os
import subprocess
import time
batch_size = [60,70,80,90]
num_filters=[55,60,65,70]
l2_reg_lambda=[0.01,0.001,0.0001,0.00001]
learning_rate = [0.01,0.001,0.0001,0.00001]
count = 0
for batch in batch_size:
	for num in num_filters:
		for l2 in l2_reg_lambda:
			for rate in learning_rate:
				print 'The ', count, 'excue\n'
				count += 1
				subprocess.call('python train.py --batch_size %d --num_filters %d --l2_reg_lambda %f --learning_rate %f' % (batch,num,l2,rate), shell = True)
