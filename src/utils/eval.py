import codecs
from sklearn import metrics
import numpy as np
import os

def dense(y):
	label_y = []
	for i in range(len(y)):
		for j in range(len(y[i])):
			label_y.append(y[i][j])

	return label_y
def get_accuracy(y, y_pre):
#	print('metric_acc:  ' + str(round(metrics.accuracy_score(y, y_pre),4)))
	sambles = len(y)
	count = 0.0
	for i in range(sambles):
		y_true = 0
		all_y = 0
		for j in range(len(y[i])):
			if y[i][j] > 0 and y_pre[i][j] > 0:
				y_true += 1
			if y[i][j] > 0 or y_pre[i][j] > 0:
				all_y += 1
		if all_y <= 0:
			all_y = 1

		count += float(y_true) / float(all_y)
	acc = float(count) / float(sambles)
	acc=round(acc,4)
	return acc
#	print('accuracy_hand:' + str(acc))


def get_metrics(y, y_pre):
	"""

	:param y:1071*6
	:param y_pre: 1071*6
	:return:
	"""
	y = y.cpu().detach().numpy()
	y_pre = y_pre.cpu().detach().numpy()
	test_labels = dense(y)
	test_pred = dense(y_pre)
#	print(metrics.classification_report(test_labels, test_pred, digits=4))
	# print(metrics.classification_report(test_labels, test_pred, digits=4))
	# print(metrics.precision_recall_fscore_support(test_labels, test_pred, average='macro'))
	# print("Micro average Test Precision, Recall and F1-Score...")
	# print(metrics.precision_recall_fscore_support(test_labels,test_pred, average='micro'))
	y=np.array(y)
	y_pre=np.array(y_pre)
#	print("hammloss: "+str(round(hamming_loss,4)))
	macro_f1 = metrics.f1_score(y, y_pre, average='macro')
	macro_precision = metrics.precision_score(y, y_pre, average='macro')
	macro_recall = metrics.recall_score(y, y_pre, average='macro')
	acc = get_accuracy(y, y_pre)
	y = np.array(y)
	y_pre = np.array(y_pre)

#	print(metrics.classification_report(y, y_pre, digits=4))
	# print("micro_precision, micro_precison,micro_recall")
	micro_f1 = metrics.f1_score(y, y_pre, average='micro')
	micro_precision = metrics.precision_score(y, y_pre, average='micro')
	micro_recall = metrics.recall_score(y, y_pre, average='micro')
	# print(""+str(round(micro_precision,4))+"\t"+str(round(micro_recall,4))+"\t"+str(round(micro_f1,4)))
	return micro_f1, micro_precision, micro_recall, acc



