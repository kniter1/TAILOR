#encoding: utf-8
#Author: jefxiong@tencent.com

from utils.metrics.pr_calculator import PRCalculator
import numpy as np
import time

def count_func_time(func):
    def call_fun(*args, **kwargs):
        start_time = time.time()
        func(*args, **kwargs)
        end_time = time.time()
        print('{} cost {:.3f} sec'.format(func.__name__, end_time-start_time))
    return call_fun

def map_func(obj, x1, x2):
  obj.accumulate(x1, x2)

class PRCalculatorPerTag():
  def __init__(self, tag_num):
    self.tag_num = tag_num
    self.pr_calculators = []
    for i in range(self.tag_num):
      self.pr_calculators.append(PRCalculator())

  #@count_func_time
  def accumulate(self, predictions, actuals):
    """
    predictions: n_example X n_classes
    actuals: n_example X n_classes
    """
    #n_example X n_classes ==> n_classes * [n_example x 1]
    pred_per_tag_list = np.expand_dims(predictions.transpose(), -1)
    actuals_per_tag_list = np.expand_dims(actuals.transpose(), -1)

    for i in range(self.tag_num):
      self.pr_calculators[i].accumulate(pred_per_tag_list[i], actuals_per_tag_list[i])
    #ret = list(map(map_func, self.pr_calculators, pred_per_tag_list, actuals_per_tag_list))

  def get_precision_list(self, th=0.5):
    return [self.pr_calculators[i].get_precision_at_conf(th) for i in range(self.tag_num)]

  def get_recall_list(self, th=0.5):
    return [self.pr_calculators[i].get_recall_at_conf(th) for i in range(self.tag_num)]

  def clear(self):
    for i in range(self.tag_num):
      self.pr_calculators[i].clear()
