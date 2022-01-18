#encoding: utf-8
#Author: jefxiong@tencent.com
import numpy as np

class PRCalculator():
  def __init__(self):
      # use only two threshold to save eval time
      self.threshold_dict={0.5:0, 0.1:1} #TODO(jefxiong, range from 0.9~0.01)
      self.precision = np.zeros((len(self.threshold_dict)))
      self.recall = np.zeros((len(self.threshold_dict)))
      self.accumulate_count = np.zeros((len(self.threshold_dict)))

  def accumulate(self, predictions, actuals):
      """
      predictions: n_example X n_classes
      actuals: n_example X n_classes
      """
      #assert isinstance(predictions, np.ndarray)
      #assert isinstance(actuals, np.ndarray)
      n_example = predictions.shape[0]

      precision_all = np.zeros((n_example, len(self.threshold_dict)))
      recall_all = np.zeros((n_example, len(self.threshold_dict)))
      for i in range(n_example):
        gt_index = np.nonzero(actuals[i])[0]
        for th, th_index in self.threshold_dict.items():
          pred_index = np.nonzero(predictions[i]>th)[0]
          tp = np.sum([actuals[i][k] for k in pred_index])
          precision_all[i][th_index]  = tp*1.0/len(pred_index) if len(pred_index)>0 else np.nan
          recall_all[i][th_index]  = tp*1.0/len(gt_index) if len(gt_index)>0 else np.nan


      valid_accumlate = (np.sum(~np.isnan(precision_all), axis=0)) != 0
      self.accumulate_count = self.accumulate_count + valid_accumlate

      precision_all = np.nansum(precision_all,axis=0)/(np.sum(~np.isnan(precision_all), axis=0)+1e-10)
      recall_all = np.nansum(recall_all,axis=0)/(np.sum(~np.isnan(recall_all), axis=0)+1e-10)

      self.precision = precision_all + self.precision
      self.recall = recall_all + self.recall

  def get_precision_at_conf(self, th=0.5):
      index = self.threshold_dict[th]
      return self.precision[index]/(1e-10+self.accumulate_count[index])

  def get_recall_at_conf(self, th=0.5):
      index = self.threshold_dict[th]
      return self.recall[index]/(1e-10+self.accumulate_count[index])

  def clear(self):
      self.accumulate_count = np.zeros((len(self.threshold_dict)))
      self.precision = np.zeros((len(self.threshold_dict)))
      self.recall = np.zeros((len(self.threshold_dict)))
