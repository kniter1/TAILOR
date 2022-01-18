"""Provides functions to help with evaluating models."""
import numpy as np
import collections
import re
import glob
import os

import tensorflow as tf
from tensorflow import logging

import utils.metrics.mean_average_precision_calculator as map_calculator
import utils.metrics.average_precision_calculator as ap_calculator
from utils.metrics.pr_calculator import PRCalculator
from utils.metrics.pr_calculator_per_tag import PRCalculatorPerTag

###
###Training utils
###
def find_class_by_name(name, modules):
    """Searches the provided modules for the named class and returns it."""
    modules = [getattr(module, name, None) for module in modules]
    return next(a for a in modules if a)

def Dequantize(feat_vector, max_quantized_value=2, min_quantized_value=-2):
  """Dequantize the feature from the byte format to the float format.
  Args:
    feat_vector: the input 1-d vector.
    max_quantized_value: the maximum of the quantized value.
    min_quantized_value: the minimum of the quantized value.
  Returns:
    A float vector which has the same shape as feat_vector.
  """
  assert max_quantized_value > min_quantized_value
  quantized_range = max_quantized_value - min_quantized_value
  scalar = quantized_range / 255.0
  bias = (quantized_range / 512.0) + min_quantized_value
  return feat_vector * scalar + bias

def MakeSummary(name, value):
  """Creates a tf.Summary proto with the given name and value."""
  summary = tf.Summary()
  val = summary.value.add()
  val.tag = str(name)
  val.simple_value = float(value)
  return summary

#def FormatBatchInfo(global_step_val, global_step_info_dict):
  #this_hit_at_one = global_step_info_dict["hit_at_one"]
  #this_perr = global_step_info_dict["perr"]
  #this_loss = global_step_info_dict["loss"]
  #examples_per_second = global_step_info_dict.get("examples_per_second", -1)

  #info = ("global_step {0} | Batch Hit@1: {1:.3f} | Batch PERR: {2:.3f} | Batch Loss: {3:.3f} "
  #        "| Examples_per_sec: {4:.3f}").format(
  #            global_step_val, this_hit_at_one, this_perr, this_loss,
  #            examples_per_second)
  #return info

def FormatEvalInfo(summary_writer,
                    global_step_val,
                    epoch_info_dict,
                    prefix='eval_fusion',
                    AddSummary=True):
  """Add the epoch summary to the Tensorboard.
  Args:
    summary_writer: Tensorflow summary_writer.
    global_step_val: a int value of the global step.
    epoch_info_dict: a dictionary of the evaluation metrics calculated for the
      whole epoch.
    summary_scope: Train or Eval.
  Returns:
    A string of this global_step summary
  """
  epoch_id = epoch_info_dict["epoch_id"]
  avg_hit_at_one = epoch_info_dict["avg_hit_at_one"]
  avg_perr = epoch_info_dict["avg_perr"]
  avg_loss = epoch_info_dict["avg_loss"]
  aps = epoch_info_dict["aps"]
  gap = epoch_info_dict["gap"]
  precision_at_1 = epoch_info_dict["precision_at_1"]
  precision_at_5 = epoch_info_dict["precision_at_5"]
  recall_at_1 = epoch_info_dict['recall_at_1']
  recall_at_5 = epoch_info_dict['recall_at_5']

  mean_ap = np.mean(aps)

  if AddSummary:
    summary_writer.add_summary(
        MakeSummary(prefix + "/Avg_Hit@1", avg_hit_at_one), global_step_val)
    summary_writer.add_summary(
        MakeSummary(prefix + "/Avg_Perr", avg_perr),global_step_val)
    summary_writer.add_summary(
        MakeSummary(prefix + "/Avg_Loss", avg_loss),  global_step_val)
    summary_writer.add_summary(
        MakeSummary(prefix + "/MAP", mean_ap), global_step_val)
    summary_writer.add_summary(
        MakeSummary(prefix +"/GAP", gap), global_step_val)
    summary_writer.add_summary(
        MakeSummary(prefix +"/precision@0.1", precision_at_1), global_step_val)
    summary_writer.add_summary(
        MakeSummary(prefix +"/precision@0.5", precision_at_5), global_step_val)
    summary_writer.add_summary(
        MakeSummary(prefix +"/recall@0.1", recall_at_1), global_step_val)
    summary_writer.add_summary(
        MakeSummary(prefix +"/recall@0.5", recall_at_5), global_step_val)
    summary_writer.flush()

  info = "epoch/eval number {} | MAP: {:.3f} | GAP: {:.3f} | p@0.1: {:.3f} | p@0.5:{:.3f} | r@0.1:{:.3f} | r@0.5: {:.3f} | Avg_Loss: {:3f}".format(epoch_id, mean_ap, gap, precision_at_1, precision_at_5, recall_at_1, recall_at_5, avg_loss)
  return info

def GetListOfFeatureNamesAndSizes(feature_names, feature_sizes):
  """Extract the list of feature names and the dimensionality of each feature
     from string of comma separated values.
  Args:
    feature_names: string containing comma separated list of feature names
    feature_sizes: string containing comma separated list of feature sizes
  Returns:
    List of the feature names and list of the dimensionality of each feature.
    Elements in the first/second list are strings/integers.
  """
  list_of_feature_names = [
      feature_names.strip() for feature_names in feature_names.split(',')]
  list_of_feature_sizes = [
      int(feature_sizes) for feature_sizes in feature_sizes.split(',')]
  if len(list_of_feature_names) != len(list_of_feature_sizes):
    logging.error("length of the feature names (=" +
                  str(len(list_of_feature_names)) + ") != length of feature "
                  "sizes (=" + str(len(list_of_feature_sizes)) + ")")

  return list_of_feature_names, list_of_feature_sizes

def clip_gradient_norms(gradients_to_variables, max_norm):
  """Clips the gradients by the given value.
  Args:
    gradients_to_variables: A list of gradient to variable pairs (tuples).
    max_norm: the maximum norm value.
  Returns:
    A list of clipped gradient to variable pairs.
  """
  clipped_grads_and_vars = []
  for grad, var in gradients_to_variables:
    if grad is not None:
      if isinstance(grad, tf.IndexedSlices):
        tmp = tf.clip_by_norm(grad.values, max_norm)
        grad = tf.IndexedSlices(tmp, grad.indices, grad.dense_shape)
      else:
        grad = tf.clip_by_norm(grad, max_norm)
    clipped_grads_and_vars.append((grad, var))
  return clipped_grads_and_vars

def combine_gradients(tower_grads):
  """Calculate the combined gradient for each shared variable across all towers.
  Note that this function provides a synchronization point across all towers.
  Args:
    tower_grads: List of lists of (gradient, variable) tuples. The outer list
      is over individual gradients. The inner list is over the gradient
      calculation for each tower.
  Returns:
     List of pairs of (gradient, variable) where the gradient has been summed
     across all towers.
  """
  filtered_grads = [[x for x in grad_list if x[0] is not None] for grad_list in tower_grads]
  final_grads = []
  for i in range(len(filtered_grads[0])):
    grads = [filtered_grads[t][i] for t in range(len(filtered_grads))]
    grad = tf.stack([x[0] for x in grads], 0)
    grad = tf.reduce_mean(grad, 0)
    final_grads.append((grad, filtered_grads[0][i][1],))

  return final_grads

###
###Validate while training
###
def flatten(l):
  """ Merges a list of lists into a single list. """
  return [item for sublist in l for item in sublist]

def get_tag_stat(labels):
  """
  get freq num of each tag
  """
  num_classes = labels.shape[1]
  num_stat = np.zeros(num_classes)
  for i in range(num_classes):
    num_stat[i] = np.sum(labels[:,i])
  return num_stat

def get_tag_correlation(preds, labels, top_k=10):
    n_example, n_class = preds.shape
    tag_correlation = np.zeros((n_class, n_class))
    top_k = min(n_class, top_k)
    #convert pred to top_k index
    pred_indx = np.zeros((n_example, n_class), dtype=np.int8)
    for i in range(n_example):
      for idx in np.argpartition(preds[i], -top_k)[-top_k:]:
        pred_indx[i][idx] = 1
    #get correlation matrix
    for i in range(n_example):
      label_index = np.nonzero(labels[i])[0]
      pred_index = np.nonzero(pred_indx[i])[0]
      for li in label_index:
        for pi in pred_index:
          tag_correlation[li][pi] +=1
    return tag_correlation

def get_tag_confidence(predictions, labels):
  n_example, n_class = predictions.shape
  tag_confidence = np.zeros(n_class)
  for i in range(n_example):
    label_index = np.nonzero(labels[i])[0]
    for j in label_index:
      tag_confidence[j]+=predictions[i][j]
  return tag_confidence

def calculate_hit_at_one(predictions, actuals):
  """Performs a local (numpy) calculation of the hit at one.

  Args:
    predictions: Matrix containing the outputs of the model.
      Dimensions are 'batch' x 'num_classes'.
    actuals: Matrix containing the ground truth labels.
      Dimensions are 'batch' x 'num_classes'.

  Returns:
    float: The average hit at one across the entire batch.
  """
  top_prediction = np.argmax(predictions, 1)
  hits = actuals[np.arange(actuals.shape[0]), top_prediction]
  return np.average(hits)


def calculate_precision_at_equal_recall_rate(predictions, actuals):
  """Performs a local (numpy) calculation of the PERR.

  Args:
    predictions: Matrix containing the outputs of the model.
      Dimensions are 'batch' x 'num_classes'.
    actuals: Matrix containing the ground truth labels.
      Dimensions are 'batch' x 'num_classes'.

  Returns:
    float: The average precision at equal recall rate across the entire batch.
  """
  aggregated_precision = 0.0
  num_videos = actuals.shape[0]
  for row in np.arange(num_videos):
    num_labels = int(np.sum(actuals[row]))
    top_indices = np.argpartition(predictions[row],
                                     -num_labels)[-num_labels:]
    item_precision = 0.0
    for label_index in top_indices:
      if predictions[row][label_index] > 0:
        item_precision += actuals[row][label_index]
    item_precision /= top_indices.size
    aggregated_precision += item_precision
  aggregated_precision /= num_videos
  return aggregated_precision

def calculate_gap(predictions, actuals, top_k=20):
  """Performs a local (numpy) calculation of the global average precision.

  Only the top_k predictions are taken for each of the videos.

  Args:
    predictions: Matrix containing the outputs of the model.
      Dimensions are 'batch' x 'num_classes'.
    actuals: Matrix containing the ground truth labels.
      Dimensions are 'batch' x 'num_classes'.
    top_k: How many predictions to use per video.

  Returns:
    float: The global average precision.
  """
  gap_calculator = ap_calculator.AveragePrecisionCalculator()
  sparse_predictions, sparse_labels, num_positives = top_k_by_class(predictions, actuals, top_k)
  gap_calculator.accumulate(flatten(sparse_predictions), flatten(sparse_labels), sum(num_positives))
  return gap_calculator.peek_ap_at_n()


def top_k_by_class(predictions, labels, k=20):
  """Extracts the top k predictions for each video, sorted by class.

  Args:
    predictions: A numpy matrix containing the outputs of the model.
      Dimensions are 'batch' x 'num_classes'.
    k: the top k non-zero entries to preserve in each prediction.

  Returns:
    A tuple (predictions,labels, true_positives). 'predictions' and 'labels'
    are lists of lists of floats. 'true_positives' is a list of scalars. The
    length of the lists are equal to the number of classes. The entries in the
    predictions variable are probability predictions, and
    the corresponding entries in the labels variable are the ground truth for
    those predictions. The entries in 'true_positives' are the number of true
    positives for each class in the ground truth.

  Raises:
    ValueError: An error occurred when the k is not a positive integer.
  """
  if k <= 0:
    raise ValueError("k must be a positive integer.")
  k = min(k, predictions.shape[1])
  num_classes = predictions.shape[1]
  prediction_triplets= []
  for video_index in range(predictions.shape[0]):
    prediction_triplets.extend(top_k_triplets(predictions[video_index],labels[video_index], k))
  out_predictions = [[] for v in range(num_classes)]
  out_labels = [[] for v in range(num_classes)]
  for triplet in prediction_triplets:
    out_predictions[triplet[0]].append(triplet[1])
    out_labels[triplet[0]].append(triplet[2])
  out_true_positives = [np.sum(labels[:,i]) for i in range(num_classes)]

  return out_predictions, out_labels, out_true_positives

def top_k_triplets(predictions, labels, k=20):
  """Get the top_k for a 1-d numpy array. Returns a sparse list of tuples in
  (prediction, class) format"""
  m = len(predictions)
  k = min(k, m)
  indices = np.argpartition(predictions, -k)[-k:]
  return [(index, predictions[index], labels[index]) for index in indices]

class EvaluationMetrics(object):
  """A class to store the evaluation metrics."""

  def __init__(self, num_class, top_k, accumulate_per_tag=False):
    """Construct an EvaluationMetrics object to store the evaluation metrics.

    Args:
      num_class: A positive integer specifying the number of classes.
      top_k: A positive integer specifying how many predictions are considered per video.

    Raises:
      ValueError: An error occurred when MeanAveragePrecisionCalculator cannot
        not be constructed.
    """
    self.sum_hit_at_one = 0.0
    self.sum_perr = 0.0
    self.sum_loss = 0.0
    self.map_calculator = map_calculator.MeanAveragePrecisionCalculator(num_class)
    self.global_ap_calculator = ap_calculator.AveragePrecisionCalculator()
    self.pr_calculator = PRCalculator()
    self.pr_calculator_per_tag = PRCalculatorPerTag(num_class)
    self.accumulate_per_tag = accumulate_per_tag

    self.top_k = top_k
    self.num_examples = 0
    self.nums_per_tag = np.zeros(num_class)
    self.tag_corrlation = np.zeros((num_class, num_class))
    self.tag_confidence = np.zeros(num_class)

  def accumulate(self, predictions, labels, loss):
    """Accumulate the metrics calculated locally for this mini-batch.

    Args:
      predictions: A numpy matrix containing the outputs of the model.
        Dimensions are 'batch' x 'num_classes'.
      labels: A numpy matrix containing the ground truth labels.
        Dimensions are 'batch' x 'num_classes'.
      loss: A numpy array containing the loss for each sample.

    Returns:
      dictionary: A dictionary storing the metrics for the mini-batch.

    Raises:
      ValueError: An error occurred when the shape of predictions and actuals
        does not match.
    """
    batch_size = labels.shape[0]
    mean_hit_at_one = calculate_hit_at_one(predictions, labels)
    mean_perr = calculate_precision_at_equal_recall_rate(predictions, labels)
    mean_loss = loss
    self.nums_per_tag = self.nums_per_tag + get_tag_stat(labels)
    self.tag_correlation = self.tag_correlation + get_tag_correlation(predictions, labels, self.top_k)
    self.tag_confidence = self.tag_confidence + get_tag_confidence(predictions, labels)

    self.pr_calculator.accumulate(predictions, labels)
    if self.accumulate_per_tag:
        self.pr_calculator_per_tag.accumulate(predictions, labels)

    # Take the top 20 predictions.
    sparse_predictions, sparse_labels, num_positives = top_k_by_class(predictions, labels, self.top_k)
    self.map_calculator.accumulate(sparse_predictions, sparse_labels, num_positives)
    self.global_ap_calculator.accumulate(flatten(sparse_predictions), flatten(sparse_labels), sum(num_positives))

    self.num_examples += batch_size
    self.sum_hit_at_one += mean_hit_at_one * batch_size
    self.sum_perr += mean_perr * batch_size
    self.sum_loss += mean_loss * batch_size

    return {"hit_at_one": mean_hit_at_one, "perr": mean_perr, "loss": mean_loss}

  def get(self):
    """Calculate the evaluation metrics for the whole epoch.

    Raises:
      ValueError: If no examples were accumulated.

    Returns:
      dictionary: a dictionary storing the evaluation metrics for the epoch. The
        dictionary has the fields: avg_hit_at_one, avg_perr, avg_loss, and
        aps (default nan).
    """
    if self.num_examples <= 0:
      raise ValueError("total_sample must be positive.")
    avg_hit_at_one = self.sum_hit_at_one / self.num_examples
    avg_perr = self.sum_perr / self.num_examples
    avg_loss = self.sum_loss / self.num_examples

    aps = self.map_calculator.peek_map_at_n()
    gap = self.global_ap_calculator.peek_ap_at_n()
    tag_confidence = self.tag_confidence/(self.nums_per_tag+1e-10)

    precision_at_1 = self.pr_calculator.get_precision_at_conf(0.1)
    recall_at_1 = self.pr_calculator.get_recall_at_conf(0.1)
    precision_at_5 = self.pr_calculator.get_precision_at_conf(0.5)
    recall_at_5 = self.pr_calculator.get_recall_at_conf(0.5)

    tag_precision = self.pr_calculator_per_tag.get_precision_list(0.5) if self.accumulate_per_tag else []
    tag_recall = self.pr_calculator_per_tag.get_recall_list(0.5) if self.accumulate_per_tag else []

    epoch_info_dict= {"avg_hit_at_one": avg_hit_at_one, "avg_perr": avg_perr,
                      "avg_loss": avg_loss, "aps": aps, "gap": gap,
                      'num': self.nums_per_tag,
                      'tag_correlation': self.tag_correlation,
                      'tag_confidence': tag_confidence,
                      "precision_at_1": precision_at_1,
                      "recall_at_1": recall_at_1,
                      "precision_at_5": precision_at_5,
                      "recall_at_5": recall_at_5,
                      "tag_precision": tag_precision,
                      "tag_recall": tag_recall
                      }
    return epoch_info_dict

  def clear(self):
    """Clear the evaluation metrics and reset the EvaluationMetrics object."""
    self.sum_hit_at_one = 0.0
    self.sum_perr = 0.0
    self.sum_loss = 0.0
    self.map_calculator.clear()
    self.global_ap_calculator.clear()
    self.pr_calculator.clear()
    self.pr_calculator_per_tag.clear()
    self.num_examples = 0
    self.tag_correlation = 0.0
    self.nums_per_tag = 0.0
    self.tag_confidence = 0.0

# 匹配加载 pretrained model
def get_assignment_map_from_checkpoint(tvars, init_checkpoint,show=False, var_prefix='tower/text/'):
  """Compute the union of the current variables and checkpoint variables."""
  assignment_map = {}
  initialized_variable_names = {}

  name_to_variable = collections.OrderedDict()
  for var in tvars:
    name = var.name
    m = re.match("^(.*):\\d+$", name)
    if m is not None:
      name = m.group(1)
    #print(name)
    name_to_variable[name] = var


  init_vars = tf.train.list_variables(init_checkpoint)
  assignment_map = collections.OrderedDict()
  for x in init_vars:
    (name, var) = (x[0], x[1])
    if var_prefix+name not in name_to_variable:
      if show:
        print('not in variables: '+name)
      continue
    assignment_map[name] = var_prefix+name
    initialized_variable_names[name] = 1
    initialized_variable_names[name + ":0"] = 1
    print("assign: ",name, var_prefix+name)

  return (assignment_map, initialized_variable_names)

def get_latest_checkpoint(train_dir):
    index_files = glob.glob(os.path.join(train_dir,'model.ckpt-*.index'))
    if not index_files:
        return None
    # Index file path with the maximum step size.
    latest_index_file = sorted(
        [(int(os.path.basename(f).split("-")[-1].split(".")[0]), f)
         for f in index_files])[-1][1]
    # Chop off .index suffix and return
    return latest_index_file[:-6]

def get_label_name_dict(tag_id_file, tag_max_num=5):
    label_name_dict={}
    with open(tag_id_file, 'r') as lnf:
        for line in lnf:
            tag, idx = line.strip().split('\t')
            if int(idx) not in label_name_dict:
              label_name_dict[int(idx)] = [tag]
            else:
              label_name_dict[int(idx)].append(tag)
        for key in label_name_dict:
          label_name_dict[key] = '-'.join(label_name_dict[key][:tag_max_num])
    return label_name_dict

def get_tag_id_dict(tag_id_file):
    tag_id_dict={}
    with open(tag_id_file, 'r') as lnf:
        for line in lnf:
            tag, idx = line.strip().split('\t')
            tag_id_dict[tag] = int(idx)
    return tag_id_dict

def task_as_string(task):
    return "/job:%s/task:%s" % (task.type, task.index)
class ParameterServer(object):
    def __init__(self, cluster, task):
        self.cluster = cluster
        self.task = task
    def run(self):
        logging.info("%s: Starting parameter server within cluster %s.",
                     task_as_string(self.task), self.cluster.as_dict())
        server = start_server(self.cluster, self.task)
        server.join()
def start_server(cluster, task):
    if not task.type:
        raise ValueError("%s: The task type must be specified." %task_as_string(task))
    if task.index is None:
        raise ValueError("%s: The task index must be specified." %task_as_string(task))
    return tf.train.Server(tf.train.ClusterSpec(cluster),protocol="grpc", job_name=task.type, task_index=task.index)

#inference utils
def format_lines(video_ids, predictions, top_k, label_name_dict):
  batch_size = len(video_ids)
  for video_index in range(batch_size):
    top_indices = np.argpartition(predictions[video_index], -top_k)[-top_k:]
    line = [(class_index, predictions[video_index][class_index])
            for class_index in top_indices]
    line = sorted(line, key=lambda p: -p[1])
    yield video_ids[video_index] + "\t" + "\t".join(
        "%s##%.3f" % (label_name_dict.get(int(label), 'NULL'), score) for (label, score) in line) + "\n"
