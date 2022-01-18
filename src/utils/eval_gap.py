import sys,os
import argparse
import numpy as np
import json
import heapq
import random
import numbers

# utils
def flatten(l):
  """ Merges a list of lists into a single list. """
  return [item for sublist in l for item in sublist]

class AveragePrecisionCalculator(object):
  """Calculate the average precision and average precision at n."""

  def __init__(self, top_n=None):
    """Construct an AveragePrecisionCalculator to calculate average precision.

    This class is used to calculate the average precision for a single label.

    Args:
      top_n: A positive Integer specifying the average precision at n, or
        None to use all provided data points.

    Raises:
      ValueError: An error occurred when the top_n is not a positive integer.
    """
    if not ((isinstance(top_n, int) and top_n >= 0) or top_n is None):
      raise ValueError("top_n must be a positive integer or None.")

    self._top_n = top_n  # average precision at n
    self._total_positives = 0  # total number of positives have seen
    self._heap = []  # max heap of (prediction, actual)

  @property
  def heap_size(self):
    """Gets the heap size maintained in the class."""
    return len(self._heap)

  @property
  def num_accumulated_positives(self):
    """Gets the number of positive samples that have been accumulated."""
    return self._total_positives

  def accumulate(self, predictions, actuals, num_positives=None):
    """Accumulate the predictions and their ground truth labels.

    After the function call, we may call peek_ap_at_n to actually calculate
    the average precision.
    Note predictions and actuals must have the same shape.

    Args:
      predictions: a list storing the prediction scores.
      actuals: a list storing the ground truth labels. Any value
      larger than 0 will be treated as positives, otherwise as negatives.
      num_positives = If the 'predictions' and 'actuals' inputs aren't complete,
      then it's possible some true positives were missed in them. In that case,
      you can provide 'num_positives' in order to accurately track recall.

    Raises:
      ValueError: An error occurred when the format of the input is not the
      numpy 1-D array or the shape of predictions and actuals does not match.
    """
    if len(predictions) != len(actuals):
      raise ValueError("the shape of predictions and actuals does not match.")

    if not num_positives is None:
      if not isinstance(num_positives, numbers.Number) or num_positives < 0:
        raise ValueError("'num_positives' was provided but it wan't a nonzero number.")

    if not num_positives is None:
      self._total_positives += num_positives
    else:
      self._total_positives += np.size(np.where(actuals > 0))
    topk = self._top_n
    heap = self._heap

    for i in range(np.size(predictions)):
      if topk is None or len(heap) < topk:
        heapq.heappush(heap, (predictions[i], actuals[i]))
      else:
        if predictions[i] > heap[0][0]:  # heap[0] is the smallest
          heapq.heappop(heap)
          heapq.heappush(heap, (predictions[i], actuals[i]))

  def clear(self):
    """Clear the accumulated predictions."""
    self._heap = []
    self._total_positives = 0

  def peek_ap_at_n(self):
    """Peek the non-interpolated average precision at n.

    Returns:
      The non-interpolated average precision at n (default 0).
      If n is larger than the length of the ranked list,
      the average precision will be returned.
    """
    if self.heap_size <= 0:
      return 0
    predlists = np.array(list(zip(*self._heap)))

    ap = self.ap_at_n(predlists[0],
                      predlists[1],
                      n=self._top_n,
                      total_num_positives=self._total_positives)
    return ap

  @staticmethod
  def ap(predictions, actuals):
    """Calculate the non-interpolated average precision.

    Args:
      predictions: a numpy 1-D array storing the sparse prediction scores.
      actuals: a numpy 1-D array storing the ground truth labels. Any value
      larger than 0 will be treated as positives, otherwise as negatives.

    Returns:
      The non-interpolated average precision at n.
      If n is larger than the length of the ranked list,
      the average precision will be returned.

    Raises:
      ValueError: An error occurred when the format of the input is not the
      numpy 1-D array or the shape of predictions and actuals does not match.
    """
    return AveragePrecisionCalculator.ap_at_n(predictions,
                                              actuals,
                                              n=None)

  @staticmethod
  def ap_at_n(predictions, actuals, n=20, total_num_positives=None):
    """Calculate the non-interpolated average precision.

    Args:
      predictions: a numpy 1-D array storing the sparse prediction scores.
      actuals: a numpy 1-D array storing the ground truth labels. Any value
      larger than 0 will be treated as positives, otherwise as negatives.
      n: the top n items to be considered in ap@n.
      total_num_positives : (optionally) you can specify the number of total
        positive
      in the list. If specified, it will be used in calculation.

    Returns:
      The non-interpolated average precision at n.
      If n is larger than the length of the ranked list,
      the average precision will be returned.

    Raises:
      ValueError: An error occurred when
      1) the format of the input is not the numpy 1-D array;
      2) the shape of predictions and actuals does not match;
      3) the input n is not a positive integer.
    """
    if len(predictions) != len(actuals):
      raise ValueError("the shape of predictions and actuals does not match.")

    if n is not None:
      if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be 'None' or a positive integer."
                         " It was '%s'." % n)

    ap = 0.0

    predictions = np.array(predictions)
    actuals = np.array(actuals)

    # add a shuffler to avoid overestimating the ap
    predictions, actuals = AveragePrecisionCalculator._shuffle(predictions,
                                                               actuals)
    sortidx = sorted(
        range(len(predictions)),
        key=lambda k: predictions[k],
        reverse=True)

    if total_num_positives is None:
      numpos = np.size(np.where(actuals > 0))
    else:
      numpos = total_num_positives

    if numpos == 0:
      return 0

    if n is not None:
      numpos = min(numpos, n)
    delta_recall = 1.0 / numpos
    poscount = 0.0

    # calculate the ap
    r = len(sortidx)
    if n is not None:
      r = min(r, n)
    for i in range(r):
      if actuals[sortidx[i]] > 0:
        poscount += 1
        ap += poscount / (i + 1) * delta_recall
    return ap

  @staticmethod
  def _shuffle(predictions, actuals):
    random.seed(0)
    suffidx = random.sample(range(len(predictions)), len(predictions))
    predictions = predictions[suffidx]
    actuals = actuals[suffidx]
    return predictions, actuals

  @staticmethod
  def _zero_one_normalize(predictions, epsilon=1e-7):
    """Normalize the predictions to the range between 0.0 and 1.0.

    For some predictions like SVM predictions, we need to normalize them before
    calculate the interpolated average precision. The normalization will not
    change the rank in the original list and thus won't change the average
    precision.

    Args:
      predictions: a numpy 1-D array storing the sparse prediction scores.
      epsilon: a small constant to avoid denominator being zero.

    Returns:
      The normalized prediction.
    """
    denominator = np.max(predictions) - np.min(predictions)
    ret = (predictions - np.min(predictions)) / np.max(denominator,
                                                             epsilon)
    return ret

def calculate_gap(predictions, actuals, top_k=6):
  gap_calculator = AveragePrecisionCalculator()
  sparse_predictions, sparse_labels, num_positives = top_k_by_class(predictions, actuals, top_k)
  gap_calculator.accumulate(flatten(sparse_predictions), flatten(sparse_labels), sum(num_positives))
  return gap_calculator.peek_ap_at_n()


def top_k_by_class(predictions, labels, k=20):
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


def get_tag_id_dict(tag_id_file):
    tag_id_dict={}
    with open(tag_id_file, 'r') as lnf:
        for line in lnf:
            tag, idx = line.strip().split('\t')
            tag_id_dict[tag] = int(idx)
    return tag_id_dict

def convert_to_hot(tag_list, scores, tag_dict):
    hot_list = np.zeros(len(tag_dict))
    for i in range(len(tag_list)):
        hot_list[int(tag_dict[tag_list[i]])] = float(scores[i])
    return hot_list


def parse_gt_json(gt_json, tag_dict):
    gt_dict = {}
    with open(gt_json, "r", encoding='utf-8') as f:
        gts = json.load(f)
    for key in gts:
        x = []
        for ann in gts[key]["annotations"]:
            x.extend(ann['labels'])
        x = list(set(x))
        gt_dict[key] = convert_to_hot(x, np.ones(len(x)), tag_dict)      
    return gt_dict

def parse_input_json(input_json, tag_dict):
    pred_dict = {}
    videos_list = []
    with open(input_json, "r", encoding='utf-8') as f:
        pred_result = json.load(f)
    for video in pred_result:
        videos_list.append(video)
        pred_dict[video] = convert_to_hot(pred_result[video]["result"][0]["labels"],
                                        pred_result[video]["result"][0]["scores"],tag_dict)
    return pred_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_json', type=str, default="test100_pred.json")
    parser.add_argument('--tag_id_file', type=str, default="tag-id-tagging.txt")
    parser.add_argument('--gt_json', type=str, default="test100.json")
    parser.add_argument('--top_k', type=int, default=20)

    args = parser.parse_args()
    
    assert os.path.exists(args.tag_id_file), "dict file {} not found".format(args.tag_id_file)
    tag_dict = get_tag_id_dict(args.tag_id_file)
    
    pred_dict = parse_input_json(args.pred_json, tag_dict)
    gt_dict =  parse_gt_json(args.gt_json, tag_dict)
    
    assert(pred_dict.keys() == gt_dict.keys())
    
    preds, labels = [], []
    for k in pred_dict:
    	preds.append(pred_dict[k])
    	labels.append(gt_dict[k])

    preds = np.stack(preds)
    labels = np.stack(labels)
    gap = calculate_gap(preds, labels, top_k = args.top_k)
    print("The GAP result is {:.3f}".format(gap))
