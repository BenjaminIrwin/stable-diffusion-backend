from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizer
import vptree
from datasets import load_dataset, Dataset

q_encoder = DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
q_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained("facebook/dpr-question_encoder-single-nq-base")

def get_vp_tree(action, number_people):
  ds = load_dataset(str(number_people) + '.csv')
  ds.load_faiss_index('embeddings', str(number_people) + '.faiss')
  question_embedding = q_encoder(**q_tokenizer(action, return_tensors="pt"))[0][0].numpy()
  scores, retrieved_examples = ds.get_nearest_examples('embeddings', question_embedding, k=200)
  keypoints = retrieved_examples[0]
  return vptree.VPTree(keypoints, weightedDistanceMatching)

class keypoint:
  def __init__(self, id, poseVector):
    self.id = id
    self.poseVector = poseVector

def parse_pose_vector(poseVector):
  XY = []
  confidences = []

  for i in range(0, len(poseVector), 3):
    # Extract x, y, and c values from the current triple
    x = poseVector[i]
    y = poseVector[i + 1]
    c = poseVector[i + 2]

    # Append x and y values to x_y_values list
    XY.append(x)
    XY.append(y)

    # Append c value to c_values list
    confidences.append(c)

  # Calculate the sum of all c values ignoring the None values
  confidence_sum = sum(filter(None, confidences))

  # Return the x_y_values, c_values, and confidence_sum
  return XY, confidences, confidence_sum

# Define distance function.
def weightedDistanceMatching(poseVector1, poseVector2):
  vector1PoseXY, vector1Confidences, vector1ConfidenceSum = parse_pose_vector(poseVector1.poseVector)
  vector2PoseXY, vector2Confidences, vector2ConfidenceSum = parse_pose_vector(poseVector2.poseVector)

  # First summation
  summation1 = 1 / vector1ConfidenceSum

  # Second summation
  summation2 = 0
  for i in range(len(vector1PoseXY)):
    tempConf = i // 2
    if vector1Confidences[tempConf] is None or vector2Confidences[tempConf] is None:
        continue
    tempSum = vector1Confidences[tempConf] * abs(vector1PoseXY[i] - vector2PoseXY[i])
    summation2 += tempSum

  return summation1 * summation2