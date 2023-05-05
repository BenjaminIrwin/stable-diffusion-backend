import json

import numpy as np
from PIL import Image
from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizer
import vptree
from datasets import load_dataset, Dataset
import pandas as pd

import base64
from io import BytesIO

from openpose_hijack.openpose import OpenposeDetector

open_pose_model = OpenposeDetector()

q_encoder = DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
q_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained("facebook/dpr-question_encoder-single-nq-base")


def get_image_open_pose(image):

    # Convert image to base64
    print('IMAGE STRING')
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue())
    print(img_str)


    # Get point by doing openpose
    open_pose = open_pose_model(np.array(image.convert('RGB')), include_body=True, include_hand=False, include_face=False,
                    return_is_index=True)
    print('GENERATED OPEN_POSE')
    print(open_pose)
    normalised_image = normalise_input_image(image, open_pose['0'])
    normalised_open_pose = \
    open_pose_model(np.array(normalised_image.convert('RGB')), include_body=True, include_hand=False,
                    include_face=False,
                    return_is_index=True)['0']
    return normalised_open_pose


def normalise_input_image(image, open_pose):
    minX = None
    minY = None
    maxX = None
    maxY = None
    # Iterate through every entry in open_pose dict
    for key in open_pose.keys():
        points = open_pose[key]
        # Iterate through points in format (x1, y1, c1, x2, y2, c2, ...)
        for i in range(0, len(points), 3):
            if points[i + 2] is None:
                continue
            x = points[i] * image.width
            y = points[i + 1] * image.height
            if minX is None or x < minX:
                minX = x
            if minY is None or y < minY:
                minY = y
            if maxX is None or x > maxX:
                maxX = x
            if maxY is None or y > maxY:
                maxY = y

    # Check if minX, minY, maxX, maxY are None
    if minX is None or minY is None or maxX is None or maxY is None:
        return None

    # Add 35% padding to crop region
    width = maxX - minX
    height = maxY - minY
    crop_region = [minX - width * 0.35, minY - height * 0.35, maxX + width * 0.35, maxY + height * 0.35]

    # Check if crop_region bigger than image
    if crop_region[0] < 0:
        crop_region[0] = 0
    if crop_region[1] < 0:
        crop_region[1] = 0
    if crop_region[2] > image.width:
        crop_region[2] = image.width
    if crop_region[3] > image.height:
        crop_region[3] = image.height

    # Crop image
    image = image.crop((crop_region[0], crop_region[1], crop_region[2], crop_region[3]))

    # Pad smaller dimension to make image square
    width = image.width
    height = image.height

    if width > height:
        # Get padding
        padding = (width - height) // 2
        # Add padding to top and bottom in the form of transparent pixels
        image_new = Image.new('RGBA', (width, width), (0, 0, 0, 0))
        image_new.paste(image, (0, padding))
        image = image_new
    elif height > width:
        # Get padding
        padding = (height - width) // 2
        # Add padding to left and right in the form of transparent pixels
        image_new = Image.new('RGBA', (height, height), (0, 0, 0, 0))
        image_new.paste(image, (padding, 0))
        image = image_new

    # Resize image to 768x768 using lanczos filter
    return image.resize((768, 768), Image.LANCZOS)


def get_vp_tree(action, number_people):

    # Use pandas to open csv file
    df = pd.read_csv('people_count_1.csv')
    # Get id, pose_data columns
    pose_data = df[['id', 'pose_data']].values.tolist()

    pose_vectors = []

    print('Loading keypoints...')

    # Iterate through every row of pose_data
    for row in pose_data:
        # Create keypoint object
        vector = json.loads(row[1]).get('0')
        if vector is not None:
            pose_vector_object = PoseVector(row[0], vector)
            # Append keypoint object to keypoints list
            pose_vectors.append(pose_vector_object)

    print('Creating VPTree...')

    return vptree.VPTree(pose_vectors, weightedDistanceMatching)


class PoseVector:
    def __init__(self, id, vector):
        self.id = id
        self.poseVector = vector


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
