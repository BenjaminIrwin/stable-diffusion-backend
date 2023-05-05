import json

import numpy as np
from PIL import Image, ImageOps
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

def get_image_pose_vector(image):

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
    normalised_image, og_size, og_bounding_box = normalise_input_image(image, open_pose.get(0))
    normalised_open_pose = \
    open_pose_model(np.array(normalised_image.convert('RGB')), include_body=True, include_hand=False,
                    include_face=False,
                    return_is_index=True).get(0)
    return PoseVector(vector=normalised_open_pose, image=image, original_size=og_size, original_bb=og_bounding_box)


from PIL import Image


def normalise_input_image(image, vector):

    minX = None
    minY = None
    maxX = None
    maxY = None
    for i in range(0, len(vector), 3):
        if vector[i + 2] is None:
            continue
        x = vector[i] * image.width
        y = vector[i + 1] * image.height
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

    og_size = image.size
    og_bounding_box = [minX, minY, maxX, maxY]

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
    image_resized = image.resize((768, 768), Image.LANCZOS)

    return image_resized, og_size, og_bounding_box


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
    def __init__(self, id=None, vector=None, image=None, original_size=None, original_bb=None):
        self.id = id
        self.vector = vector
        self.original_size = original_size
        self.original_bb = original_bb
        self.image = None

    def set_original_size(self):
        self.original_size = self.image.size

    def set_bb(self):
        minX = None
        minY = None
        maxX = None
        maxY = None
        for i in range(0, len(self.vector), 3):
            if self.vector[i + 2] is None:
                continue
            x = self.vector[i] * self.vector.image.width
            y = self.vector[i + 1] * self.vector.image.height
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

        self.original_bb = [minX, minY, maxX, maxY]

    def align_to_vector(self, pose_vector):
        if pose_vector.original_size is None:
            pose_vector.set_original_size()
        if pose_vector.original_bb is None:
            pose_vector.get_bb()
        if self.original_size is None:
            self.set_original_size()
        if self.original_bb is None:
            self.original_size()


        # Create an image the size of pose_vector.image
        image = Image.new('RGBA', pose_vector.original_size, (0, 0, 0, 0))

        # Get difference in scale between the heights of the original_bbs
        scale = (pose_vector.original_bb[3] - pose_vector.original_bb[1]) / (self.original_bb[3] - self.original_bb[1])

        # Crop self.image to self.original_bb
        crop = self.image.crop(self.original_bb)

        # Scale crop
        crop = crop.resize((int(crop.width * scale), int(crop.height * scale)), Image.LANCZOS)

        image.paste(crop, (pose_vector.original_bb[0], pose_vector.original_bb[1]))

        return image





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
    vector1PoseXY, vector1Confidences, vector1ConfidenceSum = parse_pose_vector(poseVector1.vector)
    vector2PoseXY, vector2Confidences, vector2ConfidenceSum = parse_pose_vector(poseVector2.vector)

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
