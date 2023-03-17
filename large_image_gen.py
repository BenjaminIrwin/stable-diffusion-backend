from PIL import Image, ImageDraw
import numpy as np

LANCZOS = (Image.Resampling.LANCZOS if hasattr(Image, 'Resampling') else Image.LANCZOS)

def get_mask_region(mask):
    h, w = mask.shape
    # print(f'image shape: {h} {w}')

    x1 = 0
    for i in range(w):
        if not (mask[:, i] == 0).all():
            break
        x1 += 1

    x2 = 0
    for i in reversed(range(w)):
        if not (mask[:, i] == 0).all():
            break
        x2 += 1
    x2 = w-x2

    y1 = 0
    for i in range(h):
        if not (mask[i] == 0).all():
            break
        y1 += 1

    y2 = 0
    for i in reversed(range(h)):
        if not (mask[i] == 0).all():
            break
        y2 += 1
    y2 = h-y2

    return (int(x1), int(y1), int(x2), int(y2))


def get_crop_region(x1, y1, x2, y2, mask_w, mask_h, pad=0, target_res=512):

    crop_left = x1 - pad
    crop_top = y1 - pad
    crop_right = x2 + pad
    crop_bottom = y2 + pad

    while crop_right - crop_left < target_res:
        crop_right += 1
        crop_left -= 1
    
    while crop_bottom - crop_top < target_res:
        crop_bottom += 1
        crop_top -= 1

    while crop_right - crop_left > target_res:
        if (crop_right - crop_left) - target_res == 1:
            crop_left += 1
        else:
            crop_right -= 1
            crop_left += 1

    while crop_bottom - crop_top > target_res:
        if (crop_bottom - crop_top) - target_res == 1:
            crop_top += 1
        else:
            crop_bottom -= 1
            crop_top += 1

    if crop_left < 0:
        crop_right = target_res
        crop_left = 0

    if crop_top < 0:
        crop_bottom = target_res
        crop_top = 0

    if crop_right > mask_w:
        crop_right = mask_w
        crop_left = mask_w - target_res

    if crop_bottom > mask_h:
        crop_bottom = mask_h
        crop_top = mask_h - target_res

    return (int((crop_left)), int((crop_top)), int((crop_right)), int((crop_bottom)))

image_path = "C:\\Users\\guill\\Desktop\\script_testing\\mask_risizing_tests\\1.ready_for_test\\img\\ritook_4ivqdq.jpg"
mask_path = "C:\\Users\\guill\\Desktop\\script_testing\\mask_risizing_tests\\1.ready_for_test\\msk\\m_ritook_4ivqdq_0.jpg"

# "C:\Users\guill\Desktop\script_testing\training_script\dataset_1024_person\masks\person_ramat_2nydlt_9.jpg"

padding = 35
target_resolution = 512
pil_image = Image.open(image_path).convert('RGB')
pil_mask = Image.open(mask_path).convert('L')

x1, y1, x2, y2 = get_mask_region(np.array(pil_mask))
box_w = x2 - x1
box_h = y2 - y1
max_box_dimension = max(box_w, box_h)

# Only does transformation of the box is too big 
if max_box_dimension + padding > target_resolution:
    mask_w, mask_h = pil_mask.size

    if max_box_dimension > min(mask_w, mask_h):
        raise ValueError('Irregular image size: one edge of the inpainting box is longer than one of the sides of the image. Either draw a smaller box or choose a different image.')

    padding_percent = padding/target_resolution
    box_dimension_with_padding = int(max_box_dimension * (1+padding_percent))
    crop_dimension = min(box_dimension_with_padding, mask_w, mask_h)

    crop_padding = 0
    if box_dimension_with_padding == crop_dimension:
        crop_padding = (box_dimension_with_padding - max_box_dimension)/2

    crop_region = get_crop_region(x1, y1, x2, y2, mask_w, mask_h, crop_padding, crop_dimension)

    temp_mask = pil_mask.crop(crop_region)
    temp_image = pil_image.crop(crop_region)

    # feed those two to the model
    temp_image = temp_image.resize((target_resolution, target_resolution), resample=LANCZOS)
    temp_mask = temp_mask.resize((target_resolution, target_resolution), resample=LANCZOS)

    #img_output = model(something something)
    img_output = temp_image #delete that line once the script is hooked up to the model
    img_output = img_output.resize((crop_dimension, crop_dimension), resample=LANCZOS)

    result_img = pil_image.copy()
    result_img.paste(img_output, (crop_region[0], crop_region[1]))

# if the box is not too big
else:
    print('do the normal thing')




