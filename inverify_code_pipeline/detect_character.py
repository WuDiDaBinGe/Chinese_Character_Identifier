import numpy as np
import os
import tensorflow as tf
from PIL import Image

MODEL_PATH= "../detect_models/saved_model_new"
TEST_IMAGE_DIR="./t_images"
TEST_IMAGE_PATH=[os.path.join(TEST_IMAGE_DIR,'image{}.jpg'.format(i)) for i in range(1,7)]

if not os.path.exists(TEST_IMAGE_PATH[1]):
    print("不存在模型文件！")
else:
    print("存在模型路径！")

def load_model(model_dir):
  model = tf.keras.models.load_model(model_dir)
  return model


def load_image_into_numpy_array(image):
    (im_width,im_height)=image.size
    return np.array(image.getdata()).reshape(
        (im_height,im_width,3)).astype(np.uint8)

def run_inference_for_single_image(model, image):
    image = np.asarray(image)
    # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
    input_tensor = tf.convert_to_tensor(image)
    # The model expects a batch of images, so add an axis with `tf.newaxis`.
    input_tensor = input_tensor[tf.newaxis, ...]

    # Run inference
    model_fn = model.signatures['serving_default']
    output_dict = model_fn(input_tensor)

    # All outputs are batches tensors.
    # Convert to numpy arrays, and take index [0] to remove the batch dimension.
    # We're only interested in the first num_detections.
    num_detections = int(output_dict.pop('num_detections'))
    output_dict = {key: value[0, :num_detections].numpy()
                   for key, value in output_dict.items()}
    output_dict['num_detections'] = num_detections

    # detection_classes should be ints.
    output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)

    return output_dict

def detect_character(picturs_input_path,detectmodel_path):
    '''
    :param picturs_input_path:验证码图片
    :param detectmodel_path:检测模型的路径
    :return:需要剪裁的汉字的位置，保存的汉字的文件名
    '''
    pictures_save_path,image_name=os.path.split(picturs_input_path)
    image_name=image_name.split(".")[0]
    # 载入模型
    detect_model_=load_model(detectmodel_path)
    # 读取图片
    image = Image.open(picturs_input_path)
    image_np = load_image_into_numpy_array(image)
    output_dict = run_inference_for_single_image(detect_model_, image_np)
    boxes = output_dict['detection_boxes']
    scores = output_dict['detection_scores']
    boxes_to_cut = []
    min_score_thresh = .5
    for i in range(boxes.shape[0]):
        if scores[i] > min_score_thresh:
            box = tuple(boxes[i].tolist())
            boxes_to_cut.append(box)
    character_save_list=[]
    boxes_to_cut_src=[]
    for i, box in enumerate(boxes_to_cut):
        ymin, xmin, ymax, xmax = box
        im_width, im_height = image.size
        rect = (xmin * im_width, xmax * im_width,
                ymin * im_height, ymax * im_height)
        boxes_to_cut_src.append(rect)
        new_img_64 = get_boundingbox_image(image, rect, 2, 64)
        # 不保存图片名称
        img_save_path=pictures_save_path + "/" + image_name + "__" +str(i) + ".jpg"
        character_save_list.append(img_save_path)
        new_img_64.save(img_save_path)
    return boxes_to_cut_src,character_save_list

def get_boundingbox_image(image, rect, scale,new_size):
    '''
    image:PIL.image 类型的图像
    rect：包含四个坐标的元组
    scale：从rect向外扩展的像素数
    new_size:生成的新的图像的尺寸
    return：new_image PIL.image
    '''
    im_width, im_height = image.size
    (left, right, top, bottom)=rect
    # 进行扩展后ROI区域和边界的判断
    n_left=0 if left-scale<0 else left-scale
    n_right=im_width if right+scale>im_width else right+scale
    n_top = 0 if top-scale<0 else top-scale
    n_bottom=im_height if bottom+scale>im_height else bottom+scale

    new_img=image.crop((n_left,n_top,n_right,n_bottom))
    new_img=new_img.resize((new_size,new_size)) # 转化图片
    return new_img


if __name__ == '__main__':
    input_pic_path="./t_images"
    characters_save_path="./pics_name"






