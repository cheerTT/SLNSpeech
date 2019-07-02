# @Author : cheertt
# @Time   : 2019/6/29 15:30
# @Remark :

import os
import pickle
import numpy as np
import tensorflow as tf
from scipy import misc
from PIL import Image
from FacenetSDK.facenet import prewhiten, crop,flip
from FacenetSDK.align.detect_face import create_mtcnn, detect_face
from FacenetSDK.facenet import get_model_filenames

os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'  # 只显示 warning 和 Error

MODELPATH = os.path.join(os.getcwd(), "FacenetSDK", "20180408-102900")
classifier_pkl_path = os.path.join(os.getcwd(), "FacenetSDK", "host_video.pkl")

minsize = 20  # minimum size of face
threshold = [0.6, 0.7, 0.7]  # three steps's threshold
factor = 0.709  # scale factor

# 该段代码占用过多CPU资源，放在全局供其他函数调用
with tf.Graph().as_default():
    gpu_memory_fraction = 0.25
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
    # sess = tf.Session()
    with sess.as_default():
        pnet, rnet, onet = create_mtcnn(sess, None)

with tf.Graph().as_default():
    sess = tf.Session()
    # 加载模型
    meta_file, ckpt_file = get_model_filenames(MODELPATH)
    saver = tf.train.import_meta_graph(os.path.join(MODELPATH, meta_file))
    saver.restore(sess, os.path.join(MODELPATH, ckpt_file))
    # 获得输入输出张量
    images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
    embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
    phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

    print('Creating networks and loading parameters ended.')


    def _image_array_align_data(image_arr, image_path, pnet, rnet, onet, image_size=160, margin=32,
                               detect_multiple_faces=True):
        """
        截取人脸的类
        :param image_arr: 人脸像素点数组
        :param image_path: 拍摄人脸存储路径
        :param pnet: caffe模型
        :param rnet: caffe模型
        :param onet: caffe模型
        :param image_size: 图像大小
        :param margin: 边缘截取
        :param gpu_memory_fraction: 允许的gpu内存大小
        :param detect_multiple_faces: 是否可以识别多张脸，默认为False
        :return: 若成功，返回截取的人脸数组集合如果没有检测到人脸，直接返回一个1*3的0矩阵
        """

        img = image_arr
        bounding_boxes, _ = detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
        nrof_faces = bounding_boxes.shape[0]

        nrof_successfully_aligned = 0
        if nrof_faces > 0:
            det = bounding_boxes[:, 0:4]
            det_arr = []
            img_size = np.asarray(img.shape)[0:2]
            if nrof_faces > 1:
                if detect_multiple_faces:
                    for i in range(nrof_faces):
                        det_arr.append(np.squeeze(det[i]))
                else:
                    bounding_box_size = (det[:, 2] - det[:, 0]) * (det[:, 3] - det[:, 1])
                    img_center = img_size / 2
                    offsets = np.vstack(
                        [(det[:, 0] + det[:, 2]) / 2 - img_center[1], (det[:, 1] + det[:, 3]) / 2 - img_center[0]])
                    offset_dist_squared = np.sum(np.power(offsets, 2.0), 0)
                    index = np.argmax(
                        bounding_box_size - offset_dist_squared * 2.0)  # some extra weight on the centering
                    det_arr.append(det[index, :])
            else:
                det_arr.append(np.squeeze(det))

            images = np.zeros((len(det_arr), image_size, image_size, 3))
            res = []
            for i, det in enumerate(det_arr):
                det = np.squeeze(det)
                bb = np.zeros(4, dtype=np.int32)
                bb[0] = np.maximum(det[0] - margin / 2, 0)
                bb[1] = np.maximum(det[1] - margin / 2, 0)
                bb[2] = np.minimum(det[2] + margin / 2, img_size[1])
                bb[3] = np.minimum(det[3] + margin / 2, img_size[0])
                cropped = img[bb[1]:bb[3], bb[0]:bb[2], :]
                # 进行图片缩放 cv2.resize(img,(w,h))
                scaled = misc.imresize(cropped, (image_size, image_size), interp='bilinear')
                nrof_successfully_aligned += 1

                # 保存检测的头像
                filename_base = os.path.join(os.getcwd(), "tmp_cut_image")

                if not os.path.exists(filename_base):
                    os.mkdir(filename_base)

                filename = os.path.basename(image_path)
                filename_name, file_extension = os.path.splitext(filename)
                # 多个人脸时，在picname后加_0 _1 _2 依次累加。
                output_filename_n = "{}/{}_{}{}".format(filename_base, filename_name, i, file_extension)
                res.append(output_filename_n)
                misc.imsave(output_filename_n, scaled)

                scaled = prewhiten(scaled)
                scaled = crop(scaled, False, 160)
                scaled = flip(scaled, False)

                images[i] = scaled
        if nrof_faces > 0:
            return res
        else:
            return None


    def _mtcnn_face(image_path):
        """
        基于 mtcnn 抽取人脸并保存
        :param image_path:
        :return:
        """
        # opencv读取图片，开始进行人脸识别
        img = misc.imread(os.path.expanduser(image_path), mode='RGB')
        # 设置默认插入时 detect_multiple_faces =False只检测图中的一张人脸，True则检测人脸中的多张
        # 一般入库时只检测一张人脸，查询时检测多张人脸
        res = _image_array_align_data(img, image_path, pnet, rnet, onet, detect_multiple_faces=True)

        return res


    def _predict(image_path):

        images, count_per_image = _load_and_align_data(image_path, 160, 44, )
        if len(count_per_image) == 0:
            return None
        # Run forward pass to calculate embeddings
        feed_dict = {images_placeholder: images, phase_train_placeholder: False}
        emb = sess.run(embeddings, feed_dict=feed_dict)
        classifier_filename_exp = os.path.expanduser(classifier_pkl_path)
        with open(classifier_filename_exp, 'rb') as infile:
            (model, class_names) = pickle.load(infile)
        print('Loaded classifier model from file "%s"\n' % classifier_filename_exp)
        predictions = model.predict_proba(emb)
        best_class_indices = np.argmax(predictions, axis=1)
        best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]
        k = 0
        # print predictions
        for j in range(len(count_per_image)):
            print('%s: %.3f' % (class_names[best_class_indices[k]], best_class_probabilities[k]))
            if best_class_probabilities[k] >= 0.95:
                return class_names[best_class_indices[k]]
            k += 1
        return None


def _load_and_align_data(image_path, image_size, margin, gpu_memory_fraction=0.25):

    img_list = []
    count_per_image = []
    img = misc.imread(os.path.expanduser(image_path))
    img_size = np.asarray(img.shape)[0:2]
    bounding_boxes, _ = detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
    count_per_image.append(len(bounding_boxes))
    for j in range(len(bounding_boxes)):
        det = np.squeeze(bounding_boxes[j, 0:4])
        bb = np.zeros(4, dtype=np.int32)
        bb[0] = np.maximum(det[0] - margin / 2, 0)
        bb[1] = np.maximum(det[1] - margin / 2, 0)
        bb[2] = np.minimum(det[2] + margin / 2, img_size[1])
        bb[3] = np.minimum(det[3] + margin / 2, img_size[0])
        cropped = img[bb[1]:bb[3], bb[0]:bb[2], :]
        aligned = misc.imresize(cropped, (image_size, image_size), interp='bilinear')
        prewhitened = prewhiten(aligned)
        img_list.append(prewhitened)
    if len(img_list) == 0:
        return _, []
    else:
        images = np.stack(img_list)
        return images, count_per_image


def is_face(image_path):
    return _mtcnn_face(cut_image_with_leftbottom(image_path))


def who_host(image_path):
    return _predict(image_path)


def cut_image_with_leftbottom(image_path):
    img = Image.open(image_path)
    cropped = img.crop((260, 0, 720, 576))
    # tt = image_path.replace('tmp_middle', 'tmp_middle1')
    # cropped.save(tt)
    # return tt
    return image_path


if __name__ == '__main__':
    # image_path = r'F:\datasets\host_video\train_datasets\hejia\000156.jpg'
    # image_path = r'F:\tmp\pic\9.jpg'
    # image_path = r'F:\datasets\host_video\train_datasets\baoxiaofeng\000566.jpg'
    # is_face = mtcnn_face(cut_image_with_leftbottom(image_path))

    # res = is_face(image_path)
    # print(len(res))
    image_path = r'E:\codes\pycharm\pycharm_paper\automated-cutting-video-tool\tmp_cut_image\20190611_178_178_0.jpg'
    print(_predict(image_path))

