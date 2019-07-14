# @Author : cheertt
# @Time   : 2019/6/27 10:17
# @Remark :

import os
import json
import cv2
import subprocess
import platform


# 如：'zhuguangquan', 'guguoning', 'baoxiaofeng', 'hejia'
filename = '文件名-每次需要手动编辑'
jsonname = 'raw视频json文件名称'
# 版本1数据集只有四个名称，即四个主持人的名称，固定死
res = ['zhuguangquan', 'guguoning', 'baoxiaofeng', 'hejia']


def get_image(video_path, image_path):
    """
    获取视频对应的图片帧
    :param video_path: 原始视频路径
    :param image_path: 处理后图像的存放路径
    :return:
    """
    try:
        os.system('ffmpeg -i {0} -vf fps=fps=8/1 -q 0 {1}/%06d.jpg'.format(video_path, image_path))
    except:
        print('=======================================================')
        print(video_path)
        print('while handling image, something seems to have an error!')


def get_audio(video_path, audio_path):
    """
    获取视频中的声音，声音波形为11025，单声道，wav格式
    :param video_path: 原始视频路径
    :param image_path: 处理后图像的存放路径
    :return:
    """
    try:
        os.system('ffmpeg -i {0} -ac 1 -ar 11025 {1}.mp3'.format(video_path, audio_path))
    except:
        print('=======================================================')
        print(video_path)
        print('while handling audio, something seems to have an error!')


def get_first_image(video_path, image_name):
    """
    获取单张图片
    :param video_path: 原始视频路径
    :param image_path: 处理后图像的存放路径
    :return:
    """
    try:
        os.system('ffmpeg -ss 00:00:02 -t 1 -i {0} -r 1 {1}'.format(video_path, image_name))
    except:
        print('=======================================================')
        print(video_path)
        print('while handling image, something seems to have an error!')


def get_last_image(ss, video_path, image_name):
    """
    获取单张图片,从后往前
    :param video_path: 原始视频路径
    :param image_path: 处理后图像的存放路径
    :return:
    """
    ss = sec2hoursecondsec(ss)
    try:
        os.system('ffmpeg -ss {0} -t 1 -i {1} -r 1 -frames:v 1 {2}'.format(ss, video_path, image_name))
        # os.system('ffmpeg -ss {0} -t 1 -i {1} -vcodec copy -frames:v 1 {2}'.format(ss, video_path, image_name))
    except:
        print('=======================================================')
        print(video_path)
        print('while handling image, something seems to have an error!')


def sec2hoursecondsec(sec):
    sec = int(float(sec))
    h = "%02d" % int(sec / 3600)
    m = "%02d" % int(sec % 3600 / 60)
    s = "%02d" % int(sec % 60)
    return str(h) + ':' + str(m) + ':' + str(s)


if __name__ == '__main__':
    # with open(jsonname, 'r') as load_f:
    #     load_dict = json.load(load_f)
    #     tmp = load_dict['videos']
    #     for r in res:
    #         lis = tmp.get(res[0])  # 获取某一个主持人的视频列表
    #         print('start......')
    #         for li in lis:
    #             # 判断当前路径是否存在，没有则创建new文件
    #             folder_path = 'host_video/frames/' +filename + '/' + li
    #             if not os.path.exists(folder_path):
    #                 os.makedirs(folder_path)
    #
    #             # 处理操作，获取图片
    #             video_path = 'host_video_raw/' + filename + '.mp4'
    #             image_path = folder_path
    #             get_image(video_path, image_path)
    #
    # print('success!')

    # video_path = r'E:\CNTV\Download\《共同关注》20140122\《共同关注》20140122.mp4'
    # image_path = r'F:\datasets\host_video\20190711\1'
    # get_image(video_path, image_path)

    # video_path = r'E:\codes\git\automated-cutting-video-tool\middle_file\《共同关注》20180101\《共同关注》20180101_10.avi'
    # video_path = r'E:\CNTV\Download\共同关注\《共同关注》20180101.mp4'
    # res = get_video_time(video_path)
    # print(sec2hoursecondsec(res))

    # video_path = r'E:\CNTV\Download\共同关注\《共同关注》20180101.mp4'
    # video_path = r'E:\codes\git\automated-cutting-video-tool\middle_file\《共同关注》20180101\《共同关注》20180101_0.avi'
    # image_name = 'a.jpg'
    # get_last_image(video_path, image_name)


    # video_path = r'E:\codes\git\automated-cutting-video-tool\middle_file\《共同关注》20180101\《共同关注》20180101_0.avi'
    # clip = VideoFileClip(video_path)
    # print(clip.duration)

    # video_path = r'F:\cntv\201901\《共同关注》20190105.mp4'
    # res = get_video_time(video_path)
    # print(res)

    # get_last_image(video_path, 'e.jpg')

    video = r'E:\codes\git\automated-cutting-video-tool\middle_file\《共同关注》20190101\《共同关注》20190101_35.avi'
    # a = get_video_time(video)
    # print(a)

