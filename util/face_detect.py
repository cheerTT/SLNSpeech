# @Author : cheertt
# @Time   : 2019/6/28 7:58
# @Remark :

# @Author : cheertt
# @Time   : 2019/6/27 21:49
# @Remark : 自动化工具

import os
import json
import time
from pprint import pformat
from PythonSDK.facepp import API, File


# 一些常量
faceset_folder = './baseImg'       # 用于创建faceSet，一共四张主持人的脸
# face_search_img = './imgResource/018712.jpg'  # 用于人脸搜索，随着文件夹的变化发生变化
faceset_name = 'host_video'

api = API()

def get_first_image(video_path, image_path):
    """
    获取单张图片 暂时没用
    :param video_path: 原始视频路径
    :param image_path: 处理后图像的存放路径
    :return:
    """
    try:
        os.system('ffmpeg -i {0} -r 1 -t 1 {1}/%06d.jpg'.format(video_path, image_path))
    except:
        print('=======================================================')
        print(video_path)
        print('while handling image, something seems to have an error!')


def print_result(hit, result):
    """
    此方法专用来打印api返回的信息
    :param hit:
    :param result:
    :return:
    """
    print(hit)
    print('\n'.join("  " + i for i in pformat(result, width=75).split('\n')))


def printFuctionTitle(title):
    """
    打印函数名
    :param title:
    :return:
    """
    return "\n"+"-"*60+title+"-"*60


def _create_faceset():
    """
    创建包含四张主持人脸的人脸库，此代码只需要调用一次，无需重复调用
    :return:
    """
    time.sleep(2)
    flag = False
    has_facesets = api.faceset.getfacesets()
    for has_face in has_facesets['facesets']:
        if has_face['outer_id'] == faceset_name:
            flag = True

    # 1.创建一个faceSet
    # 若没有 host_video则创建，否则不操作
    # if flag is True:
    #     ret = api.faceset.delete(outer_id=faceset_name, check_empty=0)
    # ret = api.faceset.create(outer_id=faceset_name)
    if flag is False:
        ret = api.faceset.create(outer_id=faceset_name)

    # 2.向faceSet中添加人脸信息(face_token)
    faceResStr = ""
    res1 = api.detect(image_file=File(faceset_folder + '/baoxiaofeng.jpg'))
    res2 = api.detect(image_file=File(faceset_folder + '/guguoning.jpg'))
    res3 = api.detect(image_file=File(faceset_folder + '/hejia.jpg'))
    res4 = api.detect(image_file=File(faceset_folder + '/zhuguangquan.jpg'))

    faceList = []
    faceList.append(res1["faces"][0])
    faceList.append(res2["faces"][0])
    faceList.append(res3["faces"][0])
    faceList.append(res4["faces"][0])

    for index in range(len(faceList)):
        if (index == 0):
            faceResStr = faceResStr + faceList[index]["face_token"]
        else:
            faceResStr = faceResStr + "," + faceList[index]["face_token"]

    lis = api.faceset.getdetail(outer_id=faceset_name)
    lis = lis['face_tokens']
    if len(lis) < 4:  # 若人脸库中不是4张脸，则添加

        faceMap = {}
        faceMap[res1["faces"][0]['face_token']] = 'biaoxiaofeng'
        faceMap[res2["faces"][0]['face_token']] = 'guguoning'
        faceMap[res3["faces"][0]['face_token']] = 'hejia'
        faceMap[res4["faces"][0]['face_token']] = 'zhuguangquan'

        with open('face_map.json', 'w') as json_file:
            json.dump(faceMap, json_file)

        api.faceset.addface(outer_id=faceset_name, face_tokens=faceResStr)

    with open('face_map.json', 'r') as json_file:
        faceMap = json.load(json_file)

    return faceMap


def who_host(face_search_img):
    """
    在人脸集中寻找传入图片是否是主持人视频
    :param face_search_img:
    :return:
    """
    faceMap = _create_faceset()
    # 3.开始搜索相似脸人脸信息
    # 如果发生异常 继续发送
    try:
        search_result = api.search(image_file=File(face_search_img), outer_id=faceset_name)
    except:
        return None
    # print_result('search', search_result)
    if len(search_result['faces']) == 0:
        return None
    else:
        key = search_result['results'][0]['face_token']
        confidence = search_result['results'][0]['confidence']
        thresholds = search_result['thresholds']['1e-5']
        if float(confidence) >= float(thresholds):
            return faceMap[key]
        else:
            return None


def is_face(detech_img_url):
    res = api.detect(image_file=File(detech_img_url))
    # print_result(printFuctionTitle("人脸检测"), res)
    if len(res['faces']) == 0:
        return False
    else:
        return True


if __name__ == '__main__':

    # 视频单张图片
    # video_path = r'E:\codes\git\cut_video_and_generate_color_with_python-opencv\cut\《共同关注》20190601\《共同关注》20190601_109.avi'
    # image_path = 'res'
    # get_first_image(video_path, image_path)

    # api.faceset.delete(outer_id='faceplusplus', check_empty=0)
    #
    # ret = api.faceset.getfacesets(outer_id='faceplusplus')
    #
    # # 删除无用的人脸库，这里删除了，如果在项目中请注意是否要删除
    # api.faceset.delete(outer_id='host_video', check_empty=0)
    # print(ret)
    # ret = api.faceset.getfacesets(outer_id='host_video')
    # print(ret)
    #
    # li = api.faceset.getdetail(outer_id='host_video')
    # print(li)
    #
    # res4 = api.detect(image_file=File(faceSet_folder + '/4.jpg'))
    # print(res4)


    # import PythonSDK.facepp.APIError
    # try:
    #     aa = api.faceset.create(outer_id='cheertt')
    # except:

    # aa = api.faceset.getdetail(outer_id='cheertt')

    # aa = api.faceset.create(outer_id='host_video')
    # print(aa)

    # li = api.faceset.getdetail(outer_id='host_video')
    # li = li['face_tokens']
    # print(li)
    # print(len(li))

    # face_tokena = api.detect(image_file=File(faceset_folder + '/zhuguangquan.jpg'))
    #     # print(face_tokena)

    faceMap = _create_faceset()
    print(faceMap)

    li = api.faceset.getdetail(outer_id='host_video')
    li = li['face_tokens']
    print(li)

    face_search_img = r'E:\codes\pycharm\pycharm_paper\automated-cutting-video-tool\host_video\frames\312321\20190603_3\000007.jpg'
    # face_search_img = 'baseImg/baoxiao.jpg'
    search_result = api.search(image_file=File(face_search_img), outer_id='host_video')

    key = search_result['results'][0]['face_token']
    print(key)
    print(faceMap[key])
