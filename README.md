1 准备好视频数据
需要保证文件夹仅包含需要分割的视频，且视频格式是mp4

2 拷贝文件夹20180408-102900（之前给过）到FacenetSDK下

3 `python main.py [filepath]`

4 数据集均在host_video文件夹下 

注：若项目下生成log.txt文件，则把该文件单独上传。cut文件夹无须再关注

为了能够把共同关注中主持人说话的片段截取出来制作成自监督多声分离的数据集，代码参照  
https://github.com/sunkaiiii/cut_video_and_generate_color_with_python-opencv  
https://github.com/davidsandberg/facenet
