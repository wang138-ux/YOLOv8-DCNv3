import sys

sys.path.append("autodl-tmp/yolov8/")

from ultralytics import YOLO

if __name__ == '__main__':
    # Load a model
    # 直接使用预训练模型创建模型
    model = YOLO('yolov8s.pt')
    model.train(**{'cfg':'ultralytics/cfg/default.yaml', 'data':'datasets/wheat/data.yaml'}, epochs=300, imgsz=640, batch=16, name='YOLOv8-xxx')

    # #使用yaml配置文件来创建模型，并导入预训练权重
    #model = YOLO('ultralytics/cfg/models/v8/yolov8-dcnv3.yaml')  # build a new model from YAML
    #model.train(**{'cfg':'ultralytics/cfg/default.yaml', 'data':'datasets/wheat/data.yaml'}, epochs=300, imgsz=640, batch=16, name='YOLOv8-DCNv3-xxx')  #name=YOLOv8-DCNv3-xxx 更改文件夹的名字

