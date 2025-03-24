import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO


if __name__ == '__main__':
    model = YOLO('ultralytics/cfg/models/v8/gd_yolov8_ghost.yaml', task='detect') # select your model.yaml path
    model.load('yolov8n.pt') # loading pretrain weights
    model.train(data='/home/user/cv/braillev8/data.yaml', # select your data.yaml path
                cache=False,
                imgsz=640,
                epochs=50,
                batch=8,
                patience=30, 
                close_mosaic=10,
                workers=8,
                device='0',
                # optimizer='SGD', # using SGD 
                # resume='', # last.pt path
                # amp=False, # close amp
                # fraction=0.2,
                project='runs/train',
                name='test1',
                )
