from ultralytics import YOLO


# Load a model
model = YOLO("/ultralytics-main/ultralytics/cfg/models/11/yoloxray.yaml")  # build a new model from scratch

# model.load("yolov9s.pt")  # load a pretrained model 不使用预训练权重，就注释这一行即可
# train
model.train(data='cfg/datasets/OPIXray.yaml',
                cache=False,
                imgsz=640,
                epochs=200,
                batch=16,
                close_mosaic=0,
                workers=4,
                device='0',
                optimizer='SGD', # using SGD
                amp=False, # close amp
                project='./runs/train',
                name='exp',
                )
