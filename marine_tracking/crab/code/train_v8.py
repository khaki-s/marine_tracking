from ultralytics import YOLO

if __name__=='__main__':
    model=YOLO('yolov8x.pt')
    model.train(data='data/data.yaml',epochs=300)