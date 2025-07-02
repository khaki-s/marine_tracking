from ultralytics import YOLO

if __name__=='__main__':
    model=YOLO('yolov8n.pt')
    model.train(data='data/data.yaml',epochs=200)