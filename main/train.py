import warnings, os
warnings.filterwarnings('ignore')
from ultralytics import RTDETR

if __name__ == '__main__':
      
    model = RTDETR(r'E:\Code\DD-DETR\main\models\ours\DD-DETR.yaml')
    # model.load('/home/ubuntu/proejcts/RTDETR-main/weights/rtdetr-r18.pt') # loading pretrain weights
    model.train(data=r'E:\Code\DD-DETR\main\dataset\visdrone.yaml',
                cache=False,
                imgsz=640,
                epochs=2,
                batch=4,
                workers=4, 
                device='0', 
                #resume=r'', # last.pt path
                project='runs/train',
                name='test',
                )
    

 
    
    

    

