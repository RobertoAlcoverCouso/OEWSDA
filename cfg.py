
means=[86.5628/ 255.,86.6691/ 255.,86.7348/ 255.] 
std=[0.229, 0.224, 0.225]
datasets_names = {"Cityscapes": "./../data/Cityscapes/trainCS.csv",
                    "CityscapesVal": "./../data/Cityscapes/valCS.csv",
                      "CityscapesCS": "./../data/Cityscapes/trainCS.csv",
                      "CityscapesValCS": "./../data/Cityscapes/valCS.csv",
                      "Cityscapes_test": "./../data/Cityscapes/testCS.csv",
                      "Synthia": './../data/Synthia/train.csv',
                      "Kitti": "./../data/Kitti/training/train.csv",
                      "MSS": './../data/MSS/trainCS.csv',
                      "Mapilliary": './../data/Mapilliary/data/train.csv',
                      "MapilliaryVal": './../data/Mapilliary/data/val.csv',
                      "MSS50":         './../data/MSS/50CS.csv',
                      "MSS100":         './../data/MSS/100CS.csv',
                      "MSS250":         './../data/MSS/250CS.csv',
                      "MSS750":         './../data/MSS/750CS.csv',
                      "GTAV":        './../data/GTAV/trainCS.csv',
                      }

complete_datasets = ["Kitti",  "Cityscapes","Mapilliary", "Synthia", "MSS", "GTAV"]
real_datasets_train = [ "CityscapesCS","Dron","Kitti","Mapilliary"] 
synthetic_datasets = ["MSS50",  "MSS", "MSS100", "MSS250", "MSS750", "GTAV", "Synthia"]
proportions = [.05, .15, .25, .5]#
