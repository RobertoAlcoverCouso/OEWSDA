
means=[86.5628/ 255.,86.6691/ 255.,86.7348/ 255.] 
std=[0.229, 0.224, 0.225]
datasets_names = {"Cityscapes": "./../data/Cityscapes/train.csv",
                      "CityscapesCS": "./../Cityscapes/trainCS.csv",
                      "CityscapesValCS": "./../data/Cityscapes/val.csv",
                      "Cityscapes_test": "./../Cityscapes/test.csv",
                      "Coche": './../MSS_vids/data/images/Coche.csv',
                      "Synthia": './../data/Synthia/train.csv',
                      "Kitti": "./../Kitti/training/train.csv",
                      "MSS": './../MSS/data/trainCS.csv',
                      "Bus": './../MSS_vids/data/images/AutoBus.csv',
                      "Dron": './../semantic_drone_dataset/train.csv',
                      "Helicoptero": './../MSS_vids/data/images/Helicoptero.csv',
                      "Peaton": './../MSS_vids/data/images/Peaton.csv',
                      "Video": './../MSS_vids/data/images/Video.csv',
                      "MSS5": './../MSS_vids/data5/train.csv',
                      "Coche5": './../MSS_vids/data5/images/Coche.csv',
                      "Bus5": './../MSS_vids/data5/images/AutoBus.csv',
                      "Helicoptero5": './../MSS_vids/data5/images/Helicoptero.csv',
                      "Peaton5": './../MSS_vids/data5/images/Peaton.csv',
                      "Video5": './../MSS_vids/data5/images/Video.csv',
                      "Mapilliary": './../Mapilliary/train.csv',
                      "MSS50":         './../MSS/data/50CS.csv',
                      "MSS100":         './../MSS/data/100CS.csv',
                      "MSS250":         './../MSS/data/250CS.csv',
                      "MSS750":         './../MSS/data/750CS.csv',
                      "GTAV":        './../data/GTAV/trainCS.csv',
                      "50CS": './../MSS/data/50CS.csv',
                      "100CS": './../MSS/data/100CS.csv',
                      "250CS": './../MSS/data/250CS.csv',
                      "750CS": './../MSS/data/750CS.csv',
                      "GTAVCS": './../GTAV/trainCS.csv'
                      }

complete_datasets = ["Kitti",  "Cityscapes","Mapilliary", "Synthia"]
real_datasets_train = [ "CityscapesCS","Dron""Kitti","Mapilliary"] 
synthetic_datasets = ["Helicoptero5",  "MSS", "Coche", "Bus", "Peaton", "Video", "Helicoptero", "MSS5", "Coche5", "Bus5", "Peaton5", "Video5" ] #"Synthia",
proportions = [.05, .15, .25, .5]#
