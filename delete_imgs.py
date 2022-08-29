import cv2
import os
from multiprocessing import Pool
import shutil

scence_dict = {
    'S01' :{
        'path':'train',
        'cams':('c001','c002','c003','c004','c005')
    },
    'S02' :{
        'path':'validation',
        'cams':('c006','c007','c008','c009')
    },
    'S03' :{
        'path':'train',
        'cams':('c010','c011','c012','c013','c014','c015')
    },
    'S04' :{
        'path':'train',
        'cams':('c016','c017','c018','c019','c020','c021','c022','c023','c024','c025',
                'c026','c027','c028','c029','c030','c031','c032','c033','c034','c035',
                'c036','c037','c038','c039','c040')
    },
    'S05':{
        'path':'validation',
        'cams': ('c010','c016','c017','c018','c019','c020','c021','c022','c023',
                'c024','c025','c026','c027','c028','c029','c033','c034','c035','c036'),
    },
    'S06' :{
        'path':'test',
        'cams': ('c041','c042','c043','c044','c045','c046')
    }
}

def main():
    for scence_id,cam_megs in scence_dict.items():
        cam_ids = cam_megs['cams']
        cam_path = cam_megs['path']
        for cam_id in cam_ids:
            print("scence_id:%s,cam_id:%s" %(scence_id,cam_id))
            save_path = os.path.join('mcmt/dataset',cam_path,scence_id,cam_id,'imgs')
            if os.path.exists(save_path):
                shutil.rmtree(save_path)
                
if __name__ == '__main__':
    main()