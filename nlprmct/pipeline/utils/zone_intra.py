import os
import cv2
import numpy as np
from os.path import join as opj

class zone():
    def __init__(self):
        # 0: b 1: g 3: r 123:w
        # w r 非高速
        # b g 高速
        zones = {}
        zone_path = "../zone"
        for img_name in os.listdir(zone_path):
            camnum = int(img_name.split('.')[0][-3:])
            zone_img = cv2.imread(opj(zone_path, img_name))
            zones[camnum] = zone_img

        self.zones = zones
        self.current_cam = 0

    def set_cam(self,cam):
        self.current_cam = cam
        
    def get_zone(self,bbox):
        cx = int((bbox[0] + bbox[2]) / 2)
        cy = int((bbox[1] + bbox[3]) / 2)
        pix = self.zones[self.current_cam][cy, cx, :]
        zone_num = 0
        if pix[0] > 50 and pix[1] > 50 and pix[2] > 50:  # w
            zone_num = 1
        if pix[0] < 50 and pix[1] < 50 and pix[2] > 50:  # r
            zone_num = 2
        if pix[0] < 50 and pix[1] > 50 and pix[2] < 50:  # g
            zone_num = 3
        if pix[0] > 50 and pix[1] < 50 and pix[2] < 50:  # b
            zone_num = 4
        return zone_num

    def is_ignore(self,zone_list,frame_list, cid):
        # 0 不在任何路口 1 白色 2 红色 3 绿色 4 蓝色
        zs, ze = zone_list[0], zone_list[-1]
        fs, fe = frame_list[0],frame_list[-1]
        if zs == ze:
            # 如果一直在一个区域里，排除
            if ze in [1,2]:
                return 2
            if zs!=0 and 0 in zone_list:
                return 0
            if fe-fs>1500:
                return 2
            if fs<2:
                if cid in [45]:
                    if ze in [3,4]:
                        return 1
                    else:
                        return 2
            if fe > 1999:
                if cid in [41]:
                    if ze not in [3]:
                        return 2
                    else:
                        return 0
            if fs<2 or fe>1999:
              if ze in [3,4]:
                return 0
            if ze in [3,4]:
                return 1
            return 2
        else:
            # 如果区域发生变化
            if cid in [41, 42, 43, 44, 45, 46]:
                # 如果从支路进支路出，排除
                if zs == 1 and ze == 2:
                    return 2
                if zs == 2 and ze == 1:
                    return 2
            if cid in [41]:
                # 在41相机，车辆没有进出42相机
                if (zs in [1, 2]) and ze == 4:
                    return 2
                if zs == 4 and (ze in [1, 2]):
                    return 2
            if cid in [46]:
                # 在46相机，车辆没有进出45相机
                if (zs in [1, 2]) and ze == 3:
                    return 2
                if zs == 3 and (ze in [1, 2]):
                    return 2
            return 0

    def filter_mot(self,mot_list, cid):
        new_mot_list = dict()
        sub_mot_list = dict()
        for tracklet in mot_list:
            # if cid == 45 and tracklet==207:
            #     print(tracklet)
            tracklet_dict = mot_list[tracklet]
            frame_list = list(tracklet_dict.keys())
            frame_list.sort()
            zone_list = []
            for f in frame_list:
                zone_list.append(tracklet_dict[f]['zone'])
            if self.is_ignore(zone_list,frame_list, cid)==0:
                new_mot_list[tracklet] = tracklet_dict
            if self.is_ignore(zone_list,frame_list, cid)==1:
                sub_mot_list[tracklet] = tracklet_dict
        return new_mot_list

