            # -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 18:20:25 2019

@author: hermioneX
"""

import cv2
import numpy as np
import os
from sklearn import metrics
import glob 
import re
from scipy.ndimage import map_coordinates
# import wradlib.ipol as ipol
import skimage.transform as sktf
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
from toolsx import *

class TyphoonFlow(object):
    def __init__(self, src_path, out_path, fn_head, channel, 
                    pre_mode = 'all', start_ind = -1, pre_end = 37, pre_step = 6):
        self.src_path = src_path 
        self.fn_head = fn_head   # U/V/W/X/Y/Z
        self.ch = channel
        self.pre_mode = pre_mode
        self.out_path = out_path

        # self.data_range = {8:[2700,4010],
        #                    9:[1800,3990],
        #                    10:[2450,3780]} #8,9,10 三个通道的数据范围
        self.data_range = {8:[2700,4010],
                           9:[1250,3990],
                           10:[2450,3780]} #8,9,10 三个通道的数据范围
        self.fn_mid1 = '_Hour_'
        self.fn_mid2 = '_Band_'

        self.w = 1999
        self.h = 1999
        self.y_coords, self.x_coords = np.mgrid[0:self.h, 0:self.w]
        self.coords = np.float32(np.dstack([self.x_coords, self.y_coords]))
        self.marg = 200

        self.flow_params = {'flow' : None, 
                            'pyr_scale' : 0.5, 
                            'levels' : 3, 
                            'winsize' : 15, 
                            'iterations' : 3, 
                            'poly_n' : 5, 
                            'poly_sigma' : 1.1,
                            'flags' : 0
                           }
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)) 

        if self.pre_mode != 'test':
            self.start_ind = self._search_endid()
            print('Get last idx>>>>>>>>>>>>>>>>>>>>> ',self.start_ind)
        else:
            self.start_ind = start_ind

        self.pre_end = pre_end
        self.pre_step = pre_step

        self.now_flag, self.now_frame = self._get_one_frame(self.start_ind)
        if self.now_flag:
            self.cut_now = self.now_frame[self.marg:-self.marg,self.marg:-self.marg]
            self.acu_range = [np.min(self.cut_now), np.max(self.cut_now)]
        self.prev_flag1, self.prev_frame1 = self._get_one_frame(self.start_ind - 1)
        self.prev_flag2, self.prev_frame2 = self._get_one_frame(self.start_ind - 2)

    def _norm8bit(self, data):
        data = (data - self.data_range[self.ch][0])*255.0/(self.data_range[self.ch][1] - self.data_range[self.ch][0])
        data[data > 255] = 255
        data[data < 0] = 0
        return data 

    def _recover(self, data):
        data = data*(self.data_range[self.ch][1] - self.data_range[self.ch][0])/255.0 + self.data_range[self.ch][0]
        return np.array(data,np.uint16)

    def _norm_acu(self,frame):
        frame[frame > self.acu_range[1]] = np.mean(frame[self.marg:-self.marg,self.marg:-self.marg])
        frame[frame < self.acu_range[0]] = np.mean(frame[self.marg:-self.marg,self.marg:-self.marg])
        return frame

    def _get_one_frame(self, ind):
        cnt_fn = self.src_path + self.fn_head + self.fn_mid1 + str(ind) + self.fn_mid2 + str(self.ch).zfill(2) +'.npy'
        # print('Reading:::::::',cnt_fn)
        if os.path.exists(cnt_fn):
            data = np.load(cnt_fn)
            data = self._norm8bit(data)
            return 1, data
        else:
            return 0, 0

    def _optical_flow(self, frame_pre, n):
        flow = cv2.calcOpticalFlowFarneback(self.now_frame, frame_pre, **self.flow_params)
        
        flow2 = flow.copy()
        for i in range(2):
            flow2[:,:,i] = cv2.dilate(flow[:,:,i], cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (23, 23)))
            flow2[:,:,i] = cv2.medianBlur(flow2[:,:,i].copy(),5)

        pm = self.coords + flow2*n
        frame_rst = cv2.remap(self.now_frame, pm, None, cv2.INTER_CUBIC)
        return frame_rst

    def _search_endid(self,):
        '''
        找到数据集小时ID的最大值
        '''
        band1name = self.fn_head + '_Hour_*_Band_'+ str(self.ch).zfill(2) +'.npy'
        band1files = glob.glob(os.path.join(self.src_path, band1name))
        tids = []
        for i in range(len(band1files)):
            tids.append(int(re.findall(r'Hour_(\d+)_Band', band1files[i])[0]))
        return max(tids) 

    def process_LK(self, sparse_flag = 'sd', f_tail = '_lksd'):

        if sparse_flag == 'sd':
            input_data = [self.prev_frame2, self.prev_frame1, self.now_frame]
            # print(np.array(input_data,np.uint8).shape)
            pts_source, pts_target_container = sparse_sd(np.array(input_data,np.uint8), lead_steps = self.pre_end + 1)
        else:
            pts_source, pts_target_container = sparse_linear(np.array(input_data,np.uint8), lead_steps = self.pre_end + 1)

        trf = sktf.AffineTransform()

        nowcasts = []
        print(self.pre_end+1)
        for lead_step in range(self.pre_step, self.pre_end+1, self.pre_step):

            print('Predict:::::',lead_step,' hours later with lk')

            pts_target = pts_target_container[lead_step]
            # estimate transformation matrix
            # based on source and traget points
            trf.estimate(pts_source, pts_target)

            # make a nowcast
            nowcst_frame = sktf.warp(input_data[-1]/255, trf.inverse)
            nowcst_frame = nowcst_frame*255.0

            # out_pic = self.out_path + self.fn_head + self.fn_mid1 + str(self.start_ind + lead_step) + \
            #         self.fn_mid2 + str(self.ch).zfill(2) + f_tail + '.jpg'
            # cv2.imwrite(out_pic, nowcst_frame)

            nowcasts.append(nowcst_frame)

        return np.array(nowcasts)

    def process_dense_linear(self, f_tail = '_dl'):
        nowcasts = []
        for i in np.arange(self.pre_step, self.pre_end+1, self.pre_step):
            print('Predict:::::',i,' hours later with dense_linear')
            if (self.prev_flag1 and self.prev_flag2):
                print('Merging::::::: 3')
                new_frame1 = self._optical_flow(self.prev_frame1, i)
                new_frame2 = self._optical_flow(self.prev_frame2, i/2.0)
                new_frame6 = self._optical_flow(self.prev_frame2, i/6.0)
                # new_frame = self._optical_flow(prev_frame, 1)
                
                out_frame = new_frame1*0.5 + new_frame2*0.3 + new_frame6 * 0.2

                # out_pic = self.out_path + self.fn_head + self.fn_mid1 + str(self.start_ind + i) + self.fn_mid2 + str(self.ch).zfill(2) +'_merge.jpg'
                # cv2.imwrite(out_pic, out_frame)

                nowcasts.append(out_frame)    
            else:
                print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                print('!!!!!!!!!!!! dense_linear failed !!!!!!!!!!!!!!!')
                print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')        
        return np.array(nowcasts)

    def process_main(self, ):

        if (self.prev_flag1):
            if self.pre_mode == 'ls':
                nowcast = self.process_LK(sparse_flag = 'sd')
            elif self.pre_mode == 'll':
                nowcast = self.process_LK(sparse_flag = 'linear')
            elif self.pre_mode == 'ds':
                nowcast = self.process_dense_semi()
            elif self.pre_mode == 'dl':
                nowcast = self.process_dense_linear()
            elif self.pre_mode == 'all':
                # nowcast = self.process_dense_linear()*0.7 + self.process_dense_semi()* 0.3
                tmp_dl = self.process_dense_linear()
                tmp_ds = self.process_dense_semi()
                tmp_sd = self.process_LK(sparse_flag = 'sd')
                nowcast = tmp_dl.copy()
                for i in range(len(tmp_dl)):
                    nowcast[i] = tmp_dl[i] * (1.0 - i/len(tmp_dl)) + tmp_ds[i] * i/len(tmp_dl)
                
                nowcast[:3] = np.mean((nowcast[:3], tmp_sd[:3]),0)

            # nowcast = np.array(nowcast,np.uint8)
            nowcast_data = self._recover(nowcast.copy())
            print('########################',nowcast_data.shape)

            for i in range(len(nowcast_data)):
                if i >=30:
                    return 1
                # out_npy = self.out_path + self.fn_head + self.fn_mid1 + str(self.start_ind + 6*(i+1)) \
                #         + self.fn_mid2 + str(self.ch).zfill(2) + '_'+ self.pre_mode + '.npy'
                out_npy = self.out_path + self.fn_head + self.fn_mid1 + str(self.start_ind + 6*(i+1)) \
                        + self.fn_mid2 + str(self.ch).zfill(2) + '.npy'

                out_frame = nowcast_data[i].copy()
                out_frame = cv2.medianBlur(out_frame,5)
                out_frame = cv2.dilate(out_frame, self.kernel)
                np.save(out_npy,out_frame)

                # out_pic = self.out_path + self.fn_head + self.fn_mid1 + str(self.start_ind + 6*(i+1)) \
                #         + self.fn_mid2 + str(self.ch).zfill(2) + '_'+ self.pre_mode + '_m2.jpg'
                # cv2.imwrite(out_pic, self._norm8bit(out_frame))
        else:
            print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
            print('!!!!!!!!!!!! Could not prediction !!!!!!!!!!!!!!')
            print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')

if __name__ == '__main__':

    src_path  = './data/'

    # # ######## test ######################
    # out_path = 'D:\\workData\\2019.7.30_typhoon_tianchi\\calcOpticalFlowFarneback\\out_data\\'
    # for ch in [8]:
    #     for fh in ['U']:
    #         rst = TyphoonFlow(src_path = src_path, out_path = out_path, fn_head = fh, channel = ch, 
    #                           pre_mode = 'all', start_ind = -1, pre_end = 37, pre_step = 6)
    #         nowcasts = rst.process_main()

    ###### output ######################        
    out_path = './out_data/'
    for ch in [8,9,10]:
        for fh in ['U','V','W','X','Y','Z']:
            rst = TyphoonFlow(src_path = src_path, out_path = out_path, fn_head = fh, channel = ch, 
                              pre_mode = 'all', start_ind = -1, pre_end = 37, pre_step = 6)
            nowcasts = rst.process_main()

