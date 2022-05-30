import os
import sys
import json
import math
import numpy as np
from numpy.linalg import inv

import numpy as np
import cv2 as cv
import glob


file_dir = os.path.abspath(__file__).rsplit('/',1)[0]

def compute_depth(K, invK, images, poses):
    
    print(f'num of images = {len(images)}')
    print(f'num of poses = {len(poses)}')

    ref = images[0]
    ref_pose_h = np.vstack([poses[0], np.array([0,0,0,1])]) #convert to homogeneous 4x4 matrix

    # pre-compute inverse of matrix
    print('---- compute projection matrix list')
    projections = []
    for pose in poses:
        inq_pose_h = np.vstack([pose, np.array([0,0,0,1])]) #convert to homogeneous 4x4 matrix
        projection = np.linalg.inv(inq_pose_h)
        print(f'projection = {projection}')
        projections.append(projection)
    print(f'length of inv_poses = {len(projections)}')

    height, width = ref.shape

    print(f'height = {height}')
    print(f'width = {width}')
    
    hw = 4 #height/128 # half window size of local kernel
    print(f'hw = {hw}')
    padded_ref = cv.copyMakeBorder(ref, hw, hw, hw, hw, cv.BORDER_CONSTANT, value=[0,0,0])

    # initialize final depth image
    depth = np.zeros((height, width, 1), np.float64)

    # define depth plane range 1 ~ 12 meters
    D_deck_list = []
    depth_list = np.arange(3, 12, 0.2)

    # initialize 
    for depth_index in depth_list: # around 300 depth is computed
        print(f'start to sweep depth at distance {depth_index} m')
        d = np.zeros((height, width, 1), np.float64)
        D_deck_list.append(d)

        for r in range(height):    # y
            #print(f'row = {r}')
            for c in range(width): # x

                # compute pixel (r,c)'s score at depth depth_index
                pts_ref_2d = np.array([[0, 0], [hw*2+1, 0], [hw*2+1, hw*2+1], [0, hw*2+1]])
                pts_ref = np.array([[c-hw, r-hw, 1], [c+hw, r-hw, 1], [c+hw, r+hw, 1], [c-hw, r+hw, 1]])
                pts_ref_3D = np.vstack([np.matmul(invK, np.transpose(pts_ref)) * depth_index, np.array([1,1,1,1])]) #convert to homogeneous 4xN points
                
                # convert to global coordinate and then project to specific image to compute locations
                pts_ref_3D_global = np.matmul(ref_pose_h, pts_ref_3D)
                
                ref_cropped_image = padded_ref[r:r+2*hw+1, c:c+2*hw+1]
                
                # cropped_height, cropped_width = ref_cropped_image.shape
                # cv.imshow('cropped input image', ref_cropped_image)
                # cv.waitKey(1000)
                # cv.destroyAllWindows()

                # to compute the 3D location of the 4 corners of the local window around pixel r,c
                scores = []
                for i in range(1,len(images)):
                    # print(f'    i = {i}')
                    inq = images[i]
                    projection = projections[i]
                    
                    pts_ref_3D_inq = np.matmul(projection, pts_ref_3D_global)
                    pts_inq = np.matmul(K, pts_ref_3D_inq[np.ix_([0,1,2],[0,1,2,3])])

                    # convert pts_inq into normalized z = 1 plane coordinates
                    # unrolling loop
                    pts_inq_2d = np.array([[pts_inq[0][0]/pts_inq[2][0], pts_inq[1][0]/pts_inq[2][0]], 
                        [pts_inq[0][1]/pts_inq[2][1], pts_inq[1][1]/pts_inq[2][1]], 
                        [pts_inq[0][2]/pts_inq[2][2], pts_inq[1][2]/pts_inq[2][2]], 
                        [pts_inq[0][3]/pts_inq[2][3], pts_inq[1][3]/pts_inq[2][3]]])
                    
                    #print(f'    pts_inq = {pts_inq_2d}')
                    #print(f'    pts_ref = {pts_ref_2d}')

                    h, status = cv.findHomography(pts_inq_2d, pts_ref_2d)
                    inq_cropped_image = cv.warpPerspective(inq, h, (ref_cropped_image.shape[1], ref_cropped_image.shape[0]))

                    # if (pts_inq_2d>0).all():
                    #     cv.imshow('cropped reference image', ref_cropped_image)
                    #     cv.imshow('cropped inqury image', inq_cropped_image)
                    #     cv.waitKey(1000)
                    #     cv.destroyAllWindows()

                    methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR','cv2.TM_CCORR_NORMED',     'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']
                    score = cv.matchTemplate(ref_cropped_image, inq_cropped_image, cv.TM_CCORR_NORMED)
                    scores.append(score)

                d[r][c] = np.average(scores)

        # cv.imshow('float',d)
        # cv.waitKey(0)
 
        # -------------------------------------------------------------
        # Saving the image
        # cv.imwrite(f'/Users/liguan/Downloads/research-scientist-3d-reconstruction/tmp/{depth_index:.4f}.png', cv.normalize(d, None, 0, 255, cv.NORM_MINMAX, cv.CV_8U))
        # print(f'saved /Users/liguan/Downloads/research-scientist-3d-reconstruction/tmp/{depth_index:.4f}.png')
        # D_deck_list.append(d) 

        # # save matrix into txt files
        # txt_file_name = f'/Users/liguan/Downloads/research-scientist-3d-reconstruction/tmp/{depth_index:.4f}.txt'
        # print(f'saved {txt_file_name}')
        # with open(txt_file_name , 'w') as f:
        #     for r in range(height):
        #         for c in range(width):
        #             f.write(f'{d[r][c][0]} ')
        #         f.write('\n')
            

        print(f'depth {depth_index} plane swept')

    # find the final depth from the depth volum D_deck_list 
    for r in range(height):
        for c in range(width):
            best_depth = 0
            best_score = -1
            ii = 0
            for depth_image, depth_value in zip(D_deck_list, depth_list):
                if best_score < depth_image[r][c]:
                    best_score = depth_image[r][c]
                    best_depth = depth_value
            depth[r][c] = best_depth

    return depth

if __name__ == '__main__':
    
    if len(sys.argv) < 1:
        print('not enough paratmers, espect format \n python3 batch_compute_depth_maps.py [path]')
        exit()

    data_dir = sys.argv[1]

    assert os.path.exists(data_dir), 'data folder does not exist'

    print(f'script_dir = {file_dir}')
    print(f'data_dir = {os.path.abspath(data_dir)}')

    # load intrinsics
    with open(f'{data_dir}/K.txt', 'r') as f:
        K = [[float(num) for num in line.split(' ')[:-1]] for line in f] # "-1" to skip the last spacing in the txt file

    # load images in the folders
    image_files = []

    for f in os.listdir(data_dir):
        if f.endswith(".jpg"):
            image_files.append(f)
    image_files.sort()

    # load image and pose
    # according to README
    # * 11 3072 x 2048 (W x H) JPG images (undistorted)
    # * Corresponding `.intrinsics` files containing the image dimensions and camera intrinsics as: `f, cx, cy, width, height`
    # * Corresponding `.pose` files containing the camera pose (i.e. camera-to-world transform) as a 3x4 matrix (`[R | c]`):
    # ```
    # r00 r01 r02 c0
    # r10 r11 r12 c1
    # r20 r21 r22 c2
    # ```

    image_list = [] # stores the actual image
    pose_list = []  # stores the corresponding camera pose

    scale = 0.125 # percent of original size

    K[0][0] *= scale
    K[1][1] *= scale
    K[0][2] *= scale
    K[1][2] *= scale

    for i in image_files:
        image_path = f'{data_dir}/{i}'
        print(image_path)
        img = cv.imread(image_path, cv.IMREAD_GRAYSCALE)

        w = int(img.shape[1] * scale)
        h = int(img.shape[0] * scale)
          
        # resize image
        resized = cv.resize(img, (w,h), interpolation = cv.INTER_AREA)
        image_list.append(resized)

        pose_path = f'{data_dir}/{os.path.splitext(i)[0]}.pose'
        print(pose_path)

        with open(pose_path, 'r') as f:
            pose = [[float(num) for num in line.split(' ')] for line in f]

        print(pose)
        pose_list.append(pose) 

    # check if all images are loaded correctly
    # using N neighboring frames below and N frames above (default N = 2)
    N = 2
    ref_id = 0
    for i in image_list:
        min_index = max(ref_id - N, 0)
        max_index = min(ref_id + N, len(image_list)-1)
        neighbor_list = range(min_index, max_index)
        print(f'neighbor_list = {neighbor_list}')

        # ref image as the first image
        # neighboring image follows
        tmp_img_list = []
        tmp_pose_list = []
        tmp_img_list.append(image_list[ref_id])
        tmp_pose_list.append(pose_list[ref_id])

        for tmp_i in neighbor_list:
            if (tmp_i == ref_id):
                continue

            tmp_img_list.append(image_list[tmp_i])
            tmp_pose_list.append(pose_list[tmp_i])

        # compute depth for ref image
        invK = np.linalg.inv(K)

        print(f'K = {K}')
        print(f'invK = {invK}')
        D = compute_depth(K, invK, tmp_img_list, tmp_pose_list)

        cv.imwrite(f'/Users/liguan/Downloads/research-scientist-3d-reconstruction/tmp_result/depth_{image_files[ref_id]}.png', cv.normalize(D, None, 0, 255, cv.NORM_MINMAX, cv.CV_8U))
        # print(f'saved /Users/liguan/Downloads/research-scientist-3d-reconstruction/tmp/{depth_index:.4f}.png')

        ref_id += 1
