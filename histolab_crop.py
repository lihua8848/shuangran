import openslide
from histolab.slide import Slide
from histolab.tiler import GridTiler
from shapely.geometry import Polygon, MultiPolygon
import matplotlib.pyplot as plt

import openslide
import xml.etree.ElementTree as ET
from PIL import Image
import numpy as np
import os

from shapely.validation import explain_validity

def patchify(ndpa_file, ndpi_file, save_dir):
    '''ndpa_file: .ndpa文件, ndpi_file: .ndpi文件'''
    # 解析.ndpa文件
    tree = ET.parse(ndpa_file)
    root = tree.getroot()
    # 打开.ndpi文件
    slide = openslide.open_slide(ndpi_file)
    file_name = ndpi_file.split('/')[-1].split('.')[0]
    # 列出每个level的下采样倍数,输出
    slide_downsamples = slide.level_downsamples
    # print(slide_downsamples)
    ppm_x = 1 / float(slide.properties[openslide.PROPERTY_NAME_MPP_X])
    ppm_y = 1 / float(slide.properties[openslide.PROPERTY_NAME_MPP_Y])

    slide_width, slide_height = slide.level_dimensions[0]
    img_center_x = (slide_width / 2) * 1000 / ppm_x
    img_center_y = (slide_height / 2) * 1000 / ppm_y

    # Get the properties from the OpenSlide object
    properties = slide.properties

    # Get the offsets from the image center to the slide center
    offset_from_image_center_x = float(properties['hamamatsu.XOffsetFromSlideCentre'])
    offset_from_image_center_y = float(properties['hamamatsu.YOffsetFromSlideCentre'])

    # Compute the offsets from the top left to the image center
    offset_from_top_left_x = img_center_x - offset_from_image_center_x
    offset_from_top_left_y = img_center_y - offset_from_image_center_y

    # The offsets from the top left to the slide center are the references for the X and Y coordinates
    x_reference = offset_from_top_left_x
    y_reference = offset_from_top_left_y
    os.makedirs(save_dir, exist_ok=True)
    file_dir = os.path.join(save_dir, file_name)
    os.makedirs(file_dir, exist_ok=True)

    # 提取标注的坐标
    coordinates = []
    multipolygon = []
    x_coords_list = []
    y_coords_list = []
    num_patch = 0
    for annotation in root.findall('.//ndpviewstate'):
        for pointlist in annotation.findall('.//pointlist'):
            points = []
            for point in pointlist.findall('point'):
                x, y = (int(point.find('x').text) / 1000 * ppm_x + x_reference / 1000 * ppm_x, int(point.find('y').text) / 1000 * ppm_y + y_reference / 1000 * ppm_y \
                    ) 
                points.append((int(x), int(y)))
            coordinates.append(points)
            x_coords = range(min(x[0] for x in points), max(x[0] for x in points), patch_size)
            y_coords = range(min(x[1] for x in points), max(x[1] for x in points), patch_size)
            x_coords_list.append(x_coords)
            y_coords_list.append(y_coords)

            roi_polygon = Polygon(points).buffer(0)
            multipolygon.append(roi_polygon)
            #获取Polygon的x和y坐标
            # #使用matplotlib可视化

            # fig, ax = plt.subplots()

            # if roi_polygon.geom_type == 'Polygon':
            #     x,y = roi_polygon.exterior.xy
            #     ax.plot(x, y)
            # elif roi_polygon.geom_type == 'MultiPolygon':
            #     for geom in roi_polygon.geoms:
            #         x, y = geom.exterior.xy
            #         ax.plot(x, y)
            #         ax.set_aspect('equal', 'datalim')
            #         plt.show()
            # ax.set_aspect('equal', 'datalim')
            # plt.show()
    multipolygon_all = MultiPolygon(multipolygon)
    # 使用matplotlib进行可视化
    # fig, ax = plt.subplots()

    # for polygon in multipolygon_all.geoms:
    #     x, y = polygon.exterior.xy
    #     ax.plot(x, y)
    # #坐标轴反转
    # ax.invert_yaxis()
    # plt.show()


            # 遍历每个网格单元
    for i, x_coords, y_coords in zip(range(len(x_coords_list)), x_coords_list, y_coords_list):
        for j, x_point in enumerate(x_coords):
            for k, y_point in enumerate(y_coords):
                tile_polygon = Polygon([(x_point, y_point), (x_point + patch_size, y_point), (x_point, y_point + patch_size), (x_point + patch_size, y_point + patch_size)])
                # 检查网格单元是否完全在标注区域内
                # if multipolygon_all.contains(tile_polygon):
                # if multipolygon_all.intersects(tile_polygon):
                #     fig, ax = plt.subplots()
                #     x, y = tile_polygon.exterior.xy
                #     ax.plot(x, y)
                #     ax.set_aspect('equal', 'datalim')
                #     plt.show()
                #     try:
                #         if multipolygon_all.intersection(tile_polygon).area / tile_polygon.area > 0.3:

                #             # 如果是，生成一个切片，并将其添加到列表中
                #             tile = slide.read_region((x_point, y_point), 0, (patch_size, patch_size))
                #             tile.save(save_dir + f'/patch_{i}_{j}_{k}.png')
                #     except:
                #             #保存文件名到txt
                # if multipolygon_all.is_valid and tile_polygon.is_valid:
                #     with open(save_dir + '/wsi_invalid_name.txt', 'a') as f:
                #         f.writelines(f'{ndpi_file}\n')
                if multipolygon_all.intersects(tile_polygon):
                    tile = slide.read_region((x_point, y_point), 0, (patch_size, patch_size))
                    tile.save(file_dir + f'/patch_{i}_{j}_{k}.png')
                    num_patch += 1
    return num_patch

if __name__ == "__main__":
    root_dir = '/mnt/d/data/shuangran'
    save_root_dir = "/home/omnisky/workspace/shuangran/data"
    abs_dir = []
    save_abs_dirs = []
    sub_dirs = os.listdir(root_dir)
    for sub_dir in sub_dirs:
        abs_dir.append(os.path.join(root_dir, sub_dir))
        save_abs_dirs.append(os.path.join(save_root_dir, sub_dir))
    
    # 创建文件夹来保存patch

    patch_size = 256
    num = 0 
    num_patch = 0
    for i, dir in enumerate(abs_dir):
        ndpa_files = []
        ndpi_files = []
        files = os.listdir(dir)
        for file in files:
            if file.endswith('.ndpi'):
                ndpi_file = os.path.join(dir, file)
                ndpi_files.append(ndpi_file)
            if file.endswith('.ndpa'):
                ndpa_file = os.path.join(dir, file)
                ndpa_files.append(ndpa_file)
        for j_a, k_i in zip(ndpa_files, ndpi_files):
            save_sub_abs_dir = save_abs_dirs[i]
            
            
            try:
                num_patch = patchify(j_a, k_i, save_sub_abs_dir)
                num += 1
                num_patch += num_patch
                print(f'处理 {k_i} 切片成功，已处理 {num} 个切片, 大小为 {patch_size}, 共生成 {num_patch} 个patch')
            except:
                print(f'{k_i} error')
                with open(save_sub_abs_dir + '/wsi_error_name.txt', 'a') as f:
                    f.writelines(f'{k_i}\n')
                





