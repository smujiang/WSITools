from PIL import Image
import glob
import os
# from operator import itemgetter
import numpy as np
# import pyvips
import subprocess
import re
import openslide
import multiprocessing
import cv2
import skimage

# save image patches back into a big tiff file
# file name should fellow this pattern uuid_x_y.jpg or uuid_x_y.png
# otherwise, you have to rewrite the function


def hann_2d_win(shape=(256, 256)):
    def hann2d(i, j):
        i_val = 1 - np.cos((2 * math.pi * i) / (shape[0] - 1))
        j_val = 1 - np.cos((2 * math.pi * j) / (shape[1] - 1))
        hanning = (i_val * j_val) 
        return hanning

    hann2d_win = np.fromfunction(hann2d, shape)
    return hann2d_win


class SubPatches2BigTiff:
    # out = pyvips.Image.black(100, 100, bands=3) + 255
    def __init__(self, patch_dir, save_to, ext=".jpg", down_scale=8, patch_size=(256, 256), xy_step=(128, 128)):
        self.patch_dir = patch_dir
        self.save_to = save_to
        self.ext = ext
        self.patch_size = patch_size
        self.xy_step = xy_step
        self.filenames = sorted(glob.glob(patch_dir + "/*" + ext))
        w, h, self.x_min, self.y_min = self.calculate_tiff_w_h()
        self.down_scale = down_scale
        print("Image W:%d/H:%d" % (int(w/self.down_scale), int(h/self.down_scale)))
        self.filter = hann_2d_win((int(self.patch_size[0]/self.down_scale), int(self.patch_size[1]/self.down_scale))
        self.out_arr = (np.zeros((int(h/self.down_scale), int(w/self.down_scale), 3))+255).astype(np.uint8)
        # TODO: save mode = "ABA" or "ABB" or "single"
        # TODO: if list else if directory

    @staticmethod
    def shell_cmd(cmd):
        cmd = re.sub('\s+', ' ', cmd).strip()
        cmd = cmd.split(" ")
        m = subprocess.Popen(cmd, stdout=subprocess.PIPE)
        stdout, stderr = m.communicate()
        exitCode = m.returncode
        return exitCode

    def calculate_tiff_w_h(self):
        locations = []
        for f in self.filenames:
            fn = os.path.split(f)[1]
            p = fn.split("_")
            locations.append([int(p[1]), int(p[2])])
        patch_locs = np.array(locations)
        x_min = min(patch_locs[:, 0])
        x_max = max(patch_locs[:, 0])
        y_min = min(patch_locs[:, 1])
        y_max = max(patch_locs[:, 1])
        w = x_max - x_min + self.patch_size[0]
        h = y_max - y_min + self.patch_size[1]
        row_cnt = int(h/self.xy_step[1])
        column_cnt = int(w/self.xy_step[0])
        # print(row_cnt)
        # print(column_cnt)
        return w, h, x_min, y_min

    def insert_patch(self, f):
        x_r = int(self.patch_size[0] / self.down_scale)
        y_r = int(self.patch_size[1] / self.down_scale)
        img = cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB)
        sub_arr = cv2.resize(img, dsize=(x_r, y_r), interpolation=cv2.INTER_CUBIC)
        fn = os.path.split(f)[1]
        p = fn.split("_")
        x = int(p[1])
        y = int(p[2])
        x_loc = int((x-self.x_min)/self.down_scale)
        y_loc = int((y-self.y_min)/self.down_scale)
        self.out_arr[y_loc:y_loc+y_r, x_loc:x_loc+x_r, :] += sub_arr * self.filter[:,:,None]*(self.xy_step[0]/self.patch_size[0])**2
        self.filter = hann_2d_win((512 // down_scale, 512 // down_scale))


    def parallel_save(self):    # example: save("big.tiff")
        num_processors = 5
        multiprocessing.set_start_method('spawn')
        pool = multiprocessing.Pool(processes=num_processors)
        pool.map(self.insert_patch, self.filenames)

    def save(self):
        cnt = 0
        print("Insert %d images patches" % len(self.filenames))
        for f in self.filenames:
            self.insert_patch(f)
            cnt += 1
            if cnt % 2000 == 0:
                print("Insert %d/%d images patches" % (cnt, len(self.filenames)))
            # if cnt == 4000:
            #     break
        output_dir = os.path.split(self.save_to)[0]
        temp_fn = os.path.join(output_dir, "temp_downsampled.tiff")
        Image.fromarray(self.out_arr, 'RGB').save(temp_fn)

        print('create tiff file pyramid')
        cmd = "vips tiffsave " + temp_fn + " " + self.save_to + ' --compression=deflate --Q=100 --tile ' \
              '--tile-width=' + str(self.patch_size[0]) + ' --tile-height=' + str(self.patch_size[1]) + ' --pyramid'
        print(cmd)
        returned_output = self.shell_cmd(cmd)

    # cd /projects/shart/digital_pathology/data/test/tiff_out
    # cmd = "vips tiffsave ./temp_downsampled.tiff ./176c3d0c7c6e4807a67d7fc622f1444a.tiff --compression=deflate --Q=100 --tile --tile-width=256 --tile-height=256 --pyramid'


    #
    # def get_patch_identifiers_from_filenames(self):
    #     list_val = []
    #     for f in self.filenames:
    #         print(f)
    #         fn = os.path.split(f)[1]
    #         p = fn.split("_")
    #         list_val.append([p[0], int(p[1]), int(p[2])])
    #     locations = [(i[1], i[2]) for i in list_val]
    #     case_uuid = set([i[0] for i in list_val])
    #     if len(case_uuid) > 1:
    #         raise Exception("Image patches in the folder not belong to the same case.")
    #     elif len(case_uuid) == 0:
    #         raise Exception("No case uuid found.")
    #     return case_uuid.pop(), np.array(locations)
    #
    # def complement_missing_patch(self, case_id, patch_locs):
    #     print("Complementing missing patches with white ones")
    #     # x_range = range(min(patch_locs[:, 0]), max(patch_locs[:, 0]), self.xy_step[0])
    #     # y_range = range(min(patch_locs[:, 1]), max(patch_locs[:, 1]), self.xy_step[1])
    #     x_range = range(min(patch_locs[:, 0]), 41600, self.xy_step[0])
    #     y_range = range(min(patch_locs[:, 1]), 31872, self.xy_step[1])
    #     sorted_fn_list = []
    #     temp_fn_list = []
    #
    #     for w in x_range:
    #         for h in y_range:
    #             patch_identifier = case_id + "_" + str(w) + "_" + str(h) + "_"
    #             for s in self.filenames:
    #                 if os.path.split(s)[1].startswith(patch_identifier):
    #                     sorted_fn_list.append(s)
    #                     self.filenames.remove(s)
    #             # Found = False
    #             # for f in self.filenames:
    #             #     if patch_identifier in f:
    #             #         sorted_fn_list.append(f)
    #             #         Found = True
    #             # if not Found:
    #             #     fpn = os.path.join(self.patch_dir, patch_identifier + self.ext)
    #             #     temp_fn_list.append(fpn)
    #             #     sorted_fn_list.append(fpn)
    #             #     # if ext == ".jpg":
    #             #     #     arr_size = [self.patch_size[0], self.patch_size[1], 3]
    #             #     # elif ext == ".npg":
    #             #     #     arr_size = [self.patch_size[0], self.patch_size[1], 4]
    #             #     # else:
    #             #     #     raise Exception("Not a supported image format")
    #             #     # img_arr = np.zeros(arr_size, dtype=np.uint8) + 255
    #             #     # Image.fromarray(img_arr).save(fpn)
    #     unique_x = len(x_range)
    #     return sorted_fn_list, temp_fn_list, unique_x
    #

    #
    # # @staticmethod
    # # def remove_files_list(temp_fn_list):
    # #     for i in temp_fn_list:
    # #         os.remove(i)
    #
    # # @staticmethod
    # # def remove_short_fn(patch_dir, ext, threshold):
    # #     filenames = glob.glob(patch_dir + "/*" + ext)
    # #     for f in filenames:
    # #         if len(f) < threshold:
    # #             os.remove(f)
    #
    # def save(self):    # example: save("big.tiff")
    #
    #     out = pyvips.Image.black(94208, 83968, bands=3) + 255
    #     cnt = 0
    #     for f in self.filenames:
    #         sub = pyvips.Image.jpegload(f)
    #         fn = os.path.split(f)[1]
    #         p = fn.split("_")
    #         x = int(p[1])
    #         y = int(p[2])
    #         out = out.insert(sub, x, y, expand=False)
    #         cnt += 1
    #         if cnt % 2000 == 0:
    #             print("Progress: %d/%d" % (cnt, len(self.filenames)))
    #     output_dir = os.path.split(self.save_to)[0]
    #     temp_fn = os.path.join(output_dir, "temp.tiff")
    #     out.write_to_file(temp_fn)
    #
    #     '''create tiff file pyramid'''
    #     cmd = "vips tiffsave " + temp_fn + " " + self.save_to + ' --compression=deflate --Q=100 --tile ' \
    #           '--tile-width=' + str(self.patch_size[0]) + ' --tile-height=' + str(self.patch_size[1]) + ' --pyramid'
    #     print(cmd)
    #     returned_output = self.shell_cmd(cmd)

        # case_id, patch_locations = self.get_patch_identifiers_from_filenames()
        # sorted_fn_list, temp_fn_list, unique_x = self.complement_missing_patch(case_id, patch_locations)
        # images = [pyvips.Image.new_from_file(filename, access="sequential") for filename in sorted_fn_list]
        # final = pyvips.Image.arrayjoin(images, across=unique_x)
        # output_dir = os.path.split(self.save_to)[0]
        # temp_fn = os.path.join(output_dir, "temp.tiff")
        # final.write_to_file(temp_fn)
        #
        # '''create tiff file pyramid'''
        # cmd = "vips tiffsave " + temp_fn + self.save_to + '--compression=deflate --Q=100 --tile ' \
        #       '--tile-width=' + str(self.patch_size[0]) + ' --tile-height=' + str(self.patch_size[1]) + ' --pyramid'
        # print(cmd)
        # returned_output = self.shell_cmd(cmd)
        #
        # print("Removing temporary files")
        # cmd = "rm " + temp_fn
        # print(cmd)
        # returned_output = self.shell_cmd(cmd)
        #
        # self.remove_files_list(temp_fn_list)

    def get_thumbnail(self, thumbnail_fn):
        obj = openslide.open_slide(self.save_to)
        print("WSI loaded")
        thumbnail = obj.get_thumbnail(size=(1024, 1024)).convert("RGB")
        thumbnail.save(thumbnail_fn)


if __name__ == "__main__":
    # patch_dir = "/projects/shart/digital_pathology/data/test/09ccd9c864194bb2990088348369f291"
    # wsi_fn = "/projects/shart/digital_pathology/data/PenMarking/WSIs/MELF/09ccd9c864194bb2990088348369f291.tiff"
    # patch_dir = "/projects/shart/digital_pathology/data/PenMarking/WSIs_Img_pairs_256/176c3d0c7c6e4807a67d7fc622f1444a"
    # save_to = "/projects/shart/digital_pathology/data/test/tiff_out/176c3d0c7c6e4807a67d7fc622f1444a.tiff"
    # w, h = openslide.open_slide(wsi_fn).dimensions
    # thumb_fn = "/projects/shart/digital_pathology/data/test/tiff_out/thumb_176c3d0c7c6e4807a67d7fc622f1444a.jpg"

    # patch_dir = "/projects/shart/digital_pathology/data/PenMarking/eval/pixel2pixel_256/images_dispatch/7470963d479b4576bc8768b389b1882e"
    # save_to = "/projects/shart/digital_pathology/data/PenMarking/eval/pixel2pixel_256/images_dispatch/output_7470963d479b4576bc8768b389b1882e.tiff"
    # thumb_fn = "/projects/shart/digital_pathology/data/PenMarking/eval/pixel2pixel_256/images_dispatch/thumb_7470963d479b4576bc8768b389b1882e.jpg"

    patch_dir = "/projects/shart/digital_pathology/data/PenMarking/eval/pixel2pixel_256/images_dispatch/d83cc7d1c941438e93786fc381ab5bb5"
    save_to = "/projects/shart/digital_pathology/data/PenMarking/eval/pixel2pixel_256/images_dispatch/output_d83cc7d1c941438e93786fc381ab5bb5.tiff"
    thumb_fn = "/projects/shart/digital_pathology/data/PenMarking/eval/pixel2pixel_256/images_dispatch/thumb_d83cc7d1c941438e93786fc381ab5bb5.jpg"

    patch_size = (256, 256)
    xy_step = (128, 128)
    ext = ".jpg"
    # sv_wsi = SubPatches2BigTiff(patch_dir, save_to, ".jpg", 8, patch_size, xy_step)
    # sv_wsi = SubPatches2BigTiff(patch_dir, save_to, "inputs.png", 16, patch_size, xy_step)
    sv_wsi = SubPatches2BigTiff(patch_dir, save_to, "outputs.png", 1, patch_size, xy_step)
    # sv_wsi = SubPatches2BigTiff(patch_dir, save_to, "outputs.png", 16, patch_size, xy_step)

    # sv_wsi.parallel_save()
    sv_wsi.save()
    print("Saving thumbnail")
    # sv_wsi.get_thumbnail(thumb_fn)


    # def remove_short_fn(patch_dir, ext, threshold):
    #     filenames = glob.glob(patch_dir + "/*" + ext)
    #     for f in filenames:
    #         fn = os.path.split(f)[1]
    #         if len(fn) < threshold:
    #             os.remove(f)
    #
    # remove_short_fn(patch_dir, ext, 80)






