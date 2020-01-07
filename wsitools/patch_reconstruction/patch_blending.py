import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
from itertools import product


def parse_filename(f_name, ext=".png"):
    trimed_line = f_name[0:-len(ext)]
    temp_ele = trimed_line.split("-")
    label = temp_ele[1]
    f_type = temp_ele[2]
    ele = temp_ele[0].split("_")
    return ele, label, f_type


# directly stitch image patches together to get a larger ROI
def direct_reconstruct(patches, step, channel, region_size):
    i_h, i_w = region_size[:2]
    p_h, p_w = patches.shape[2:4]
    img = np.zeros([i_h, i_w, channel]).astype(np.float)
    img_cnt = np.zeros([i_h, i_w, channel]).astype(np.float)
    n_h = i_h - p_h + 1       # compute the dimensions of the patches array
    n_w = i_w - p_w + 1
    for (i, j) in product(range(0, n_h, step), range(0, n_w, step)):
        img[i:i + p_h, j:j + p_w] += patches[int(i/step), int(j/step)]
        img_cnt[i:i + p_h, j:j + p_w, :] += np.ones([p_h, p_w, channel])
    rec_img = np.array(img / img_cnt).astype(np.uint8)
    return rec_img


# generate a smooth gradual change mask for image blending
def get_blending_mask(x_size=128, y_size=256, type="H", channel=3):  # type= "H" or "V"
    if type == "H":
        row = np.arange(0, 1, 1 / x_size, np.float).reshape(1, -1)
        mask = np.repeat(row, y_size, axis=0)
    else:
        col = np.arange(0, 1, 1 / y_size, np.float).reshape(-1, 1)
        mask = np.repeat(col, x_size, axis=1)
    mask = np.dstack([mask] * channel)
    # plt.imshow((mask*255).astype(np.uint8))
    # plt.axis('off')
    # plt.show()
    return mask


# Blending patches with half overlapped patches and gradual change masks.
def blending_patches(patch_arr, region_size, patch_sz=256, step_sz=128):
    roi_img = np.zeros((region_size[0], region_size[1], 3))  # define a black(empty) 3 channels image
    row_past_img = np.zeros((patch_arr.shape[1], patch_sz, region_size[1], 3))  # h
    h_mask = get_blending_mask(step_sz, patch_sz, "H")
    v_mask = get_blending_mask(region_size[1], step_sz, "V")
    for i in range(patch_arr.shape[1]):  # rows
        row_past_img[i, 0:patch_sz, 0: patch_sz, :] = patch_arr[i, 0, :]
        for j in range(1, patch_arr.shape[0]):  # columns
            patch = patch_arr[i, j, :]
            row_past_img[i, :, j*step_sz:(j+1)*step_sz, :] = row_past_img[i, :, j*step_sz:(j+1)*step_sz,:]*(1-h_mask) + patch[:, 0:step_sz, :]*h_mask
            row_past_img[i, :, (j+1)*step_sz:(j+2)*step_sz, :] = patch[:, step_sz:, :]
            # plt.imshow(row_past_img[i, :].astype(np.uint8))
            # plt.show()
    roi_img[0:patch_sz, :] = row_past_img[0]
    for i in range(1, patch_arr.shape[1]):
        roi_img[i*step_sz:(i+1)*step_sz, :] = roi_img[i*step_sz:(i+1)*step_sz, :]*(1-v_mask) + row_past_img[i, 0:step_sz, :, :]*v_mask
        roi_img[(i+1)*step_sz:(i+2)*step_sz, :] = row_past_img[i, step_sz:, :]
        # plt.imshow(roi_img.astype(np.uint8))
        # plt.show()
    return roi_img.astype(np.uint8)


# given a location and a region to be reconstruct, convert the coordinates into
# coordinates that patches have been processed.
def get_closes_location(location, region_size, step_sz):
    loc_x = location[0]
    loc_y = location[1]
    reg_x = region_size[0]
    reg_y = region_size[1]
    for i in range(step_sz):
        if loc_x % step_sz == 0:
            break
        else:
            loc_x -= 1
    for i in range(step_sz):
        if loc_y % step_sz == 0:
            break
        else:
            loc_y -= 1
    for i in range(step_sz):
        if (loc_x+reg_x) % step_sz == 0:
            break
        else:
            reg_x += 1
    for i in range(step_sz):
        if (loc_y+reg_y) % step_sz == 0:
            break
        else:
            reg_y += 1
    return [loc_x, loc_y], [reg_x, reg_y]


# Get relevant image file names from a folder according to coordinates saved in the file names
# Depends on how the image names are defined, modification may be necessary, but coordinates must be part of file name
def get_relevant_img_fn(opt_location, opt_region_size, step_sz, img_dir):
    fn_map = {}
    img_list = os.listdir(img_dir)
    for (i, j) in product(range(0, opt_region_size[0], step_sz), range(0, opt_region_size[1], step_sz)):
        loc_x = opt_location[0] + i
        loc_y = opt_location[1] + j
        for img_name in img_list:
            if str(loc_x) in img_name and str(loc_y) in img_name:
                if "outputs" in img_name:
                    # print("%d, %d, %s" % (int(i / step_sz), int(j / step_sz), img_name))
                    dic_key_str = str.format("%d, %d" % (int(i / step_sz), int(j / step_sz)))
                    fn_map[dic_key_str] = img_name
    return fn_map


def get_relevant_img_fn_testing(org_location, opt_region_size, step_sz, img_dir, f_type="outputs"):
    fn_map = {}
    img_list = os.listdir(img_dir)
    for (i, j) in product(range(0, opt_region_size[0], step_sz), range(0, opt_region_size[1], step_sz)):
        loc_x = org_location[0] + i
        loc_y = org_location[1] + j
        for img_name in img_list:
            if str(loc_x) in img_name and str(loc_y) in img_name:
                if f_type in img_name:
                    # print("%d, %d, %s" % (int(i / step_sz), int(j / step_sz), img_name))
                    dic_key_str = str.format("%d, %d" % (int(i/step_sz), int(j/step_sz)))
                    fn_map[dic_key_str] = img_name
    return fn_map


def get_relevant_uuid_img_fn(org_location, opt_region_size, step_sz, img_dir, uuid, f_type="outputs"):
    fn_map = {}
    img_list = os.listdir(img_dir)
    for img in img_list:
        if uuid not in img:
            img_list.remove(img)
    for (i, j) in product(range(0, opt_region_size[0], step_sz), range(0, opt_region_size[1], step_sz)):
        loc_x = org_location[0] + i
        loc_y = org_location[1] + j
        for img_name in img_list:
            ele, label, file_type = parse_filename(img_name, os.path.splitext(img_name)[1])
            # print("%s, %s, %s, %s" % (str(loc_x), ele[2], str(loc_y), ele[2]))
            if str(loc_x) == ele[1] and str(loc_y) == ele[2]:
                print(img_name)
                print("%s, %s, %s, %s" % (str(loc_x), ele[1], str(loc_y), ele[2]))
                print("%s, %s" % (f_type, file_type))
                if f_type == file_type:
                    print("%d, %d, %s" % (int(i / step_sz), int(j / step_sz), img_name))
                    dic_key_str = str.format("%d, %d" % (int(i/step_sz), int(j/step_sz)))
                    fn_map[dic_key_str] = img_name
    return fn_map


# Read relevant images and save RGB data into an array
def get_patch_arr(img_dir, img_file_name_map, opt_region_size, patch_sz, channels, step_sz):
    col_cnt = len(range(0, opt_region_size[0], step_sz))
    row_cnt = len(range(0, opt_region_size[1], step_sz))
    patch_arr = np.zeros([row_cnt, col_cnt, patch_sz, patch_sz, channels]).astype(np.uint8)
    for i in range(col_cnt):
        for j in range(row_cnt):
            dic_key_str = str.format("%d, %d" % (i, j))  # depends on how you save the image file names.
            if dic_key_str in img_file_name_map.keys():
                img = Image.open(os.path.join(img_dir, img_file_name_map.get(dic_key_str))).convert("RGB")
                patch_arr[j, i, :] = np.array(img)
            else:
                patch_arr[j, i, :] = np.ones([patch_sz, patch_sz, channels]).astype(np.uint8)+254
                print("Warning, no specific (coord: %d, %d) patch in the folder, replace it with white" % (i, j))
            # plt.imshow(patch_arr[j, i])
            # plt.show()
    return patch_arr


# restore image region
def restore_region(location, region_size, step_sz, patch_sz, channels, img_dir, uuid, chop=True):
    opt_location, opt_region_size = get_closes_location(location, region_size, step_sz)
    # img_fn_map = get_relevant_img_fn(opt_location, opt_region_size, step_sz, img_dir)
    # img_fn_map = get_relevant_img_fn_testing(location, opt_region_size, step_sz, img_dir)
    # img_fn_map = get_relevant_uuid_img_fn(location, opt_region_size, step_sz, img_dir, uuid)
    img_fn_map = get_relevant_uuid_img_fn(opt_location, opt_region_size, step_sz, img_dir, uuid, f_type="outputs")

    patch_arr = get_patch_arr(img_dir, img_fn_map, opt_region_size, patch_sz, channels, step_sz)
    p_region_size = (opt_region_size[0] + step_sz, opt_region_size[1] + step_sz)

    # org_img_fn_map = get_relevant_img_fn_testing(location, opt_region_size, step_sz, img_dir, f_type="inputs")
    org_img_fn_map = get_relevant_uuid_img_fn(opt_location, opt_region_size, step_sz, img_dir, uuid, f_type="inputs")
    org_patch_arr = get_patch_arr(img_dir, org_img_fn_map, opt_region_size, patch_sz, channels, step_sz)
    # org_direct_img = blending_patches(org_patch_arr, p_region_size)
    org_direct_img = direct_reconstruct(org_patch_arr, step_sz, channels, p_region_size)

    direct_rec_img = direct_reconstruct(patch_arr, step_sz, channels, p_region_size)
    blended_rec_img = blending_patches(patch_arr, p_region_size)
    if not chop:
        return blended_rec_img, direct_rec_img, org_direct_img
    else:
        roi_x = location[0] - opt_location[0]
        roi_y = location[1] - opt_location[1]
        blend_img = blended_rec_img[roi_x:roi_x + region_size[0], roi_y: roi_y + region_size[1], :]
        direct_img = direct_rec_img[roi_x:roi_x + region_size[0], roi_y: roi_y + region_size[1], :]
        org_img = org_direct_img[roi_x:roi_x + region_size[0], roi_y: roi_y + region_size[1], :]
        return blend_img, direct_img, org_img


if __name__ == "__main__":
    # region_size = [1000, 1000]
    # org_location = [39824, 40800]
    # region_size = [256, 256]
    # org_location = [40200, 41200]
    # org_location = [41200, 41200]
    region_size = [256, 256]
    org_location = [40416, 41908]

    step_sz, patch_sz = (128, 256)
    channels = 3
    wsi_uuid = "7470963d479b4576bc8768b389b1882e"

    img_dir = "/projects/shart/digital_pathology/data/PenMarking/eval/pixel2pixel_256/images_dispatch/7470963d479b4576bc8768b389b1882e/"
    img_dir_out = "/projects/shart/digital_pathology/data/PenMarking/eval/pixel2pixel_256/patch_blendings"

    # img_dir = "H:\\temp1\\large_restored\\images"
    # img_dir_out = "H:\\temp1\\large_restored\\images_out"

    # blended_rec_img, direct_rec_img = restore_region(org_location, region_size, step_sz, patch_sz, channels, img_dir, chop=True)
    blended_rec_img, direct_rec_img, original_img = restore_region(org_location, region_size, step_sz, patch_sz, channels, img_dir, wsi_uuid, chop=False)
    print(blended_rec_img.shape)
    print(direct_rec_img.shape)
    fig, (axs1, axs2) = plt.subplots(1, 2)
    axs1.imshow(direct_rec_img)
    axs2.imshow(blended_rec_img)
    axs1.axis("off")
    axs2.axis("off")
    plt.show()

    direct_name = os.path.join(img_dir_out, "direct_stitch.jpg")
    Image.fromarray(direct_rec_img).save(direct_name)
    blending_name = os.path.join(img_dir_out, "blending_stitch.jpg")
    Image.fromarray(blended_rec_img).save(blending_name)
    org_name = os.path.join(img_dir_out, "original_img.jpg")
    Image.fromarray(original_img).save(org_name)






