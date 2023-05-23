import openslide
import glob
import csv

NUM = 10
root_dir = "/lus/grand/projects/gpu_hack/mayopath/data/TCGA"
fn_list = glob.glob(root_dir+"/*/*", recursive=True)

save_to = "./wsi_list_40x.csv"
fp = open(save_to, 'w')
wrt_str = ""

save_to_2 = "./wsi_list_others.csv"
fp2 = open(save_to, 'w')
wrt_str_2 = ""

for f in fn_list:
    if ".svs" in f or '.tif' in f:
        wsi_obj = openslide.open_slide(f)
        wsi_prop = dict(wsi_obj.properties)
        if wsi_prop.get("openslide.mpp-x") == None:
            print("Don't know the pixel size")
        else:
            pixel_size = wsi_prop["openslide.mpp-x"]
            print(pixel_size)
            if float(pixel_size)< 0.27: # 40x
                wrt_str += f + "\n"
            else:
                wrt_str_2 += f + "\n"

fp2.write(wrt_str_2)
fp.write(wrt_str)
fp.close()
fp2.close()








