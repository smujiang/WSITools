import openslide
import glob
import csv

NUM = 10
root_dir = "/lus/grand/projects/gpu_hack/mayopath/data/TCGA"
fn_list = glob.glob(root_dir+"/*/*", recursive=True)
all_wsi_fn_eligible = []
all_wsi_fn_low_resolution = []
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
                all_wsi_fn_eligible.append(f)
            else:
                all_wsi_fn_low_resolution.append(f)

save_to = "./wsi_list_40x.csv"
f = open(save_to, 'w')
write = csv.writer(f)
write.writerows(all_wsi_fn_eligible)








