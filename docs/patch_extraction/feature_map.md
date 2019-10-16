We use a csv file to define the key-value pairs to be saved into tfRecords. Maybe there are better ways, but currently we just make this way works.

Currently, we provide [four examples](../../wsitools/patch_extraction/feature_maps) to configure feature map, they correspond to four patch extraction scenarios in the [main readme](../../README.md).
 
The content of the csv file looks like below:

| key      | data_type | eval                 | description      |
|----------|-----------|----------------------|------------------|
| loc_x    | int       | int(loc_x[idx])      | patch location x |
| loc_y    | int       | int(loc_y[idx])      | patch location y |
| img_mode | bytes     | "RGB".encode("utf8") | image mode       |
| img_w    | int       | self.patch_size      | image width      |
| img_h    | int       | self.patch_size      | image height     |
| image    | bytes     | patch.tobytes()      | image data       |

1. The first column defines the keys;
2. The second column defines the data type saved into the tfRecords. We tested int and bytes only.
3. The third column defines the script should be evaluated in python code with ```eval()```. In this column, each row denotes how the value will be assigned.
Be aware, the name of variables should be the same as in [patch_extractor.py](../../wsitools/patch_extraction/patch_extractor.py
) or [pairwise_patch_extractor.py](../../wsitools/patch_extraction/pairwise_patch_extractor.py). If the variable you need is not accessible in these two files, you may need to clone our code and modify as you wish.  
4. The fourth column is the description of each feature to be saved into tfRecord.







