Tissue detection is necessary for WSI analysis, because expansive areas of WSI are blank.
Tissue detector can help us save a lot of time for patch extraction.

We usually do tissue detection in a downscaled WSI. For each dataset/project, the downscale factor should be the constant, for example 128. 
The tissue detection works in this steps:
1. get the downscaled WSI
2. get the tissue(foreground) of the downscaled WSI, it's a binary mask
3. get the pixel indices from the binary image 
4. map the pixel indices back into the actual location in the original size of WSI by multiply downscale factor.

```python
# Relevant Code
pos_indices = np.where(wsi_tissue_mask > 0)
loc_x = (np.array(pos_indices[1]) * rescale_factor).astype(np.int)
loc_y = (np.array(pos_indices[0]) * rescale_factor).astype(np.int)
```

Currently we provide two options for tissue detection. 1. Threshold; 2. Gaussian Naive Bayes (GNB) Model.
1) Threshold method just convert RGB color image into LAB space, and do threshold.
2) If you would like to use GNB for tissue detector, you may need to annotate some pixel points for model training. The trained model can be used to predict which pixels are tissue in a certain of downscaled WSI.  
We provide an [annotation tool](../../wsitools/tissue_detection/pixel_sampling_tool)to help. This tool can track left and right key of mouse to sample the pixel values of foreground (Blue) and background (Red), and save the values into a tsv file.   
We also provide [sample tsv files](../../wsitools/tissue_detection/model_files), which can be used on H&E slides.
![annotation tool](../imgs/tissue_anno.png)


