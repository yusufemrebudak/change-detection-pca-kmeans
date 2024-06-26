import cv2
import gc
import numpy as np
import pandas as pd
import rasterio
from skimage.color import lab2lch, rgb2lab
from skimage.exposure import rescale_intensity
from skimage.morphology import disk
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


# 79119_50979
picname = "79119_50988"
image1_path = "A/"+picname+".jpg"
image2_path = "B/"+picname+".jpg"

# picname1 = "ElephantButte_08201991_md"
# picname2 = "ElephantButte_08272011"
# image1_path = "images/ElephantButte/"+picname1+".jpg"
# image2_path = "images/ElephantButte/"+picname2+".jpg"

out_dir = "../output/"

print('[INFO] Start Change Detection ...')
print('[INFO] Importing Librairies ...')

import sklearn
from sklearn.cluster import KMeans
from collections import Counter
from sklearn.decomposition import PCA
import skimage.morphology
import time


def shadow_detection(image_file, shadow_mask_file, convolve_window_size , num_thresholds , struc_elem_size ):
    """
    This function is used to detect shadow - covered areas in an image, as proposed in the paper 
    'Near Real - Time Shadow Detection and Removal in Aerial Motion Imagery Application' by Silva G.F., Carneiro G.B., 
    Doth R., Amaral L.A., de Azevedo D.F.G. (2017)
    
    Inputs:
    - image_file: Path of image to be processed for shadow removal. It is assumed that the first 3 channels are ordered as
                  Red, Green and Blue respectively
    - shadow_mask_file: Path of shadow mask to be saved
    - convolve_window_size: Size of convolutional matrix filter to be used for blurring of specthem ratio image
    - num_thresholds: Number of thresholds to be used for automatic multilevel global threshold determination
    - struc_elem_size: Size of disk - shaped structuring element to be used for morphological closing operation
    
    Outputs:
    - shadow_mask: Shadow mask for input image
    
    """
    
    if (convolve_window_size % 2 == 0):
        raise ValueError('Please make sure that convolve_window_size is an odd integer')
        
    buffer = int((convolve_window_size - 1) / 2)
    
    
    with rasterio.open(image_file) as f:
        metadata = f.profile
        img = rescale_intensity(np.transpose(f.read(tuple(np.arange(metadata['count']) + 1)), [1, 2, 0]), out_range = 'uint8')
        img = img[:, :, 0 : 3]
    
    
    lch_img = np.float32(lab2lch(rgb2lab(img)))
    
    
    l_norm = rescale_intensity(lch_img[:, :, 0], out_range = (0, 1))
    h_norm = rescale_intensity(lch_img[:, :, 2], out_range = (0, 1))
    sr_img = (h_norm + 1) / (l_norm + 1)
    log_sr_img = np.log(sr_img + 1)
    
    del l_norm, h_norm, sr_img
    gc.collect()

    

    avg_kernel = np.ones((convolve_window_size, convolve_window_size)) / (convolve_window_size ** 2)
    blurred_sr_img = cv2.filter2D(log_sr_img, ddepth = -1, kernel = avg_kernel)
      
    
    del log_sr_img
    gc.collect()
    
                
    flattened_sr_img = blurred_sr_img.flatten().reshape((-1, 1))
    labels = KMeans(n_clusters = num_thresholds + 1, max_iter = 10000).fit(flattened_sr_img).labels_
    flattened_sr_img = flattened_sr_img.flatten()
    df = pd.DataFrame({'sample_pixels': flattened_sr_img, 'cluster': labels})
    threshold_value = df.groupby(['cluster']).min().max()[0]
    df['Segmented'] = np.uint8(df['sample_pixels'] >= threshold_value)
    
    
    del blurred_sr_img, flattened_sr_img, labels, threshold_value
    gc.collect()
    
    
    shadow_mask_initial = np.array(df['Segmented']).reshape((img.shape[0], img.shape[1]))
    struc_elem = disk(struc_elem_size)
    shadow_mask = np.expand_dims(np.uint8(cv2.morphologyEx(shadow_mask_initial, cv2.MORPH_CLOSE, struc_elem)), axis = 0)
    
    
    del df, shadow_mask_initial, struc_elem
    gc.collect()
    

    metadata['count'] = 1
    with rasterio.open(shadow_mask_file, 'w', **metadata) as dst:
        dst.write(shadow_mask)
        
    return shadow_mask



def shadow_correction(image_file, shadow_mask_file, corrected_image_file, exponent = 1):
    """
    This function is used to adjust brightness for shadow - covered areas in an image, as proposed in the paper 
    'Near Real - Time Shadow Detection and Removal in Aerial Motion Imagery Application' by Silva G.F., Carneiro G.B., 
    Doth R., Amaral L.A., de Azevedo D.F.G. (2017)
    
    Inputs:
    - image_file: Path of 3 - channel (red, green, blue) image to be processed for shadow removal
    - shadow_mask_file: Path of shadow mask for corresponding input image
    - corrected_image_file: Path of corrected image to be saved
    - exponent: Exponent to be used for the calculcation of statistics for unshaded and shaded areas
    
    Outputs:
    - corrected_img: Corrected input image
    
    """
    
    with rasterio.open(image_file) as f:
        metadata = f.profile
        img = rescale_intensity(np.transpose(f.read(tuple(np.arange(metadata['count']) + 1)), [1, 2, 0]), out_range = 'uint8')
        
    with rasterio.open(shadow_mask_file) as s:
        shadow_mask = s.read(1)
    
    corrected_img = np.zeros((img.shape), dtype = np.uint8)
    non_shadow_mask = np.uint8(shadow_mask == 0)
    
    
    for i in range(img.shape[2]):
        shadow_area_mask = shadow_mask * img[:, :, i]
        non_shadow_area_mask = non_shadow_mask * img[:, :, i]
        shadow_stats = np.float32(np.mean(((shadow_area_mask ** exponent) / np.sum(shadow_mask))) ** (1 / exponent))
        non_shadow_stats = np.float32(np.mean(((non_shadow_area_mask ** exponent) / np.sum(non_shadow_mask))) ** (1 / exponent))
        mul_ratio = ((non_shadow_stats - shadow_stats) / shadow_stats) + 1
        corrected_img[:, :, i] = np.uint8(non_shadow_area_mask + np.clip(shadow_area_mask * mul_ratio, 0, 255))
    

    with rasterio.open(corrected_image_file, 'w', **metadata) as dst:
        dst.write(np.transpose(corrected_img, [2, 0, 1]))
        
    return corrected_img


def normalize_image(image):
    return cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)

def find_vector_set(diff_image, new_size):
 
    i = 0
    j = 0
    vector_set = np.zeros((int(new_size[0] * new_size[1] / 25),25))
    while j < new_size[1]:
        k = 0
        while k < new_size[0]:
            block   = diff_image[j:j+5, k:k+5]
            feature = block.ravel()
            vector_set[i, :] = feature
            k = k + 5
            i = i + 1
        j = j + 5
       
    mean_vec   = np.mean(vector_set, axis = 0)
    # 5*5 lik blok bir feature olarak düşünüyoruz ve mean i alınınca her bir featuremın ortalamasını buluyoruz.
    # mean_vec   = np.mean(vector_set, axis = 1)
    # Mean normalization
    
    vector_set = vector_set-mean_vec
   
    return vector_set, mean_vec

def find_FVS(EVS, diff_image, mean_vec, new):
 
    i = 2
    feature_vector_set = []
 
    while i < new[1] - 2:
        j = 2
        while j < new[0] - 2:
            block = diff_image[i-2:i+3, j-2:j+3]
            feature = block.flatten()
            feature_vector_set.append(feature)
            j = j+1
        i = i+1
 
    FVS = np.dot(feature_vector_set, EVS)
    #  a*b lik bir matribi b*c ye sahip bir matrisle çarpabilirim ve a*c oluşur.
    
    FVS = FVS - mean_vec # we will see this later.
    # this is our new space defined by pca, we transfrom our data to new space defined by pca
    print ("[INFO] Feature vector space size", FVS.shape)
    return FVS

def clustering(FVS, components, new):
    kmeans = KMeans(components, verbose = 0)
    kmeans.fit(FVS)
    output = kmeans.predict(FVS)
    count  = Counter(output)
 
    least_index = min(count, key = count.get)
    change_map  = np.reshape(output,(new[1] - 4, new[0] - 4))
    return least_index, change_map
    

# Read Images
print('[INFO] Reading Images ...')
start = time.time()

image1 = cv2.imread(image1_path)
image2 = cv2.imread(image2_path)
# cv2.IMREAD_GRAYSCALE

mask_out_dir1 = "masked/masked1.jpeg"
mask_out_dir2 = "masked/masked2.jpeg"

corrected_out_dir1 = "maskedCorrected/corrected1.jpeg"
corrected_out_dir2 = "maskedCorrected/corrected2.jpeg"


shadow_mask1 = shadow_detection(image1_path, mask_out_dir1, 5 , 5 , 5)
shadow_mask1= shadow_mask1[0,:,:]
# corrected_img1 = shadow_correction(image1_path, mask_out_dir1, corrected_out_dir1,1)

shadow_mask2 = shadow_detection(image2_path, mask_out_dir2, 5 , 5 , 5)
shadow_mask2= shadow_mask2[0,:,:]
concatenate_shadow_mask = np.logical_or(shadow_mask1,shadow_mask2).astype(int)
concatenate_shadow_mask[concatenate_shadow_mask == False] = 0
concatenate_shadow_mask[concatenate_shadow_mask == True] = 1

# corrected_img2 = shadow_correction(image2_path, mask_out_dir2, corrected_out_dir2,1)
# image1 = image1
# image2 = image1

end = time.time()
print('[INFO] Reading Images took {} seconds'.format(end-start))


# Resize Images
print('[INFO] Resizing Images ...')
start = time.time()
new_size = np.asarray(image1.shape) /5
new_size = new_size.astype(int) *5
image1 = cv2.resize(image1, (new_size[0],new_size[1])).astype(np.float32)
image2 = cv2.resize(image2, (new_size[0],new_size[1])).astype(np.float32)
image1 = cv2.cvtColor(image1, cv2.COLOR_RGB2GRAY)
image2 = cv2.cvtColor(image2, cv2.COLOR_RGB2GRAY)

normalize_image(image1)
normalize_image(image2)
end = time.time()
print('[INFO] Resizing Images took {} seconds'.format(end-start))

# Difference Image
print('[INFO] Computing Difference Image ...')
start = time.time()
diff_image = abs(image1 - image2)

cv2.imwrite(out_dir+'difference.jpg', diff_image)
end = time.time()
print('[INFO] Computing Difference Image took {} seconds'.format(end-start))
# diff_image=diff_image[:,:,1]



print('[INFO] Performing PCA ...')
start = time.time()
pca = PCA()
# pca = PCA(n_components=4)
vector_set, mean_vec=find_vector_set(diff_image, new_size)
pca.fit(vector_set)
EVS = pca.components_

# evs de her 0.kolondan 24.kolona benim eigenvektorlerim 
end = time.time()
print('[INFO] Performing PCA took {} seconds'.format(end-start))

print('[INFO] Building Feature Vector Space ...')
start = time.time()
FVS = find_FVS(EVS, diff_image, mean_vec, new_size)
# fvs lerde her bir satır benim kutum. 25 tane , 5 e 5
components =3
end = time.time()
print('[INFO] Building Feature Vector Space took {} seconds'.format(end-start))

print('[INFO] Clustering ...')
start = time.time()
least_index, change_map = clustering(FVS, components, new_size)
end = time.time()
print('[INFO] Clustering took {} seconds'.format(end-start))

change_map[change_map == least_index] = 255
change_map[change_map != 255] = 0
change_map = change_map.astype(np.uint8)
concatenate_shadow_mask = concatenate_shadow_mask[0:new_size[0]-4,0:new_size[0]-4]
# concatenate_shadow_mask = cv2.resize(concatenate_shadow_mask, (new_size[0]-4,new_size[1]-4)).astype(np.float32)

change_map[concatenate_shadow_mask == True] = 0 
print('[INFO] Save Change Map ...')
cv2.imwrite(out_dir+'ChangeMap.jpg', change_map)

print('[INFO] Performing Closing ...')
print('[WARNING] Kernel is fixed depending on image topology')
print('[WARNING] Closing with disk-shaped structuring element with radius equal to 6')
kernel = skimage.morphology.disk(6)
CloseMap = cv2.morphologyEx(change_map, cv2.MORPH_CLOSE, kernel)
cv2.imwrite(out_dir+'CloseMap.jpg', CloseMap)

print('[INFO] Performing Opening ...')
OpenMap = cv2.morphologyEx(CloseMap, cv2.MORPH_OPEN, kernel)
cv2.imwrite(out_dir+'OpenMap.jpg', OpenMap)
#########################################
import matplotlib.pyplot as plt
fig, axs = plt.subplots(1, 3)
image3 = plt.imread(image1_path)
image4 = plt.imread(image2_path)
axs[0].imshow(image3)
axs[1].imshow(image4)

label = np.load("label/"+picname+".npz")
for key_ in label:
	coord = label[key_]
	xs, ys = zip(*coord) #create lists of x and y values

	#plt.figure()
	axs[1].plot(xs,ys)

axs[0].imshow(image3)
axs[1].imshow(image4)
axs[2].imshow(change_map,cmap="gray")
fig.show()













print('[INFO] End Change Detection')