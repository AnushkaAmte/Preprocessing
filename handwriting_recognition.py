#from tensorflow.keras.layers.experimental.preprocessing import StringLookup
from numpy.core.numeric import indices
from numpy.testing._private.utils import assert_
from tensorflow import keras

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os
from tensorflow._api.v2 import image

from tensorflow.python.ops.gen_array_ops import pad

np.random.seed(42)

tf.random.set_seed(42)


base_path = "data"
words_list = []

words =open(f"{base_path}/words.txt","r").readlines()

for line in words:
    if line[0] == '#':
        continue
    if line.split(" ")[1] != "err":
        words_list.append(line)

len(words_list)
#print(len(words_list))
np.random.shuffle(words_list)

split_idx = int(0.9 * len(words_list))
train_samples= words_list[:split_idx]
test_samples = words_list[split_idx:]

val_split_idx = int(0.5 * len(test_samples))
validation_samples = test_samples[: val_split_idx]
test_samples =test_samples[val_split_idx:]

assert_ 
len(words_list) == len(train_samples) + len(validation_samples) + len(test_samples)

print(f"Total training samples: {len(train_samples)}")
print(f"Total validation samples: {len(validation_samples)}")
print(f"Total testing samples: {len(test_samples)}")


base_image_path= os.path.join(base_path,"words")

def get_image_paths_and_labels(samples):
    paths= []
    corrected_samples=[]
    for (i,file_line) in enumerate(samples):
        line_split= file_line.strip()
        line_split = line_split.split(" ")


        image_name = line_split[0]
        part1= image_name.split("-")[0]
        part2 = image_name.split("-")[1]
        img_path= os.path.join(base_image_path,part1,part1 + "-"+ part2,image_name + ".png")

        if os.path.getsize(img_path):
            paths.append(img_path)
            corrected_samples.append(file_line.split("\n")[0])

    return paths, corrected_samples


train_img_paths, train_labels = get_image_paths_and_labels(train_samples)
validation_img_paths, validation_labels = get_image_paths_and_labels(validation_samples) 
test_img_paths, test_labels = get_image_paths_and_labels(test_samples)


train_labels_cleaned = []
characters = set()
max_len=0

for label in train_labels:
    label = label.split(" ")[-1].strip()
    for char in label:
        characters.add(char)

    max_len= max(max_len,len(label))
    train_labels_cleaned.append(label)

print("Max length: ",max_len)
print("vocab size: ",len(characters))

print(" ", train_labels_cleaned[:10])

def clean_labels(labels):
    cleaned_labels=[]
    for label in labels:
        label=label.split(" ")[-1].strip()
        cleaned_labels.append(label)
    return cleaned_labels

validation_labels_cleaned = clean_labels(validation_labels)
test_labels_cleaned= clean_labels(test_labels)


AUTOTUNE = tf.data.AUTOTUNE

char_to_num = StringLookup(vocabulary=list(characters),mask_token=None)
num_to_char = StringLookup(vocabulary=char_to_num.get_vocabulary(), mask_token=None, invert=True)

def distortion_free_resize(image,img_size):
    w,h= img_size
    image= tf.image.resize(image,size=(h,w),preserve_aspect_ratio=True)

    pad_height = h- tf.shape(image)[0]
    pad_width = w- tf.shape(image)[1]

    if pad_height%2 !=0:
        heigth = pad_height //2
        pad_height_top = heigth +1
        pad_height_bottom =heigth
    else: 
        pad_height_top = pad_height_bottom = pad_height //2

    if pad_width %2 !=0:
        width = pad_width //2
        pad_width_left = width +1
        pad_width_right = width
    else:
        pad_width_left= pad_width_right = pad_width //2

    image = tf.pad(
        image,
        paddings=[
            [pad_height_top,pad_height_bottom],
            [pad_width_left,pad_width_right],
            [0,0]
        ],
    )

    image =tf.transpose(image, perm=[1,0,2])
    image =tf.image.flip_left_right(image)
    return image

batch_size =64
padding_token =99
image_width =128
image_height = 32

def preprocess_image(image_path, img_size=(image_width,image_height)):
    image= tf.io.read_file(image_path)
    image = tf.image.decode_png(image,1)
    image= distortion_free_resize(image,img_size)
    image =tf.cast(image, tf.float32) / 255.0

    return image

def vectorize_label(label):
    label= char_to_num(tf.strings.unicode_split(label, input_encoding="UTF-8"))
    length= tf.shape(label)[0]
    pad_amount = max_len -length
    label= tf.pad(label, paddings=[[0,pad_amount]],constant_values=padding_token)

    return label

def process_image_labels(image_path,label):
    image = preprocess_image(image_path)
    label= vectorize_label(label)

    return {"image": image, "label": label}

def prepare_datset(image_paths, labels):
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels)).map(
        process_image_labels, num_parallel_calls= AUTOTUNE
    )

    return dataset.batch(batch_size).cache().prefetch(AUTOTUNE)


train_ds = prepare_datset(train_img_paths, train_labels_cleaned)
validation_ds = prepare_datset(validation_img_paths, validation_labels_cleaned)
test_ds = prepare_datset(test_img_paths, test_labels_cleaned)


for data in train_ds.take(1):
    images, labels = data["image"], data["label"]

    _, ax = plt.subplots(4,4, figsize = (15,8))
    for i in range(16):
        img = image[i]
        img= tf.image.flip_left_right(img)
        img = tf.transpose(img,perm=[1,0,2])
        img= (img *255.0).numpy().clip(0,255).astype(np.uint8)
        img =img [:,:,0]

        label = labels[i]
        indices = tf.gather(label, tf.where(tf.math.not_equal(label, padding_token)))

        label =tf.strings.reduce_join(num_to_char(indices))

        label = label.numpy().decode("utf-8")

        ax[i //4, i% 4].imshow(img,cmap="gray")
        ax[i //4, i% 4].set_title(label)
        ax[i//4,i%4].axis("off")

plt.show()
