Linux = False
train_name = "BSDS"
encoder = "m1"  # m1 or m2
Batch_size = 2
# ---------------指定数据路径(无需改动)--------------------
if Linux:
    basic_dir = "/home/featurize/work/Datasets/"
    sp = "/"
else:
    basic_dir = "F:/2.Datasets/"
    sp = "\\"

basic_dir = basic_dir + train_name
train_images_dir = basic_dir + "/train/image/*/*/*.jpg"
train_annos_dir = basic_dir + "/train/gt/*/*/*.png"
# train_noises_dir = basic_dir + "/train/noise/*.png"

test_images_dir = basic_dir + "/test/image/*.jpg"
# precess_dir = "./sample/*.*"
# pre_path = './result/'
# label_path = basic_dir + "/test/mask_resize"
