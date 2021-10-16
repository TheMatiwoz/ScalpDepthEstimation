import os

# for root, dirs, files in os.walk(r"D:\Programowanie\DL\sfm\try2\images"):
#     for filename in files:
#         os.rename(r'file path\OLD file name.file type', r'file path\NEW file name.file type')
#         print(filename)
entries = os.listdir(r"D:\Programowanie\DL\sfm\try2\images")

for image_path in entries:
    os.rename(rf"D:\Programowanie\DL\sfm\try2\images\{image_path}", rf"D:\Programowanie\DL\sfm\try2\images\000{image_path}")

