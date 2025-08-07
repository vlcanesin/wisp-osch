import os
import shutil
import glob

source_dir = '~/Downloads/archive'
dest_dir = './lsun/bedrooms'

os.makedirs(dest_dir, exist_ok=True)

image_extensions = ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.webp')
image_paths = []

for ext in image_extensions:
    image_paths.extend(glob.glob(os.path.join(source_dir, '**', ext), recursive=True))

image_paths.sort()

selected_images = image_paths[:50000]

for i, path in enumerate(selected_images):
    filename = f'image_{i:05d}' + os.path.splitext(path)[1]
    shutil.copy2(path, os.path.join(dest_dir, filename))

print(f'Copied {len(selected_images)} images to {dest_dir}')
