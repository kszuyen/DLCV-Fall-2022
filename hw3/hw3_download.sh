#!/bin/bash

# for part 1
python -c "import clip; clip.load('ViT-B/32')"

# for part 2
wget -O vit_large_patch14_224_clip_laion2b.pth "https://www.dropbox.com/s/c6nzqx1hzdauup9/vit_large_patch14_224_clip_laion2b.pth?dl=1"