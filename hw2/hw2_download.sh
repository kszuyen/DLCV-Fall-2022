#!/bin/bash

# download part 1 dcgan model
wget -O dcgan_checkpoint.pth "https://www.dropbox.com/s/lql636lz50h5qes/dcgan_checkpoint.pth?dl=1"

# download part 2 ddpm model
wget -O ddpm_checkpoint.pth "https://www.dropbox.com/s/tee4dfez19x4zuf/ddpm_checkpoint.pth?dl=1"

# download part 3 dann models
wget -O dann_mnistm_2_svhn.pth "https://www.dropbox.com/s/khuxpx12ohdooft/dann_mnistm_2_svhn.pth?dl=1"
wget -O dann_mnistm_2_usps.pth "https://www.dropbox.com/s/uey9ng3y97enql8/dann_mnistm_2_usps.pth?dl=1"