#!/usr/bin/env bash
# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

# https://roboimagedata.compute.dtu.dk/?page_id=36

OUTPUT_PATH=$1

cd $OUTPUT_PATH

# From official website.
wget http://roboimagedata2.compute.dtu.dk/data/MVS/SampleSet.zip
wget http://roboimagedata2.compute.dtu.dk/data/MVS/Points.zip
wget http://roboimagedata2.compute.dtu.dk/data/MVS/Points_MVS.zip
wget http://roboimagedata2.compute.dtu.dk/data/MVS/Surfaces.zip
wget http://roboimagedata2.compute.dtu.dk/data/MVS/Cleaned.zip
wget http://roboimagedata2.compute.dtu.dk/data/MVS/Rectified.zip

# For other datasets, we follow: https://github.com/YoYo000/MVSNet
#  Training set: https://drive.google.com/file/d/1eDjh-_bxKKnEuz5h-HXS7EDJn59clx6V/view
#  Test set: https://drive.google.com/open?id=135oKPefcPTsdtLRzoDAQtPpHuoIrpRI_
#  GT Depths map: https://virutalbuy-public.oss-cn-hangzhou.aliyuncs.com/share/cascade-stereo/CasMVSNet/dtu_data/dtu_train_hr/Depths_raw.zip

# After downloading, please arrange the dataset into the following structure:
######################
# dtu
# ├── scan1
# │   ├── images
# │   ├── cams
# │   └── depths
# ├── scan2
# │   ├── images
# │   ├── cams
# │   └── depths
# │   ├── ...
########################
