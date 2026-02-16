# -*- coding: utf-8 -*-
"""
Created on Thu Dec 11 14:18:55 2025

@author: Aron Kirschner
"""

import subprocess
import time

commands = [
    r'python "E:/Digital_Pathology/Project/Aron_to_share/Scripts/Train_multi_UNet_villi_pairweights.py"',
    r'python "E:/Digital_Pathology/Project/Aron_to_share/Scripts/Train_multi_UNet_villi_pairweights1.py"',
    r'python "E:/Digital_Pathology/Project/Aron_to_share/Scripts/Train_multi_UNet_villi_pairweights2.py"',
]


for cmd in commands:
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True)

    if result.returncode != 0:
        print(f"Command failed with code {result.returncode}")
        # break   # uncomment if you want to stop on error

