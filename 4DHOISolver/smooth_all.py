
import argparse
import os
import json
import sys
import cv2
import numpy as np
import torch
                   
import open3d as o3d
from copy import deepcopy

from video_optimizer.utils.dataset_util import (
    get_records_by_annotation_progress,
    update_record_annotation_progress,
    get_static_flag_from_merged,
    validate_record_for_optimization
)
from video_optimizer.utils.smoothing_utils import lowpass_smooth_all_dict
import datetime, inspect
import re
from pathlib import Path
def main():                              
    ready_records = get_records_by_annotation_progress(5)
    if not ready_records:
        return
    
    print(f"{len(ready_records)} records to smooth:")
    for i, record in enumerate(ready_records):
        print(f"  {i+1}. {record.get('object_category', 'Unknown')} - {record.get('file_name', 'Unknown')} (ID: {record.get('id', 'Unknown')})")
    batch_size = len(ready_records)// 2 + 1  
    success_count = 0
    for i, record in enumerate(ready_records):
        sesseion_folder = record.get("session_folder", "")
        final_json_path = os.path.join(sesseion_folder, 'transformed_parameters_final.json')
        try:
            with open(final_json_path, 'r', encoding='utf-8') as f:
                final_json = json.load(f)
        except:
            print(f"{record.get('file_name', 'Unknown')}no global parameters")
            continue
        smooth_json = lowpass_smooth_all_dict(final_json)
        with open(final_json_path, 'w', encoding='utf-8') as f:
            json.dump(smooth_json, f, indent=4)
        print(sesseion_folder)
        print(f"\n{'='*60}")
        print(f"Smoothing {i+1}/{len(ready_records)} record")
        print(f"Object category: {record.get('object_category', 'Unknown')}")
        print(f"File name: {record.get('file_name', 'Unknown')}")
        print(f"{'='*60}")                          
        success_count += 1
    print(f"\n Successfully smoothed {success_count}/{len(ready_records)} records")
if __name__ == "__main__":
    main()