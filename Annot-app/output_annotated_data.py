#!/usr/bin/env python3
"""
收集已完成的关键点标注数据到Final_records文件夹
"""
import os
import json
import shutil
from datetime import datetime

def collect_annotations():
    # 设置路径
    data_root = "./data_to_annotate"
    # final_records_root = "./Final_records"
    final_records_root = "/data/boran/4dhoi/Dataset/data_to_optimize/"

    if not os.path.exists(final_records_root):
        os.makedirs(final_records_root)
        print(f"Created final records root folder: {final_records_root}")
    
    # 创建以日期时间命名的文件夹
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    collection_folder = os.path.join(final_records_root, f"collection_{timestamp}")
    
    if not os.path.exists(collection_folder):
        os.makedirs(collection_folder)
        print(f"Created collection folder: {collection_folder}")
    
    collected_count = 0
    skipped_count = 0
    
    # 遍历data_to_annotate下的所有文件夹
    if not os.path.exists(data_root):
        print(f"Error: {data_root} does not exist")
        return
    
    for pack in os.listdir(data_root):
        pack_dir = os.path.join(data_root, pack)
        if not os.path.isdir(pack_dir):
            continue
            
        # 检查是否有子文件夹结构
        for second_level in sorted(os.listdir(pack_dir)):
            ds_dir = os.path.join(pack_dir, second_level)
            if not os.path.isdir(ds_dir):
                continue
            
            # 检查是否存在kp_record_merged.json
            kp_file = os.path.join(ds_dir, "kp_record_merged.json")
            if not os.path.exists(kp_file):
                continue
            
            # 使用最后一级文件夹作为文件名
            dataset_name = second_level
            
            # 检查是否已经处理过（存在meta.json）
            meta_file = os.path.join(ds_dir, "meta.json")
            if os.path.exists(meta_file):
                print(f"Skipped {dataset_name}: already processed (meta.json exists)")
                skipped_count += 1
                continue
            
            # 直接复制到collection_folder，不创建子文件夹
            target_file = os.path.join(collection_folder, f"{dataset_name}.json")
            try:
                shutil.copy2(kp_file, target_file)
                print(f"Copied: {kp_file} -> {target_file}")
                
                # 创建meta.json标记文件
                meta_data = {
                    "processed_at": datetime.now().isoformat(),
                    "source_path": ds_dir,
                    "target_path": target_file,
                    "collection_batch": f"collection_{timestamp}",
                    "status": "collected"
                }
                
                with open(meta_file, "w", encoding="utf-8") as f:
                    json.dump(meta_data, f, indent=2, ensure_ascii=False)
                
                print(f"Created meta file: {meta_file}")
                collected_count += 1
                
            except Exception as e:
                print(f"Error processing {dataset_name}: {e}")
    
    # # 创建收集批次的总结文件
    # summary_file = os.path.join(collection_folder, "collection_summary.json")
    # summary_data = {
    #     "collection_time": datetime.now().isoformat(),
    #     "collected_datasets": collected_count,
    #     "skipped_datasets": skipped_count,
    #     "total_processed": collected_count + skipped_count
    # }
    
    # with open(summary_file, "w", encoding="utf-8") as f:
    #     json.dump(summary_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n=== Collection Summary ===")
    print(f"Collection folder: {collection_folder}")
    print(f"Collected datasets: {collected_count}")
    print(f"Skipped datasets: {skipped_count}")
    print(f"Total processed: {collected_count + skipped_count}")
    # print(f"Summary saved to: {summary_file}")

if __name__ == "__main__":
    collect_annotations()