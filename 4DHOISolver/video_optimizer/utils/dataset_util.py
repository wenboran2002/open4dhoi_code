import json
import os
from typing import Dict, List, Any, Optional

def load_upload_records(records_path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(records_path):
        print(f"upload_records.json not found: {records_path}")
        return []
    
    try:
        with open(records_path, 'r', encoding='utf-8') as f:
            records = json.load(f)
        return records if isinstance(records, list) else []
    except Exception as e:
        print(f"Failed to read upload_records.json: {e}")
        return []

def get_records_by_annotation_progress(annotation_progress: int, 
                                     records_path: str = "./upload_records.json") -> List[Dict[str, Any]]:
    records = load_upload_records(records_path)
    filtered_records = []
    for record in records:
        json_name = record.get("session_folder").split('/')[-1]
        json_path = os.path.join(record.get("session_folder"), f"{json_name}.json")
        if not os.path.exists(json_path):
            continue
        with open(json_path, 'r', encoding='utf-8') as f:
            merged = json.load(f)
        if len(merged) < 5 and isinstance(record, dict) and record.get("annotation_progress") == annotation_progress:
            continue
        if isinstance(record, dict) and record.get("annotation_progress") == annotation_progress:
            filtered_records.append(record)
    return filtered_records

def update_record_annotation_progress(record_id: str, new_progress: int, 
                                    records_path: str = "./upload_records.json") -> bool:
    if not os.path.exists(records_path):
        print(f"upload_records.json not found: {records_path}")
        return False
    
    try:
        with open(records_path, 'r', encoding='utf-8') as f:
            records = json.load(f)
        
        updated = False
        for record in records:
            if isinstance(record, dict) and record.get("id") == record_id:
                record["annotation_progress"] = new_progress
                updated = True
                break
        
        if not updated:
            print(f"Record with ID {record_id} not found")
            return False
        with open(records_path, 'w', encoding='utf-8') as f:
            json.dump(records, f, indent=2, ensure_ascii=False)
        
        return True
    except Exception as e:
        print(f"Failed to update annotation_progress: {e}")
        return False

def get_static_flag_from_merged(merged_path: str) -> bool:
    if not os.path.exists(merged_path):
        print(f"kp_record_merged.json not found: {merged_path}")
        return False
    try:
        with open(merged_path, 'r', encoding='utf-8') as f:
            merged = json.load(f)
        return merged.get("is_static_object", False)
    except Exception as e:
        print(f"Failed to read static flag: {e}")
        return False

def validate_record_for_optimization(record: Dict[str, Any]) -> bool:
    if not isinstance(record, dict):
        return False
    required_fields = ["id", "session_folder", "object_category"]
    for field in required_fields:
        if not record.get(field):
            print(f"Record missing required field: {field}")
            return False
    session_folder = record.get("session_folder")
    if not session_folder or not os.path.exists(session_folder):
        print(f"Session folder does not exist: {session_folder}")
        return False
    required_files = [
        "video.mp4",
        "obj_org.obj",
        "motion/result.pt",
        "output/obj_poses.json"
    ]
    for file_path in required_files:
        full_path = os.path.join(str(session_folder), file_path)
        if not os.path.exists(full_path):
            print(f"Missing required file: {full_path}")
            return False
    return True