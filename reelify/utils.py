import os
import time

def is_file_outdated(file_path, max_age_seconds=3600):
    if os.path.exists(file_path):
        last_modified_time = os.path.getmtime(file_path)
        current_time = time.time()
        age_seconds = current_time - last_modified_time
        return age_seconds > max_age_seconds
    return True
