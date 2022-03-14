"""
Current version finds the failed files based on .out file not their .err file
"""

import os
import re
import argparse

def get_failed(root_dir, failed_pattern):
    failed_ids = []
    for f in os.listdir(root_dir):
        if re.match('slurm.*.out', f):
            file_path = os.path.join(root_dir, f)
            with open(file_path) as file: 
                data = file.read()
            if failed_pattern not in data:
                match = re.search("0-([0-9]*).out", f)
                if match:
                    failed_ids.append(match.group(1))
    
    return failed_ids

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Finds the failed files')
    
    parser.add_argument('--root_dir', type=str, default='scripts')
    parser.add_argument('--failed_pattern', type=str, default='steps 100000')
    
    args = parser.parse_args()
    
    failed_ids = get_failed(args.root_dir, args.failed_pattern)
    
    print('Failed ones: ', ','.join(failed_ids))
    print('Number of failed: ', len(failed_ids))


