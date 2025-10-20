"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import argparse
import subprocess
from tqdm import tqdm

def main(args):
    all_substr = []
    for feat in args.feat:
        for category in args.category:
            all_substr.append(f"{category.upper()}_{feat}")
    
    download_line = []
    with open(args.src, 'r') as file:
        for line in file:
            if any(substring in line for substring in all_substr):
                download_line.append(line.strip())

    for line in tqdm(download_line):
        command = f"wget {line}"
        e = subprocess.run(command, shell=True)
        if e != 0:
            print(f"Error downloading {line}")
            exit(1)
        print("Downloaded", line)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True, help="path to the txt file containing the list of videos to download")
    parser.add_argument("--feat", nargs="+", type=str, default=['smplx', 'audio', 'text', 'videos'], choices=['smplx', 'audio', 'text', 'videos'], help="specify the features to download")
    parser.add_argument("--category", nargs="+", type=str, default=['charades', 'daylife', 'dyadic', 'hands', "locomotion", "multiperson", "scenarios"], choices=['charades', 'daylife', 'dyadic', 'hands', "locomotion", "multiperson", "scenarios"], help="specify the category to download")
    args = parser.parse_args()
    # greps through the string and downloads only features that match your arg list.
    main(args)
