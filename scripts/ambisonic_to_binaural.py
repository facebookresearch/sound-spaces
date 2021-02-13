#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import subprocess
from concurrent.futures import ThreadPoolExecutor
import os


def binauralize_ambisonic_irs_with_angle(scene_ambisonic_dir, scene_binaural_dir, angle, exe_file):
    angle_scene_binaural_dir = os.path.join(scene_binaural_dir, str(angle))

    command = [exe_file, "-i", scene_ambisonic_dir, "-o", angle_scene_binaural_dir, "-a", str(angle)]
    ret = subprocess.run(command, check=True, capture_output=True)
    print(ret.stdout, ret.stderr)


def main():
    data_dir = './data'
    ambisonic_dir = os.path.join(data_dir, 'ambisonic_rirs/mp3d')
    binaural_dir = os.path.join(data_dir, 'binaural_rirs/mp3d')

    angles = [0, 90, 180, 270]
    exe_file = './scripts/AmbisonicBinauralizer'
    scenes = os.listdir(ambisonic_dir)
    args = list()
    for scene in scenes:
        scene_ambisonic_dir = os.path.join(ambisonic_dir, scene, 'irs')
        scene_binaural_dir = os.path.join(binaural_dir, scene)
        os.makedirs(scene_binaural_dir, exist_ok=True)

        for angle in angles:
            angle_scene_binaural_dir = os.path.join(scene_binaural_dir, str(angle))
            if os.path.exists(angle_scene_binaural_dir) and len(os.listdir(scene_ambisonic_dir)) == len(
                    os.listdir(angle_scene_binaural_dir)):
                continue
            print(angle_scene_binaural_dir)
            args.append((scene_ambisonic_dir, scene_binaural_dir, angle, exe_file))
    with ThreadPoolExecutor(max_workers=160) as executor:
        executor.map(binauralize_ambisonic_irs_with_angle, *[[arg[i] for arg in args] for i in range(len(args[0]))])


if __name__ == '__main__':
    main()
