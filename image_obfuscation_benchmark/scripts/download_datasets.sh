#!/bin/bash
# This script uses curl to download all the obfuscation subsets. It tries to
# resume downloads if interrupted.
#
# Basic usage: ./download_datasets.sh (train|valid|all) ${OUTPUT_DIR}

set -e

OBFUSCATIONS=("Clean"
              "AdversarialPatches"
              "BackgroundBlurComposition"
              "ColorNoiseBlocks"
              "ColorPatternOverlay"
              "Halftoning"
              "HighContrastBorder"
              "IconOverlay"
              "ImageOverlay"
              "Interleave"
              "InvertLines"
              "LineShift"
              "LowContrastTriangles"
              "PerspectiveComposition"
              "PerspectiveTransform"
              "PhotoComposition"
              "RotateBlocks"
              "RotateImage"
              "StyleTransfer"
              "SwirlWarp"
              "TextOverlay"
              "Texturize"
              "WavyColorWarp")

BASEPATH="https://storage.googleapis.com/dm_image_obfuscation_benchmark"

if [[ $# -ne 2 ]]; then
    echo "Illegal number of parameters"
    echo "Correct usage: ./download_datasets.sh (train|valid|all) output_dir."
    exit 2
fi

case $1 in
  train | training)
  echo "Downloading all training datasets."
  subsets=("train")
  ;;

  valid | validation)
  echo "Downloading all validation datasets."
  subsets=("validation")
  ;;

  all)
  echo "Downloading all training and validation datasets."
  subsets=("train" "validation")
  ;;

  *)
  echo "Unknown subset $1. "
  echo "Allowed values are (train|training|valid|validation|all)."
  exit 2
  ;;

esac

output_path=$2
for filename in "dataset_info.json" "features.json"; do
  curl --continue-at - -o ${output_path}/${filename} ${BASEPATH}/${filename}
done
for subset in ${subsets[@]}; do
  for obfuscation in ${OBFUSCATIONS[@]}; do
    filename="${subset}_${obfuscation}.tar"
    echo "Downloading ${filename}..."
    curl --continue-at - -o ${output_path}/${filename} ${BASEPATH}/${filename}
  done
done
