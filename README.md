# Image Obfuscation Benchmark

[TOC]

This repository contains the code to evaluate models on the image obfuscation benchmark, first presented in
[Benchmarking Robustness to Adversarial Image Obfuscations](TODO Arxiv)
(Stimberg et al., 2023).

## Dataset

The dataset consists of 22 obfuscations and the Clean data. 19 obfuscations are training obfuscations and 3 are hold-out obfuscations. All images are central cropped to 224 x 224 and saved as compressed JPEG images. Each obfuscation is applied to each image in the [ILSVRC2012](https://www.image-net.org/challenges/LSVRC/2012/) dataset. For each image, the original_id, label and obfuscation hyper-parameters are stored with it. The dataset can be loaded through the [TensorFlow datasets](https://www.tensorflow.org/datasets) API. Each combination of `train` / `validation` and an obfuscation is its own split, e.g. to load the validation split obfuscated with the `StyleTransfer` obfuscation do

```
import tensorflow_datasets as tfds

ds = tfds.load('obfuscated_imagenet', split='validation_StyleTransfer', data_dir='/path/to/extracted/dataset/')
```

To load multiple obfuscations together, e.g. for training use the [`sample_from_datasets`](https://www.tensorflow.org/api_docs/python/tf/data/Dataset#sample_from_datasets) function.

### Obfuscation Examples

Clean|AdversarialPatches|BackgroundBlurComposition|
-----|------------------|------------------------|
![Clean](images/Clean.png)|![AdversarialPatches](images/AdversarialPatches.png)|![BackgroundBlurComposition](images/BackgroundBlurComposition.png)

ColorNoiseBlocks|ColorPatternOverlay|Halftoning|
----------------|-------------------|----------|
![ColorNoiseBlocks](images/ColorNoiseBlocks.png)|![ColorPatternOverlay](images/ColorPatternOverlay.png)|![Halftoning](images/Halftoning.png)

HighContrastBorder|IconOverlay|ImageOverlay|
------------------|-----------|------------|
![HighContrastBorder](images/HighContrastBorder.png)|![IconOverlay](images/IconOverlay.png)|![ImageOverlay](images/ImageOverlay.png)

Interleave|InvertLines|LineShift|
----------|-----------|---------|
![Interleave](images/Interleave.png)|![InvertLines](images/InvertLines.png)|![LineShift](images/LineShift.png)

LowContrastTriangles|PerspectiveComposition|PerspectiveTransform|
--------------------|----------------------|--------------------|
![LowContrastTriangles](images/LowContrastTriangles.png)|![PerspectiveComposition](images/PerspectiveComposition.png)|![PerspectiveTransform](images/PerspectiveTransform.png)

PhotoComposition|RotateBlocks|RotateImage|
----------------|------------|-----------|
![PhotoComposition](images/PhotoComposition.png)|![RotateBlocks](images/RotateBlocks.png)|![RotateImage](images/RotateImage.png)

StyleTransfer|SwirlWarp|TextOverlay|
-------------|---------|-----------|
![StyleTransfer](images/StyleTransfer.png)|![SwirlWarp](images/SwirlWarp.png)|![TextOverlay](images/TextOverlay.png)

Texturize|WavyColorWarp|
---------|-------------|
![Texturize](images/Texturize.png)|![WavyColorWarp](images/WavyColorWarp.png)

### Download {#dataset-download}

[Train Dataset](TODO) (X GB)
[Eval Dataset](TODO) (X GB)


## Usage Instructions

### Installing

Download the [eval dataset](#dataset-download) and extract it to a folder.

Clone this repository.

```
git clone https://github.com/deepmind/image_obfuscation_benchmark.git
```

Execute `run.sh` to create and activate a virtualenv, install all necessary
dependencies and run a test program to ensure that you can import all the
modules.

```
# Run from the parent directory.
sh image_obfuscation_benchmark/run.sh
```

### Evaluating a model

```
source /tmp/distribution_shift_framework/bin/activate
```

and then run

```
python3 -m image_obfuscation_benchmark.eval.predict \
--dataset_path=/path/to/the/downloaded/dataset/ \
--model_path=https://tfhub.dev/google/imagenet/resnet_v2_50/classification/1 \
--evaluate_obfuscation=Clean \
--normalization=zero_one \
--output_dir=/tmp/
```

Which will write predictions to `/tmp/Clean.csv`. This has to be done for all
obfuscations. Afterwards you run

```
python3 -m image_obfuscation_benchmark.eval.gather_results \
--output_dir=/tmp/
```

which will load all the predictions, calculate the metrics and save them to
`/tmp/metrics.csv`.

### Training a model

We do not supply code to train models on the dataset at the moment but it can be
easily loaded with [tensorflow_datasets](https://github.com/tensorflow/datasets) into any pipeline.

## Ethical Considerations
The specific obfuscations that we use in our benchmark may have the potential to fool automatic filters and therefore increase the amount of harmful con￾tent on digital platforms. To reduce this risk, we decided against releasing the code to create the obfuscations systematically and instead only releasing the precomputed
dataset and code to evaluate on it.

## Citing this work

If you use this code (or any derived code) in your work, please cite the accompanying paper:

```
@article{stimberg2023benchmarking,
  author={},
  title={Benchmarking Robustness to Adversarial Image Obfuscations},
  year={2023},
  url={TODO},
  eprinttype={arXiv}
}
```

## License and Disclaimer

Copyright 2022 DeepMind Technologies Limited.

All software is licensed under the Apache License, Version 2.0 (Apache 2.0);
you may not use this file except in compliance with the License. You may obtain
a copy of the Apache 2.0 license at

[https://www.apache.org/licenses/LICENSE-2.0](https://www.apache.org/licenses/LICENSE-2.0)

Attribution-NonCommercial 2.0 Generic (CC BY-NC 2.0). You may obtain a copy of the CC BY-NC License at:

[https://creativecommons.org/licenses/by/4.0/legalcode](https://creativecommons.org/licenses/by-nc/2.0/legalcode)

You may not use the non-code portions of this file except in compliance with the
CC BY-NC License.

Unless required by applicable law or agreed to in writing, software distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

This is not an official Google product.