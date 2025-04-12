# [AI Project for CS4100](https://github.com/PaoloLanaro/AI-project)
 - Currently a placeholder name for the project
Should we greyscale the images?

https://www.kaggle.com/datasets/tomasslama/indoor-climbing-gym-hold-segmentation 

### Project Proposal can be found [here](./PROPOSAL.md)

## Abstract

## Introduction

## Background

## Related work

## Project Description

## Experiments

## Conclusion


# Commands
- `python data_prep.py --images ../data/.raw_image --annotations ../data/.segmented_images/merged_project.json --visualize`
- `python unn.py`
- `python unn_test.py --model best_model.pth --test_images data/test/images --test_annotations data/test/annotations`
- `python canny_test.py --mode evaluate --show --colors red yellow purple`
- `python canny_test.py --colors yellow purple --image ../data/train/images/4.jpg --mode predict`
