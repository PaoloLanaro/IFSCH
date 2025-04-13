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


# Example Usage
If you want to run the model training / testing yourself, the commands below are a starting point

## Combinatorial Object Detection:
- `python canny_test.py --mode evaluate --show --colors red yellow purple`
- `python canny_test.py --colors yellow purple --image ../data/train/images/4.jpg --mode predict`

## U-Net Neural Network:
- First, you'll want to process the data in to the correct format: `python data_prep.py --images ../data/.raw_image --annotation ../data/.segmented_images/merged_project.json --visualize`
- Then you'll want to actually train the model with that data: `python unn.py`
- And finally, you can test the model! `python unn_test.py --model best_model.pth` 
* Note: This is running the model in eval mode, but you can also run it in predict mode. See more by using the `-h` flag
