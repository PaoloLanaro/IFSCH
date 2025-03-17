CS4100 Project Proposal
Members:
Jacob Wu-Chen, Paolo Lanaro

## Table of Contents
- [Problem Description](#problem_description)
- [Algorithms](#algorithms)
- [Results](#results)
- [Links](#links)

## Problem Description
As climbers, we both understand the realm of opportunity for using artificial intelligence in climbing. More specifically, we intend on using image segmentation and classification models such as CNN or ViTs to identify certain types of climbs. Types of climbs can be classified based on characteristics such as how small the holds are, what types of moves the climb requires, how long the climb is in feet, and much more. In this problem, our inputs would just be images of climbs. 

Based on the climb, our models would analyze it and produce outputs such as how difficult the climb is, which part of the climb is most difficult, etc. The idea for this project stems from a lack of “standardization” over climbing grades in general, but specifically from gym to gym. This may seem like an issue because of the requirement for “clean data”, but there’s been research into the difficulty grading and there’s a common difficulty range (average) that will encompass many climbs of a given difficulty. 

Our project idea is that if we can get an AI / ML model to begin to recognize and understand climb difficulties, it could be offered as a sort of “unbiased referee”. You can imagine a world in which every climbing gym uses our model, and suddenly climbing difficulties are standardized over different cities and countries. We want to focus on the recognition and grading of climbs as that allows this idea to branch off further into route move recommendations–a model telling you the “optimal path”– and other forms of analysis including the “crux” or most difficult part of the climb. 

From a computational perspective, we’d receive images (and possibly videos) of routes as input, and we’d output either singular difficulties or a difficulty range with some certainty score. Images would have to be analyzed and an intermediary step would be using those converted data points from the image to then actually make the predictions and general “AI”. 

It’s of special interest to both of us because, as aforementioned, we’re both climbers and have realized that our gym is beginning to grade harder and harder climbs that don’t feel realistic. With something like a model that is able to give a better estimation of the climb, we could get the “real difficulties” of a climb. We’d be able to gather images of climbing by both using existing datasets on Kaggle and Github as well as recruiting our family and friends that climb to take pictures of gyms in other cities, states, and countries. The truth values to train the models would be the somewhat subjective (but on average correct) grades for the climbs and a way to evaluate model performance would be using it against human benchmarks.



## Algorithms
We plan to use image processing AI algorithms–CNN and ViTs–to solve the problem. We’ll use two AI models—CNNs and Vision Transformers (ViTs)—to analyze climbing route images. For the CNN, we’ll start with a simple model that’s already good at recognizing patterns in images. We’ll modify it to do three tasks: predict the climb’s difficulty (like guessing a V3 or V5 grade), identify hold types (jugs, crimps, slopers), and highlight the toughest section using heatmaps. CNNs work well here because they’re great at spotting details like hold shapes and spacing, which are key for grading climbs. For the ViT, we’ll break the image into smaller parts and analyze how they relate to each other (e.g., seeing if holds are spread out or clustered). To improve accuracy, we’ll combine the ViT with the CNN so the model uses both close-up details (from the CNN) and the “big picture” (from the ViT). This hybrid approach helps the AI understand the route’s overall complexity. Both models will be trained on climbing images with known grades and hold types, and we’ll test their accuracy against human climbers’ ratings. We’ll keep the project manageable by starting with a small dataset (100-200 images) and using frameworks like TensorFlow/PyTorch to build the models. If the hybrid ViT-CNN is too complex, we’ll focus on refining the CNN alone.








## Results
The results we would expect would be confidence levels of predictions using our models. This could be in the form of an mse or something similar. We would also expect to produce a segmented image that highlights which parts of the climb are more difficult and which are easier. This is mentioned above. We’ll also use outputs from our Vision Transformer by classifying different parts of the climb and relating them to the difficulty. As we combine the ViT with the CNN, we will be able to analyze the difficulty of climbs and allow the user to visually see the climb split into distinct sections. 

## Links

https://www.sciencedirect.com/science/article/abs/pii/S0164121222000292 [1]

https://dl.acm.org/doi/pdf/10.1145/2071423.2071433 [2]

https://github.com/TheVGLC/TheVGLC/tree/master/Super%20Mario%20Bros%202 

# Resources
# pytorch[https://pytorch.org/]
