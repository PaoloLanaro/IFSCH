CS4100 Project Proposal
Members:
Jacob Wu-Chen, Paolo Lanaro

## Table of Contents
- [Problem Description](#problem_description)
- [Algorithms](#algorithms)
- [Results](#results)
- [Links](#links)

## Problem Description
Creating video game levels can be tedious as it requires time and energy to think of new designs that many just don’t have the time for. Artificial Intelligence has been used in the realm of video games through things like genetic algorithms and evolutionary algorithms, which allow developers to cut down on time doing meaningless tasks. 

As we input more data or have the algorithms run more, they can learn about the game and adapt to solve the given issue. As such, the question that we have decided to face will be creating new levels of differing difficulty, based on user input, and evaluating the validity of such levels. 

As noted in their paper[1] “A co-evolutionary genetic algorithms approach to detect video game bugs”, the authors mention “Hence, utilizing automated agents can optimize and support the playtesting process … where agents can be used to support finding bugs and invalid states, improving the level design and visuals, or enhancing the challenges and fun factor.” 

Genetic algorithms may provide crucial help for under-resourced developers who may not have the time to test and check each game level. We aim to solve the issue of level generation, allowing for levels that are completable, so that unique, never before seen, levels can be generated with no effort. 

The inputs will be previously created, and existing, levels, such as those found on the Video Game Level Corpus and we expect our output to be levels that can be played to their finish goals.


## Algorithms
We will be exploring the usage of genetic algorithms for this application. More broadly, we will be exploring evolutionary algorithms in this context. This concept is by no means new, however, we will be investigating how genetic algorithms can best be applied to creating new video game levels, something that can be quite time-consuming.  

As one paper[2] stated, “we believe that, with proper changes, a similar approach can be used for a generic platform game. The most significant aspect that guided that inspiration is that this game, like many others, has areas represented in a grid.” Genetic algorithms are so versatile because they can be utilized as long as the game state can be represented as a graph. 

Since these games are similar to the ones we discussed in class, such as chess, we can use state space graphs or trees to represent branching factors and moves. Genetic algorithms are also a new form of algorithm for both of us, and we’ve never had experience developing them, so we hope this project allows us to further our understanding of such algorithms.









## Results
The results we expect to show would be evaluations of game levels created by our algorithms. We would plan to use heuristics to evaluate the validity of such game levels to determine if our output levels match the input that we request. It’s also important that our output levels are able to be completed or finished. 

For example, in the case of a Mario level, we want the player to be able to complete the level and not get stuck at some point during the level. Most importantly, we want creative designs that will challenge the player and excite them to use our model. 

## Links

https://www.sciencedirect.com/science/article/abs/pii/S0164121222000292 [1]

https://dl.acm.org/doi/pdf/10.1145/2071423.2071433 [2]

https://github.com/TheVGLC/TheVGLC/tree/master/Super%20Mario%20Bros%202 

# Resources
# pytorch[https://pytorch.org/]
