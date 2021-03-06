PictorialStructures
===================

Implementation of the pictorial structures algorithm.

Two part filters have been implemented:
  - Normalized Correlation
  - HOG + Adaboost weak learner response.
  
The supported image transformations are image scaling and rotation. The part filters will be evaluated on all scales and orientations that the user selects.

Note that only the part filters are evaluated on multiple scales. A global multiscale detection has not been implemented, keep that in mind when using the algorithm.

The Kruskal algorithm has been implemented to automatically determine the best set of part relations.

A small dataset module was developed to handle annotations persistance.

Once a pictorial structure has been trained, it can be saved to a YAML file. It also can be read.

This algorithm uses my distance transform implementation:
https://github.com/stbnps/Generalized-Distance-Transform

And my implementation of the Kruskal algorithm:
https://github.com/stbnps/Kruskal-building-block


References:
http://www.cs.cornell.edu/~dph/papers/pict-struct-ijcv.pdf

Dependencies:

OpenCV (tested with 2.4.8)

QT 5.X
