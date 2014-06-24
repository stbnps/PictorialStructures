PictorialStructures
===================

Implementation of the pictorial structures algorithm.

Two part filters have been implemented:
  - Normalized Correlation
  - HOG + Adaboost weak learner response.
  
The supported image transformations are image scaling and rotation. The part filters will be evaluated on all scales and orientations that the user selects.

The Kruskal algorithm has been implemented to automatically determine the best set of part relations.

A small dataset library was developed to handle annotations persistance.

Once a pictorial structure has been trained, it can be saved to a YAML file. It also can be read.


References:
http://www.cs.cornell.edu/~dph/papers/pict-struct-ijcv.pdf

Dependencies:
OpenCV (tested with 2.4.8)
QT 5.X
