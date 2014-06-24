//	Copyright (c) 2014, Esteban Pardo Sánchez
//	All rights reserved.
//
//	Redistribution and use in source and binary forms, with or without modification,
//	are permitted provided that the following conditions are met:
//
//	1. Redistributions of source code must retain the above copyright notice, this
//	list of conditions and the following disclaimer.
//
//	2. Redistributions in binary form must reproduce the above copyright notice,
//	this list of conditions and the following disclaimer in the documentation and/or
//	other materials provided with the distribution.
//
//	3. Neither the name of the copyright holder nor the names of its contributors
//	may be used to endorse or promote products derived from this software without
//	specific prior written permission.
//
//	THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
//	ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
//	WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
//	DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
//	ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
//	(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
//	LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
//	ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
//	(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
//	SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#ifndef UTIL_H
#define UTIL_H

#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/opencv_modules.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include "dataset/dataset.hpp"
#include "filter/partFilter.hpp"
#include "filter/adaboostHogFilter.hpp"
#include "filter/nCFilter.hpp"

/*
 * Prints the annotations on top of the input image
 * Useful to visualize results.
 */
void overlayAnnotations(cv::Mat &inputImage, std::vector<AnnotationRect> annotations);

/*
 * Converts a graysale energy image to a rgb energy image.
 * The rgb energy images use color to show the energy response,
 * this way, some details are easier to see.
 */
void energy2RGB(const cv::Mat &energyImage, cv::Mat &outputImage);

/*
 * Displays a X-Y slice of a N dimensional matrix.
 * The slice displayed will be the one which contains
 * the point represented by the location parameter.
 */
void displayNDMat(cv::Mat inputMatrix, std::vector<int> location, bool rgb = false);

/*
 * Factory method to create filters based on FilterType::Type
 */
PartFilter *instantiateFilter(int type);

/*
 * Poulates the indices vector with k random groups.
 * This is useful to evaluate datasets using cross validation.
 */
void kfold(int *indices, int size, int k);

#endif
