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

#ifndef ADABOOSTHOGPARTFILTER_H
#define ADABOOSTHOGPARTFILTER_H

#include "partFilter.hpp"

#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/opencv_modules.hpp"



/*
 * Filter based on the pseudo-probabilistic response of Adaboost classifiers.
 * By taking the response sum of the weak classifiers and
 * normalizing it using the maximum and minimum possible sum an energy image is generated.
 * The input of the Adaboost classifier is the histogram of oriented gradients of each window.
 */
class AdaboostHogPartFilter: public PartFilter {
private:
	// AdaBoost instance
	CvBoost partFilter;

	// This will perform the histogram calculation.
	cv::HOGDescriptor hog;

	cv::Size winStride;

	/*
	 * Maximum and minimum response possible.
	 * They are calculated going through the weak classifiers
	 * held in the CvBoost object
	 */
	double maxSum;
	double minSum;

	/*
	 * OpenCV's implementation of the AdaBoost algorithm uses decision trees
	 * as weak classifiers.
	 * The following set of functions calculates the maximum and minimum cumulative
	 * response of those trees by adding the maximum and minimum output of each tree.
	 */
	void getTreeExtrema(const CvDTreeNode* node, double &max, double &min);
	void getTreeExtrema(const CvDTreeNode* node, double &max, double &min,
			bool &maxSet, bool &minSet);
	void calcMaxMin(double &max, double &min);

	// Calculates the histogram of oriented gradients for all the input images.
	void getHOGs(const std::vector<cv::Mat> &images, cv::Mat &outputHistograms);

public:
	AdaboostHogPartFilter();
	~AdaboostHogPartFilter();

	void filter(const cv::Mat &inputImage, cv::Mat &energyImage);
	void setPartFilter(const char* filename) {
		partFilter.load(filename, "hog");
	}
	cv::Size getWindowSize() {
		return hog.winSize;
	}
	void setWindowSize(cv::Size s) {
		hog.winSize = s;
	}
	void train(const std::vector<cv::Mat> positiveImages,
			const std::vector<cv::Mat> negativeImages);

	bool read(cv::FileNode& fn);
	void write(cv::FileStorage& fs, const std::string& objname = "") const;

	bool load(const std::string& filename, const std::string& objname = "");
	void save(const std::string& filename,
			const std::string& objname = "") const;

};

#endif
