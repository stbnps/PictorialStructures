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

#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/opencv_modules.hpp"

#include "filter/adaboostHogFilter.hpp"

using namespace std;
using namespace cv;

AdaboostHogPartFilter::AdaboostHogPartFilter() :
		winStride(8, 8), maxSum(0), minSum(0) {
	hog.blockSize = Size2f(8, 8);
	hog.cellSize = Size2f(8, 8);
	hog.blockStride = winStride;
	filterType = FilterType::ABH;
}

AdaboostHogPartFilter::~AdaboostHogPartFilter() {

}

// Finds maximum and minimum value of leaf nodes
void AdaboostHogPartFilter::getTreeExtrema(const CvDTreeNode* node, double &max,
		double &min, bool &maxSet, bool &minSet) {
	if (!node->left && !node->right) { // Is a leaf node
		if (maxSet) {
			if (node->value > max) {
				max = node->value;
			}
		} else {
			maxSet = true;
			max = node->value;
		}
		if (minSet) {
			if (node->value < min) {
				min = node->value;
			}
		} else {
			minSet = true;
			min = node->value;
		}
	} else {
		if (node->left) {
			getTreeExtrema(node->left, max, min, maxSet, minSet);
		}
		if (node->right) {
			getTreeExtrema(node->right, max, min, maxSet, minSet);
		}
	}
}

// Gets maximum and minimum response
void AdaboostHogPartFilter::calcMaxMin(double &max, double &min) {
	CvSeq* weakPredictors = partFilter.get_weak_predictors();
	CvSeqReader reader;
	cvStartReadSeq(weakPredictors, &reader);
	int weak_count = weakPredictors->total;
	max = 0;
	min = 0;
	for (int i = 0; i < weak_count; i++) {
		CvBoostTree* wtree;
		const CvDTreeNode* node;
		CV_READ_SEQ_ELEM(wtree, reader);

		node = wtree->get_root();

		double currentMin = 0;
		double currentMax = 0;
		getTreeExtrema(node, currentMax, currentMin);
		max += currentMax;
		min += currentMin;

	}
}

// Overloaded function; this is the one called first. Gets maximum and minimum response
void AdaboostHogPartFilter::getTreeExtrema(const CvDTreeNode* node, double &max,
		double &min) {
	bool minSet = false;
	bool maxSet = false;
	getTreeExtrema(node, max, min, maxSet, minSet);
}

void AdaboostHogPartFilter::filter(const cv::Mat &inputImage,
		cv::Mat &energyImage) {
	vector<float> descriptors;
	hog.compute(inputImage, descriptors, winStride);
	size_t nWindowsX = (inputImage.cols / winStride.width)
			- (hog.winSize.width / winStride.width) + 1;
	size_t nWindowsY = (inputImage.rows / winStride.height)
			- (hog.winSize.height / winStride.height) + 1;

	size_t nWindows = nWindowsX * nWindowsY;
	int hogSize = hog.getDescriptorSize();
	int descriptorSize = descriptors.size();
	nWindows = descriptorSize / hogSize;
	CV_Assert(nWindows == (nWindowsX * nWindowsY));
	Mat descriptorsMat = Mat(descriptors).reshape(1, nWindows);
	energyImage = Mat::ones(Size(nWindowsX + 2, nWindowsY + 2), CV_8UC1) * 255; // +2 because edge pixels aren't detected

	for (size_t x = 0; x < nWindowsX; ++x) {
		for (size_t y = 0; y < nWindowsY; ++y) {
			int descriptorRow = x + y * nWindowsX;
			float windowEnergy = partFilter.predict(
					descriptorsMat.row(descriptorRow), Mat(), Range::all(),
					false, true);

			// Inverted adaboost probability; in the output image, low means good match
			int value = 255 - 255 * (windowEnergy - minSum) / (maxSum - minSum);

			energyImage.at<uchar>(y + 1, x + 1) = value;
		}
	}

	resize(energyImage, energyImage, Size(inputImage.cols, inputImage.rows), 0,
			0, INTER_CUBIC);

}

/*
 * Computes HOG descriptor for a list of images.
 * The output matrix has one descriptor per row.
 */
void AdaboostHogPartFilter::getHOGs(const vector<Mat> &images,
		Mat &outputHistograms) {
	Mat HOGs;
	for (vector<Mat>::const_iterator it = images.begin(); it != images.end();
			++it) {

		Mat currentHistogram;
		vector<float> descriptors;
		hog.compute(*it, descriptors, winStride);

		currentHistogram = Mat(descriptors).clone();
		currentHistogram = currentHistogram.reshape(1, 1);
		outputHistograms.push_back(currentHistogram);

	}
}

void AdaboostHogPartFilter::train(const std::vector<cv::Mat> positiveImages,
		const std::vector<cv::Mat> negativeImages) {

	Mat inputs;
	Mat targets;
	Mat positiveHOGs;
	getHOGs(positiveImages, positiveHOGs);
	Mat negativeHOGs;
	getHOGs(negativeImages, negativeHOGs);
	vconcat(positiveHOGs, negativeHOGs, inputs);

	vector<int> v;
	for (int i = 0; i < positiveHOGs.rows; ++i) {
		v.push_back(1);
	}

	for (int i = 0; i < negativeHOGs.rows; ++i) {
		v.push_back(0);
	}

	Mat(v).copyTo(targets);

	Mat var_types;


	CvBoostParams params(CvBoost::DISCRETE, // boost_type
			600, // weak_count
			0.95, // weight_trim_rate
			2, // max_depth
			false, //use_surrogates
			0 // priors
			);

	partFilter.train(inputs, CV_ROW_SAMPLE, targets, Mat(), Mat(), var_types,
			Mat(), params);
	calcMaxMin(maxSum, minSum);
}

bool AdaboostHogPartFilter::read(cv::FileNode& obj) {
	if (!obj.isMap())
		return false;

	cv::FileNode fn;
	fn = obj["partFilter"];
	CvFileStorage *storage = const_cast<CvFileStorage *>(fn.fs);
	partFilter.read(storage, *fn);
	fn = obj["hog"];
	hog.read(fn);
	obj["maxSum"] >> maxSum;
	obj["minSum"] >> minSum;

	return true;
}

void AdaboostHogPartFilter::write(cv::FileStorage& fs,
		const std::string& objName) const {

	if (!objName.empty())
		fs << objName;

	fs << "{";

	partFilter.write(fs.fs, "partFilter");
	hog.write(fs, "hog");
	fs << "maxSum" << maxSum;
	fs << "minSum" << minSum;

	fs << "}";
}

bool AdaboostHogPartFilter::load(const std::string& filename,
		const std::string& objname) {
	cv::FileStorage fs(filename, cv::FileStorage::READ);
	cv::FileNode obj =
			!objname.empty() ? fs[objname] : fs.getFirstTopLevelNode();
	return read(obj);
}

void AdaboostHogPartFilter::save(const std::string& filename,
		const std::string& objName) const {
	cv::FileStorage fs(filename, cv::FileStorage::WRITE);
	// This will be the root element; it must have a name.
	write(fs,
			!objName.empty() ?
					objName : cv::FileStorage::getDefaultObjectName(filename));
}
