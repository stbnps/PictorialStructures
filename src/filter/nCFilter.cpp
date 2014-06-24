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

#include "filter/nCFilter.hpp"

using namespace std;
using namespace cv;

NCFilter::NCFilter() {
	filterType = FilterType::NC;
}

NCFilter::~NCFilter() {

}

void NCFilter::filter(const cv::Mat &inputImage, cv::Mat &energyImage) {

	matchTemplate(inputImage, averageImage, energyImage, CV_TM_CCOEFF_NORMED);
	energyImage.convertTo(energyImage, CV_8UC1, 255, 0);

	energyImage = Mat::ones(Size(energyImage.cols, energyImage.rows),
			energyImage.type()) * 255 - energyImage;
	int borderY = averageImage.rows / 2;
	int borderX = averageImage.cols / 2;
	copyMakeBorder(energyImage, energyImage, borderY, borderY, borderX, borderX,
			BORDER_CONSTANT, 255);

}

void NCFilter::train(const std::vector<cv::Mat> positiveImages,
		const std::vector<cv::Mat> negativeImages) {

	Mat firstSample = positiveImages[0];
	averageImage = Mat::zeros(Size(firstSample.cols, firstSample.rows),
	CV_8UC1);
	double sampleWeight = 1.0 / positiveImages.size();
	std::vector<cv::Mat>::const_iterator it;
	for (it = positiveImages.begin(); it != positiveImages.end(); ++it) {
		Mat tmp;
		cvtColor(*it, tmp, CV_BGR2GRAY);
		averageImage += tmp * sampleWeight;
	}

}

bool NCFilter::read(cv::FileNode& obj) {
	if (!obj.isMap())
		return false;

	cv::FileNodeIterator it;
	it = obj["winSize"].begin();
	int width, height;
	it >> width >> height;
	winSize = cv::Size(width, height);
    obj["averageImage"] >> averageImage;

	return true;
}

void NCFilter::write(cv::FileStorage& fs, const std::string& objName) const {

	if (!objName.empty())
		fs << objName;

	fs << "{";

	fs << "winSize" << winSize;

	fs << "averageImage" << averageImage;

	fs << "}";
}

bool NCFilter::load(const std::string& filename, const std::string& objname) {
	cv::FileStorage fs(filename, cv::FileStorage::READ);
	cv::FileNode obj =
			!objname.empty() ? fs[objname] : fs.getFirstTopLevelNode();
	return read(obj);
}

void NCFilter::save(const std::string& filename,
		const std::string& objName) const {
	cv::FileStorage fs(filename, cv::FileStorage::WRITE);
	// This will be the root element; it must have a name.
	write(fs,
			!objName.empty() ?
					objName : cv::FileStorage::getDefaultObjectName(filename));
}

