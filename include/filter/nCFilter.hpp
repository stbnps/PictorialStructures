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

#include "partFilter.hpp"

#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/opencv_modules.hpp"

#ifndef NCFILTER_H
#define NCFILTER_H

/*
 * Filter based on normalized correlation.
 */
class NCFilter: public PartFilter {
private:

	cv::Size winSize;

	/*
	 * The training process will calculate the average image of all the
	 * ROIs containing the same part.
	 * Those ROIs will be oriented and scaled so that when calculating the average image
	 * the result will look like the target part.
	 */
	cv::Mat averageImage;

public:

	NCFilter();
	~NCFilter();

	void filter(const cv::Mat &inputImage, cv::Mat &energyImage);

	cv::Size getWindowSize() {
		return winSize;
	}
	void setWindowSize(cv::Size s) {
		winSize = s;
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
