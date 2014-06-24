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

#include "util/util.hpp"

void overlayAnnotations(cv::Mat &inputImage,
		std::vector<AnnotationRect> annotations) {
	std::vector<AnnotationRect>::const_iterator it;
	for (it = annotations.begin(); it != annotations.end(); ++it) {
		cv::Point2f vertices[4];
		it->annotation.points(vertices);
		for (int i = 0; i < 4; i++)
			cv::line(inputImage, vertices[i], vertices[(i + 1) % 4],
					cv::Scalar(0, 255, 0), 2);
	}

}

void energy2RGB(const cv::Mat &energyImage, cv::Mat &outputImage) {
	std::vector<cv::Mat> channels;

	cv::Mat o = cv::Mat::ones(cv::Size(energyImage.cols, energyImage.rows),
			energyImage.type()) * 255;

	cv::Mat energyNormalized;
	normalize(energyImage, energyNormalized, 0, 255, cv::NORM_MINMAX);
	channels.push_back(energyNormalized);
	channels.push_back(o);
	channels.push_back(o);

	cv::Mat resultHSV;
	cv::merge(channels, resultHSV);

	cv::cvtColor(resultHSV, outputImage, CV_HSV2BGR);
}

void displayNDMat(cv::Mat inputMatrix, std::vector<int> location, bool rgb) {
	cv::Mat res = cv::Mat(inputMatrix.size[inputMatrix.dims - 2], inputMatrix.size[inputMatrix.dims - 1], CV_8UC1);

	memcpy(res.data,
			inputMatrix.data + location[1] * inputMatrix.step[1]
					+ location[0] * inputMatrix.step[0],
					inputMatrix.size[inputMatrix.dims - 2] * inputMatrix.size[inputMatrix.dims - 1] * sizeof(uchar));

	cv::namedWindow("Result", 0);
	if (rgb) {
		energy2RGB(res, res);
	}
	for (;;) {
		cv::imshow("Result", res);
		if (cv::waitKey(30) >= 0)
			break;

	}
}

PartFilter *instantiateFilter(int type) {
	switch (type) {
	case FilterType::ABH:
		return new AdaboostHogPartFilter();
	case FilterType::NC:
			return new NCFilter();
	default:
		return new AdaboostHogPartFilter();
	}
	return 0;
}

void shuffleArray(int* inputVector, int size)
{
	int n = size;
	while (n > 1)
	{
		int k = rand()%n;
		n--;
		int tmp = inputVector[n];
		inputVector[n] = inputVector[k];
		inputVector[k] = tmp;
	}
}

void kfold(int *indices, int size, int k)
{
	float inc=(float)k/size;

	for (int i=0;i<size;i++)
		indices[i]=ceil((i+0.9)*inc)-1;

	shuffleArray(indices,size);
}
