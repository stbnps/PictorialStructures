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

#include "dataset/annotation.hpp"

Annotation::Annotation() {

}

Annotation::~Annotation() {

}

AnnotationRect& Annotation::operator[](std::string name) {
	return annotationRects[name];
}

cv::Mat Annotation::getNegativeSample(std::string name) {

	srand (time(NULL));

	cv::Mat image = cv::imread(datasetFoler + imagePath);

	cv::RotatedRect rect = annotationRects[name].annotation;

	/*
	 * Translate the negative annotation to a place where it doesn't overlap
	 * the positive annotation.
	 */
	int x = rand() % image.cols;
	int y = rand() % image.rows;

	while (rect.boundingRect().contains(cv::Point2i(x, y))) {
		x = rand() % image.cols;
		y = rand() % image.rows;
	}


	rect.center = cv::Point2f(x, y);

	// matrices we'll use
	cv::Mat M, rotated, cropped;
	// get angle and size from the bounding box
	float angle = rect.angle;
	cv::Size rect_size = rect.size;

	// get the rotation matrix
	M = getRotationMatrix2D(rect.center, angle, 1.0);
	// perform the affine transformation
	cv::warpAffine(image, rotated, M, image.size(), cv::INTER_CUBIC);
	// crop the resulting image
	getRectSubPix(rotated, rect_size, rect.center, cropped);

	return cropped;
}

cv::Mat Annotation::getPartROI(std::string name) {
	cv::Mat image = cv::imread(datasetFoler + imagePath);

	cv::RotatedRect rect = annotationRects[name].annotation;
	// matrices we'll use
	cv::Mat M, rotated, cropped;
	// get angle and size from the bounding box
	float angle = rect.angle;
	cv::Size rect_size = rect.size;

	// get the rotation matrix
	M = getRotationMatrix2D(rect.center, angle, 1.0);
	// perform the affine transformation
	cv::warpAffine(image, rotated, M, image.size(), cv::INTER_CUBIC);
	// crop the resulting image
	getRectSubPix(rotated, rect_size, rect.center, cropped);

	return cropped;
}

void Annotation::addAnnotationRect(AnnotationRect rect) {
	annotationRects[rect.getID()] = rect;
}

void Annotation::removeAnnotationRect(std::string name) {
	annotationRects.erase(name);
}

std::string Annotation::getImagePath() {
	return imagePath;
}

void Annotation::setImagePath(std::string imagePath) {
	this->imagePath = imagePath;
}

unsigned Annotation::size() {
	return annotationRects.size();
}

bool Annotation::exists(std::string name) {
	std::map<std::string, AnnotationRect>::iterator it = annotationRects.find(
			name);
	return it != annotationRects.end();
}

bool Annotation::read(cv::FileNode& obj) {
	if (!obj.isMap())
		return false;

	cv::FileNodeIterator it = obj["annotationRects"].begin(), it_end =
			obj["annotationRects"].end();
	annotationRects.clear();
	for (; it != it_end; ++it) {
		cv::FileNode fn = *it;
		AnnotationRect ar;
		ar.read(fn);
		annotationRects[ar.getID()] = ar;
	}

	obj["imagePath"] >> imagePath;

	return true;
}

void Annotation::write(cv::FileStorage& fs, const std::string& objName) const {

	if (!objName.empty())
		fs << objName;

	fs << "{";

	fs << "annotationRects" << "[";

	std::map<std::string, AnnotationRect>::const_iterator it;
	for (it = annotationRects.begin(); it != annotationRects.end(); ++it) {
		(*it).second.write(fs);
	}

	fs << "]";

	fs << "imagePath" << imagePath;

	fs << "}";
}

bool Annotation::load(const std::string& filename, const std::string& objname) {
	cv::FileStorage fs(filename, cv::FileStorage::READ);
	cv::FileNode obj =
			!objname.empty() ? fs[objname] : fs.getFirstTopLevelNode();
	return read(obj);
}

void Annotation::save(const std::string& filename,
		const std::string& objName) const {
	cv::FileStorage fs(filename, cv::FileStorage::WRITE);
	write(fs, objName);
}
