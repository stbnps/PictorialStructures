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

#include "dataset/annotationRect.hpp"

AnnotationRect::AnnotationRect() {
	targetWidth = 0;
	targetHeight = 0;
	annotation = cv::RotatedRect();
	id = "";
}

AnnotationRect::AnnotationRect(std::string id, int targetWidth,
		int targetHeight, cv::RotatedRect annotation) {
	this->id = id;
	this->targetWidth = targetWidth;
	this->targetHeight = targetHeight;
	this->annotation = annotation;
}

AnnotationRect::~AnnotationRect() {

}

bool AnnotationRect::read(cv::FileNode& obj) {
	if (!obj.isMap())
		return false;

	cv::FileNodeIterator it;

	cv::Point2f center;
	cv::Size2f size;
	float angle;

	obj["targetWidth"] >> targetWidth;
	obj["targetHeight"] >> targetHeight;
	obj["targetHeight"] >> targetHeight;
	obj["annotationAngle"] >> angle;
	it = obj["annotationCenter"].begin();
	it >> center.x >> center.y;
	it = obj["annotationSize"].begin();
	it >> size.width >> size.height;
	obj["id"] >> id;

	annotation = cv::RotatedRect(center, size, angle);

	return true;
}

void AnnotationRect::write(cv::FileStorage& fs,
		const std::string& objName) const {

	if (!objName.empty())
		fs << objName;

	fs << "{";

	fs << "targetWidth" << targetWidth;
	fs << "targetHeight" << targetHeight;
	fs << "annotationAngle" << annotation.angle;
	fs << "annotationCenter" << annotation.center;
	fs << "annotationSize" << annotation.size;
	fs << "id" << id;

	fs << "}";
}

bool AnnotationRect::load(const std::string& filename,
		const std::string& objname) {
	cv::FileStorage fs(filename, cv::FileStorage::READ);
	cv::FileNode obj =
			!objname.empty() ? fs[objname] : fs.getFirstTopLevelNode();
	return read(obj);
}

void AnnotationRect::save(const std::string& filename,
		const std::string& objName) const {
	cv::FileStorage fs(filename, cv::FileStorage::WRITE);
	write(fs, objName);
}

std::string AnnotationRect::getID() {
	return id;
}

cv::Point2i AnnotationRect::getPosition() {
	return cv::Point2i(annotation.center.x, annotation.center.y);

}

float AnnotationRect::getScale() {
	int widthOffset = annotation.size.width - targetWidth;
	int heightOffset = annotation.size.height - targetHeight;

	float scale = 0;

	if (widthOffset < heightOffset) {
		scale = annotation.size.width / targetWidth;
	} else {
		scale = annotation.size.height / targetHeight;
	}

	return scale;
}

float AnnotationRect::getAngle() {
	return annotation.angle;
}
