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

#ifndef ANNOTATION_H
#define ANNOTATION_H

#include <map>
#include "annotationRect.hpp"

class Annotation {

public:
	std::map<std::string, AnnotationRect> annotationRects;
	std::string imagePath;
	std::string datasetFoler;



	Annotation();
	~Annotation();

	AnnotationRect& operator[](std::string name);

	cv::Mat getPartROI(std::string name);

	/*
	 * Returns a negative ROI of the same size and orientation as the reference part.
	 */
	cv::Mat getNegativeSample(std::string name);

	void addAnnotationRect(AnnotationRect rect);
	void removeAnnotationRect(std::string name);

	std::string getImagePath();
	void setImagePath(std::string imagePath);

	unsigned size();

	bool exists(std::string name);

	bool read(cv::FileNode& fn);
	void write(cv::FileStorage& fs, const std::string& objname = "") const;

	bool load(const std::string& filename, const std::string& objname = "");
	void save(const std::string& filename,
			const std::string& objname = "") const;

};

#endif
