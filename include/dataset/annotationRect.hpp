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

#ifndef ANNOTATIONRECT_H
#define ANNOTATIONRECT_H

#include "opencv2/opencv.hpp"

class AnnotationRect {

public:
	/*
	 * Target rectangle size in pixels.
	 * The filter's window will have this shape.
	 * By storing the original shape, we can calculate deformations.
	 */
	int targetWidth;
	int targetHeight;

	/*
	 * The actual annotation.
	 * Stores position, size and orientation of the part.
	 */
	cv::RotatedRect annotation;

	// Part name
	std::string id;




	AnnotationRect();
	AnnotationRect(std::string, int targetWidth, int targetHeight, cv::RotatedRect annotation);
	~AnnotationRect();



	bool read(cv::FileNode& fn);
	void write(cv::FileStorage& fs, const std::string& objname = "") const;

	bool load(const std::string& filename, const std::string& objname = "");
	void save(const std::string& filename,
			const std::string& objname = "") const;

	std::string getID();

	cv::Point2i getPosition();

	// The scale is measured as the annotation size divided by the target size.
	float getScale();

	float getAngle();

	/*
	 * As shown in:
	 * Yang, Y. & Ramanan, D. Articulated pose estimation with flexible mixtures-of-parts., in 'CVPR' , IEEE, , pp. 1385-1392 (2011).
	 * The use of more (smaller) parts can automatically encode deformations such as foreshortening.
	 * That's why I decided not to support foreshortening for now.
	 */


};

#endif
