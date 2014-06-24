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

#include "dataset/dataset.hpp"

Dataset::Dataset() {

}

Dataset::~Dataset() {

}

Annotation& Dataset::operator[](unsigned i) {
	return annotations[i];
}

void Dataset::addAnnotation(Annotation annotation) {
	annotations.push_back(annotation);
}

void Dataset::removeAnnotation(unsigned pos) {
	annotations.erase(annotations.begin() + pos);
}

unsigned Dataset::size() {
	return annotations.size();
}

bool Dataset::read(cv::FileNode& obj) {
	if (!obj.isMap())
		return false;

	cv::FileNodeIterator it = obj["annotations"].begin(), it_end =
			obj["annotations"].end();
	annotations.clear();
	for (; it != it_end; ++it) {
		cv::FileNode fn = *it;
		Annotation a;
		a.read(fn);
		a.datasetFoler = datasetFolder;
		annotations.push_back(a);
	}

	return true;
}

void Dataset::write(cv::FileStorage& fs, const std::string& objName) const {

	if (!objName.empty())
		fs << objName;

	fs << "{";

	fs << "annotations" << "[";

	std::vector<Annotation>::const_iterator it;
	for (it = annotations.begin(); it != annotations.end(); ++it) {
		it->write(fs);
	}

	fs << "]";

	fs << "}";
}

bool Dataset::load(const std::string& filename, const std::string& objname) {

	size_t found = filename.find_last_of("/\\");
	if (found != std::string::npos) {
		datasetFolder = filename.substr(0, found) + "/";
	}
	cv::FileStorage fs(filename, cv::FileStorage::READ);
	cv::FileNode obj =
			!objname.empty() ? fs[objname] : fs.getFirstTopLevelNode();
	return read(obj);
}

void Dataset::save(const std::string& filename,
		const std::string& objName) const {
	cv::FileStorage fs(filename, cv::FileStorage::WRITE);
	// This will be the root element; it must have a name.
	write(fs,
			!objName.empty() ?
					objName : cv::FileStorage::getDefaultObjectName(filename));
}
