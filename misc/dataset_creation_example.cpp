#include<iostream>
#include<fstream>
#include <stdio.h>
#include <iomanip>
#include <math.h>
#include "opencv2/opencv.hpp"
#include "annotationRect.hpp"
#include "annotation.hpp"
#include "dataset.hpp"

using namespace std;
using namespace cv;

AnnotationRect getAnnotationRect(string id, float x1, float y1, float x2,
		float y2) {
	const double PI = 3.141592653589793238463;
	string line;
	istringstream in;

	int centerX, centerY;
	centerX = (x1 + x2) / 2;
	centerY = (y1 + y2) / 2;
	int dX = x2 - x1;
	int dY = std::abs(y2 - y1);
	float angle = atan2(dY, dX) * 180 / PI;
	//angle -= 90;
	// On the canonical configuration, the distance between the points is only influenced by the Y displacement.
	float dist = sqrt(dX * dX + dY * dY);

	int targetWidth = 0;
	int targetHeight = 0;
	int annotationWidth = 0;
	int annotationHeight = 0;

	if (id == "leye" || id == "reye") {
		targetWidth = 40;
		targetHeight = 32;
		annotationWidth = dist;
		annotationHeight = dist / 1.25;
	}

	if (id == "leyeoc" || id == "reyeoc") {
		targetWidth = 24;
		targetHeight = 48;
		annotationWidth = dist;
		annotationHeight = dist / 0.5;
	}

	if (id == "lebrow" || id == "rebrow") {
		targetWidth = 32; //24
		targetHeight = 24;
		annotationWidth = dist;
		annotationHeight = dist / 1.3;
	}

	if (id == "sbebrow") {
		targetWidth = 56;
		targetHeight = 32;
		annotationWidth = dist;
		annotationHeight = dist / 1.75;
	}

	if (id == "nose") {
		targetWidth = 40;
		targetHeight = 32;
		annotationWidth = dist;
		annotationHeight = dist/1.25;
	}

	if (id == "mouth") {
		targetWidth = 48;
		targetHeight = 32;
		annotationWidth = dist;
		annotationHeight = dist / 1.5;
	}

	if (id == "llipc" || id == "rlipc") {
		targetWidth = 48;
		targetHeight = 32;
		annotationWidth = dist;
		annotationHeight = dist / 1.5;
	}

	if (id == "chin") {
		targetWidth = 48;
		targetHeight = 32;
		annotationWidth = dist;
		annotationHeight = dist / 1.5;
	}

	AnnotationRect ar(id, targetWidth, targetHeight,
			cv::RotatedRect(cv::Point2f(centerX, centerY),
					cv::Size2f(annotationWidth, annotationHeight), angle));
	return ar;
}

void readPoints(float **points, ifstream &inputFile) {
	string line;
	istringstream in;
	getline(inputFile, line);
	getline(inputFile, line);
	getline(inputFile, line);
	for (int i = 0; i < 20; ++i) {
		float x, y;
		getline(inputFile, line);
		in.str(line);
		in >> x >> y;
		points[i][0] = x;
		points[i][1] = y;
	}
}

/*
 * http://www.bioid.com/index.php?q=downloads/software/bioid-face-database.html
 */

int main() {
	Dataset ds;

	double avg = 0;
	for (int i = 0; i < 1521; ++i) {
		Annotation a;
		AnnotationRect ar;

		stringstream ss;
		ss << setfill('0') << setw(4) << i;
		a.imagePath = "bioid/BioID_" + ss.str() + ".pgm";

		ifstream annotation;
		string annotationFileName = "bioid/bioid_" + ss.str() + ".pts";
		annotation.open(annotationFileName.c_str());

		float **points = new float*[20]();
		for (int i = 0; i < 20; ++i) {
			points[i] = new float[2]();
		}
		readPoints(points, annotation);

		avg += points[2][0] - points[3][0];

		ar = getAnnotationRect("leye", points[1][0] - (points[13][0] - points[1][0]), points[1][1],
				points[1][0] + (points[13][0] - points[1][0] ), points[1][1]);
		a["leye"] = ar;

		ar = getAnnotationRect("reye", points[0][0] - (points[0][0] - points[8][0]), points[0][1],
				points[0][0] + (points[0][0] - points[8][0]), points[0][1]);
		a["reye"] = ar;

		ar = getAnnotationRect("sbebrow", points[0][0], points[0][1],
				points[1][0], points[1][1]);
		a["sbebrow"] = ar;

		ar = getAnnotationRect("lebrow", points[4][0], points[5][1],
				points[5][0], points[5][1]);
		a["lebrow"] = ar;

		ar = getAnnotationRect("rebrow", points[6][0], points[6][1],
				points[7][0], points[6][1]);
		a["rebrow"] = ar;

		float noseCenterX = (points[15][0] + points[16][0]) / 2.0;
		float noseCenterY = (points[15][1] + points[16][1]) / 2.0;
		float noseWidth = std::abs(points[15][0] - points[16][0]);

		ar = getAnnotationRect("nose", noseCenterX - noseWidth, noseCenterY,
				noseCenterX + noseWidth, noseCenterY);
		a["nose"] = ar;

		ar = getAnnotationRect("mouth", points[2][0], points[2][1],
				points[3][0], points[3][1]);
		a["mouth"] = ar;

		ar = getAnnotationRect("chin", points[2][0], points[19][1],
				points[3][0], points[19][1]);
		a["chin"] = ar;

		float mouthCenterX = (points[17][0] + points[18][0]) / 2.0;
		//float mouthCenterY = (points[17][1] + points[18][1]) / 2.0;

		ar = getAnnotationRect("rlipc",
				points[2][0] - (mouthCenterX - points[2][0]), points[2][1],
				points[2][0] + (mouthCenterX - points[2][0]), points[2][1]);
		a["rlipc"] = ar;

		ar = getAnnotationRect("llipc",
				points[3][0] - (points[3][0] - mouthCenterX), points[3][1],
				points[3][0] + (points[3][0] - mouthCenterX), points[3][1]);
		a["llipc"] = ar;

		float rightEyeCenterX = (points[9][0] + points[10][0]) / 2.0;
		float leftEyeCenterX = (points[11][0] + points[12][0]) / 2.0;

		ar = getAnnotationRect("reyeoc",
				points[9][0] - (rightEyeCenterX - points[9][0]), points[9][1],
				points[9][0] + (rightEyeCenterX - points[9][0]), points[9][1]);
		a["reyeoc"] = ar;

		ar = getAnnotationRect("leyeoc",
				points[12][0] - (points[12][0] - leftEyeCenterX), points[12][1],
				points[12][0] + (points[12][0] - leftEyeCenterX),
				points[12][1]);
		a["leyeoc"] = ar;

		ds.addAnnotation(a);

	}

	avg /= 1521;

	std::cout << avg << std::endl;

	ds.save("bioid.ds");

	int i = 0;

//	Mat test = ds[i].getPartROI("torso");
//	imwrite("t.jpg", test);

	for (i = 0; i < 50; ++i) {
		Mat test = ds[i].getPartROI("mouth");
		stringstream ss;
		ss << i;
		imwrite(ss.str() + ".jpg", test);
	}

	cout << ds[i].imagePath << endl;

	return 0;
}
