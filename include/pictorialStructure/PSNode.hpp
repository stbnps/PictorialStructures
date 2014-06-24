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

#ifndef PSNODE_H
#define PSNODE_H

#include <memory>

#include <vector>
#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/opencv_modules.hpp"
#include "filter/partFilter.hpp"
#include "connection.hpp"

/*
 * Pictorial structure tree node.
 */
class PSNode {

private:

	//	PartFilter *filter;
	/*
	 * Part filter, used for evaluating the part appearance.
	 * Pictorial structures don't impose a particular method for part filtering.
	 * We allow any filter as long as it implements PartFilter.
	 *
	 * Why do I use a shared pointer?
	 * One of the part filters uses OpenCV's AdaBoost implementation.
	 * When this was implemented CvBoost didn't follow the rule of three:
	 *     http://en.wikipedia.org/wiki/Rule_of_three_(C%2B%2B_programming
	 *     http://code.opencv.org/issues/3751
	 *
	 * So even if this class uses PartFilter *filter, deletes it in the destructor,
	 * and follows the rule of three, you could still get a double free error
	 * because CvBoost doesn't provide copy and assignment operations.
	 *
	 * Then, we are stuck with either a memory leak, or the use of shared pointers.
	 */
	std::shared_ptr<PartFilter> filter;

	// Name of the part
	std::string id;

	// Number of scale levels to be evaluated
	int nScaleLevels;
	// Scale factor between consecutive scales, the input image will be downscaled.
	double scaleFactor;
	// Number of orientation levels to be evaluated
	int nRotationLevels;

	/*
	 * True when the connection distribution needs to be updated.
	 * Set to true when the user changed nScaleLevels, scaleFactor or nRotationLevels.
	 */
	bool scaleChanged;

	// Mean of the relative location to its parent. It's expressed on discretized units.
	std::vector<float> means;
	// Standard deviation of the relative location to its parent. It's expressed on discretized units.
	std::vector<float> sDevs;
	// Stores the step of each dimension
	std::vector<float> dimensionScales;

	// Called to update means and sDevs when the scaleChanged is set to true.
	void updateMoments();

	// Minimum energy inverse
	double score;

	/*
	 * Stores where the part is once the model is solved.
	 * The location is expressed in discretized coordinates.
	 */
	std::vector<int> partLocation;

	// The result of translating each children to the expected relative position of this part.
	std::vector<cv::Mat> childrenTranslatedEnergyMatrices;
	// The locations of the parabolas once the distance transform has been applied to childrenTranslatedEnergyMatrices
	std::vector<cv::Mat> childrenTranslatedEnergyLocations;
	// The combined evidence of the part appearance and children translated response.
	cv::Mat partEnergyMatrix;

	/*
	 * Multivariate Normal distribution parameters.
	 * Just save the variances, since we will use a diagonal covariance matrix.
	 * Important note:
	 * Some means are differences between variables (eg: differences in X or Y position)
	 * Others are factors (eg: a part has a scale of twice another part)
	 */
	std::vector<ConnectionData> connectionData; //Needed to recalculate moments when there is a scale change

public:

	PSNode();

	~PSNode();

	/*
	 * Children of the node. Recursive declaration to build a tree.
	 * Made public to ease working with children during training and tree building.
	 */
	std::vector<PSNode> children;

	int getNScaleLevels();
	void setNScaleLevels(int nScaleLevels);

	double getScaleFactor();
	void setScaleFactor(double scaleFactor);

	int getNRotationLevels();
	void setNRotationLevels(int nRotationLevels);

	double getScore();

	void setConnectionData(const std::vector<ConnectionData> &connectionData) {
		this->connectionData = connectionData;
	}

	/*
	 * dimensionOp and dimensionIOp know which type of scaling is performed on each dimension.
	 * dimensionIOp will be used to calculate the discretized means and standard deviations.
	 * dimensionOp will be used to translate discretized locations, to real locations.
	 */
	std::vector<float (PSNode::*)(float a, float b)> dimensionIOp;
	std::vector<float (PSNode::*)(float a, float b)> dimensionOp;

	/*
	 * This is a set of functors used to populate dimensionIOp and dimensionOp
	 * Both dimensionIOp, dimensionOp and the functors are made public since they are
	 * explicitly used during training and retrieval of the solutions.
	 */
	float divide(float dividend, float divisor) {
		return dividend / divisor;
	}

	float logarithm(float input, float base) {
		return log(input) / log(base);
	}

	float multiply(float a, float b) {
		return a * b;
	}

	float raise(float input, float exponent) {
		return pow(input, exponent);
	}

	/*
	 * The pictorial structure model will be solved starting from the current part
	 */
	void solve(const cv::Mat &inputImage, cv::Mat &energyMatrix, bool root =
			true);

	// Just for convenience
	void solve(const cv::Mat &inputImage);

	/*
	 * The parent of the node will call this method telling partLocation is where this part is located.
	 * When this method is called, the node will calculate the location of the children and call propagateSolution
	 * on each of them.
	 * Currently, only one location is propagated, the one with the minimum energy. However, the solution
	 * propagation can start from any point, allowing the sampling of the global energy matrix.
	 */
	void propagateSolution(std::vector<int> partLocation);

	// Set part ID
	void setID(std::string id) {
		this->id = id;
	}

	// Get the location once the model has been solved
	std::vector<int> getPartLocation() {
		return partLocation;
	}

	// Returns vector containing the step on each dimension.
	std::vector<float> getDimensionScales() {
		return dimensionScales;
	}

	// Returns the part ID
	std::string getID() {
		return id;
	}

	// Sets the part filter
	void setPartFilter(std::shared_ptr<PartFilter> filter) {
		this->filter = filter;
	}

	// Trains the part filter
	void trainPartFilter(const std::vector<cv::Mat> positiveImages,
			const std::vector<cv::Mat> negativeImages) {
		filter->train(positiveImages, negativeImages);
	}

	/*
	 * Densely evaluates the input image considering all desired part positions, orientations, and scales.
	 */
	void calcMatchCost(const cv::Mat originalImage, cv::Mat &energyMatrix);

	// Translates the response to the expected location of the parent.
	void translateResponse(const cv::Mat energyMatrix,
			cv::Mat &translatedEnergyMatrix);

	// Performs the Mahalanobis distance transform on translatedEnergyMatrix
	void mahalanobisDistanceTransform(const cv::Mat translatedEnergyMatrix,
			cv::Mat &outputEnergyMatrix, cv::Mat &locations);

	bool read(cv::FileNode& fn);
	void write(cv::FileStorage& fs, const std::string& objname = "") const;

	bool load(const std::string& filename, const std::string& objname = "");
	void save(const std::string& filename,
			const std::string& objname = "") const;

};

#endif

