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

#ifndef PICTORIALSTRUCTURE_H_
#define PICTORIALSTRUCTURE_H_

#include <vector>
#include <map>
#include "opencv2/opencv.hpp"
#include "dataset/annotation.hpp"
#include "dataset/annotationRect.hpp"
#include "dataset/dataset.hpp"
#include "PSNode.hpp"
#include "kruskal/kruskal.hpp"
#include "connection.hpp"

class KruskalNode: public Node {

public:
	KruskalNode(std::string name) :
			name(name) {

	}

	std::string name;

};

class KruskalEdge: public Edge {

public:
	float w;

	KruskalEdge(Node *a, Node *b, float w) {
		this->a = a;
		this->b = b;
		this->w = w;
	}

	float getWeight() {
		return w;
	}

};

class PictorialStructure {
private:

	/*
	 * The actual pictorial structure tree.
	 * When PictorialStructure::train is called, this tree will be built.
	 * When PictorialStructure::detect is called, the input image will be passed to the tree.
	 */
	PSNode psTree;

	// A reference to the dataset.
	Dataset dataset;

	/*
	 * Assuming all annotations on the dataset have the same annotations, builds a map used to
	 * find out which parts are connected.
	 * The output is a map containing an array of connection parameters.
	 */
	std::map<std::string, std::map<std::string, std::vector<ConnectionData> > > buildConnectionMap(
			Dataset dataset);

	/*
	 * Models all connections as multivariate normal distributions with diagonal covariance matrix.
	 */
	std::map<std::string, std::map<std::string, ConnectionDataDistribution> > calcConnDistribution(
			std::map<std::string,
					std::map<std::string, std::vector<ConnectionData> > > connectionMap);

	/*
	 * Calculates the energy of the connections. This will be used as the input of the Kruskal algorithm.
	 */
	std::map<std::string, std::map<std::string, double> > calcConnEnergy(
			std::map<std::string,
					std::map<std::string, std::vector<ConnectionData> > > connectionMap,
			std::map<std::string,
					std::map<std::string, ConnectionDataDistribution> > connDistribution);

	/*
	 * Decide which parts are connected using Kruskal's algorithm.
	 * Filters the connection distribution map, returning only the connections that were selected.
	 */
	std::map<std::string, std::map<std::string, ConnectionDataDistribution> > findConnectedParts(
			std::map<std::string, std::map<std::string, double> > connEnergy,
			std::map<std::string,
					std::map<std::string, ConnectionDataDistribution> > connDistribution);

	// Searchs the Pictorial structure tree for a part which name is key.
	PSNode *searchTree(PSNode *psTree, std::string key);

	/*
	 * Some methods to populate each tree node.
	 */
	void setNodeDistribution(PSNode &newNode, ConnectionDataDistribution cdd);
	void setNodeOperations(PSNode &newNode);
	void setNodeConnectionData(PSNode &newNode,
			const std::vector<ConnectionData> &connectionData);
	std::vector<ConnectionData> invertConnectionData(
			std::vector<ConnectionData> d);

	/*
	 * This method will create a new tree using the minimum spanning tree edges calculated using Kruskal's algorithm.
	 * mstConnnDistribution are the actual edges with the real distribution data.
	 * connectionMap stores the connection data of all possible edges. This argument is passed since each node
	 * on the pictorial structure tree stores the connection training data to allow rediscretization.
	 */
	PSNode buildPSTree(
			std::map<std::string,
					std::map<std::string, ConnectionDataDistribution> > mstConnnDistribution,
			std::map<std::string,
					std::map<std::string, std::vector<ConnectionData> > > connectionMap);

	// Prepares the training data and calls PictorialStructure::trainNodes
	void trainAppearanceFilters(PSNode &psTree, Dataset dataset);

	// Trains the part filters using the ROIs in the dataset.
	void trainNodes(PSNode &psTree,
			std::map<std::string, std::vector<cv::Mat> > &trainingImages);

	// Inverts the connection data.
	ConnectionDataDistribution invertDistribution(ConnectionDataDistribution d);

	// Goes through the Pictorial Structure tree filling the AnnotationRect vector
	void retrieveConfiguration(PSNode &psTree,
			std::vector<AnnotationRect> &annotationRects);

	void retrieveRect(PSNode &psTree, AnnotationRect &annotationRect);

	int filterType;

	bool trained;

public:

	PictorialStructure();

	virtual ~PictorialStructure();

	void train(Dataset dataset);

	bool isTrained() {
		return trained;
	}

	std::vector<AnnotationRect> detect(cv::Mat inputImage, double *score = 0);

	int getFilterType();
	void setFilterType(int filterType);

	int getNScaleLevels();
	void setNScaleLevels(int nScaleLevels);

	double getScaleFactor();
	void setScaleFactor(double scaleFactor);

	int getNRotationLevels();
	void setNRotationLevels(int nRotationLevels);

	bool read(cv::FileNode& fn);
	void write(cv::FileStorage& fs, const std::string& objname = "") const;

	bool load(const std::string& filename, const std::string& objname = "");
	void save(const std::string& filename,
			const std::string& objname = "") const;
};

#endif
