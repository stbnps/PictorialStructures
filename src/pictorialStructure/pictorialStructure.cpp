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

#include "pictorialStructure/pictorialStructure.hpp"

#include "opencv2/opencv.hpp"

#include "opencv2/core/core.hpp"

#include <cmath>

#include "util/util.hpp"

PictorialStructure::PictorialStructure() {

	filterType = FilterType::ABH;
	trained = false;

}

PictorialStructure::~PictorialStructure() {
}

std::map<std::string, std::map<std::string, std::vector<ConnectionData> > > PictorialStructure::buildConnectionMap(
		Dataset dataset) {

	std::map<std::string, std::map<std::string, std::vector<ConnectionData> > > map;
	/*
	 * Stores all possible connections that will be used to build the connection map.
	 * This lookup map wont have any reflexive nor reciprocal connection.
	 */
	std::vector<std::pair<std::string, std::string> > referenceConnections;

	Annotation annotation = dataset[0];
	std::map<std::string, AnnotationRect>::const_iterator it1;
	std::map<std::string, AnnotationRect>::const_iterator it2;
	for (it1 = annotation.annotationRects.begin();
			it1 != annotation.annotationRects.end(); ++it1) {
		for (it2 = it1; it2 != annotation.annotationRects.end(); ++it2) {
			if (it1->first == it2->first) {
				continue;
			}
			referenceConnections.push_back(
					std::make_pair(it1->first, it2->first));
		}
	}

	/*
	 * Go through the whole dataset storing connection parameters for the previously calculated connections.
	 */
	for (unsigned i = 0; i < dataset.size(); ++i) {
		Annotation currentAnnotation = dataset[i];

		std::vector<std::pair<std::string, std::string> >::const_iterator it_refConn;
		for (it_refConn = referenceConnections.begin();
				it_refConn != referenceConnections.end(); ++it_refConn) {
			ConnectionData c;
			c.dX = currentAnnotation[it_refConn->second].getPosition().x
					- currentAnnotation[it_refConn->first].getPosition().x;
			c.dY = currentAnnotation[it_refConn->second].getPosition().y
					- currentAnnotation[it_refConn->first].getPosition().y;
			c.dScale = currentAnnotation[it_refConn->second].getScale()
					/ currentAnnotation[it_refConn->first].getScale();
			c.dAngle = currentAnnotation[it_refConn->second].getAngle()
					- currentAnnotation[it_refConn->first].getAngle();
			map[it_refConn->first][it_refConn->second].push_back(c);

		}

	}

	return map;
}

std::map<std::string, std::map<std::string, ConnectionDataDistribution> > PictorialStructure::calcConnDistribution(
		std::map<std::string,
				std::map<std::string, std::vector<ConnectionData> > > connectionMap) {
	std::map<std::string, std::map<std::string, ConnectionDataDistribution> > map;

	std::map<std::string, std::map<std::string, std::vector<ConnectionData> > >::const_iterator it1;
	for (it1 = connectionMap.begin(); it1 != connectionMap.end(); ++it1) {

		std::map<std::string, std::vector<ConnectionData> >::const_iterator it2;
		for (it2 = it1->second.begin(); it2 != it1->second.end(); ++it2) {
			std::vector<ConnectionData>::const_iterator it3;
			std::vector<int> dX;
			std::vector<int> dY;
			std::vector<float> dScale;
			std::vector<float> dAngle;
			for (it3 = it2->second.begin(); it3 != it2->second.end(); ++it3) {
				dX.push_back(it3->dX);
				dY.push_back(it3->dY);
				dScale.push_back(it3->dScale);
				dAngle.push_back(it3->dAngle);
			}
			ConnectionDataDistribution currentDistribution;

			cv::Scalar mX;
			cv::Scalar sdX;
			cv::meanStdDev(dX, mX, sdX);
			currentDistribution.mX = mX[0];
			currentDistribution.sdX = sdX[0];

			cv::Scalar mY;
			cv::Scalar sdY;
			cv::meanStdDev(dY, mY, sdY);
			currentDistribution.mY = mY[0];
			currentDistribution.sdY = sdY[0];

			cv::Scalar mScale;
			cv::Scalar sdScale;
			cv::meanStdDev(dScale, mScale, sdScale);
			currentDistribution.mScale = mScale[0];
			currentDistribution.sdScale = sdScale[0];

			cv::Scalar mAngle;
			cv::Scalar sdAngle;
			cv::meanStdDev(dAngle, mAngle, sdAngle);
			currentDistribution.mAngle = mAngle[0];
			currentDistribution.sdAngle = sdAngle[0];

			map[it1->first][it2->first] = currentDistribution;

		}

	}

	return map;
}

std::map<std::string, std::map<std::string, double> > PictorialStructure::calcConnEnergy(
		std::map<std::string,
				std::map<std::string, std::vector<ConnectionData> > > connectionMap,
		std::map<std::string, std::map<std::string, ConnectionDataDistribution> > connDistribution) {

	std::map<std::string, std::map<std::string, double> > map;

	std::map<std::string, std::map<std::string, std::vector<ConnectionData> > >::iterator it1;
	std::map<std::string, std::map<std::string, double> >::iterator it2;

	// First calculate the quality of each connection
	// For each node in one side of a connection
	for (it1 = connectionMap.begin(); it1 != connectionMap.end(); ++it1) {
		std::map<std::string, std::vector<ConnectionData> >::iterator it3;
		// For each node connected with the previous one
		for (it3 = it1->second.begin(); it3 != it1->second.end(); ++it3) {

			/*
			 * The more variance, the more energy.
			 * Thats because the more the relative location varies,
			 * the less probability of finding a part in that exact relative location.
			 * Since the pictorial structures algorithm works best when the connection
			 * between parts is strong, the connections with low variance will be
			 * top candidates.
			 */
			double energy = connDistribution[it1->first][it3->first].sdX
					+ connDistribution[it1->first][it3->first].sdY
					+ connDistribution[it1->first][it3->first].sdScale
					+ connDistribution[it1->first][it3->first].sdAngle;
			map[it1->first][it3->first] = energy / 4.0;

		}
	}

	return map;
}

ConnectionDataDistribution PictorialStructure::invertDistribution(
		ConnectionDataDistribution d) {
	ConnectionDataDistribution cdd;

	cdd.mX = -d.mX;
	cdd.sdX = d.sdX;
	cdd.mY = -d.mY;
	cdd.sdY = d.sdY;
	cdd.mScale = 1.0 / d.mScale;
	cdd.sdScale = d.sdScale;
	cdd.mAngle = -d.mAngle;
	cdd.sdAngle = d.sdAngle;

	return cdd;
}

std::map<std::string, std::map<std::string, ConnectionDataDistribution> > PictorialStructure::findConnectedParts(
		std::map<std::string, std::map<std::string, double> > connEnergy,
		std::map<std::string, std::map<std::string, ConnectionDataDistribution> > connDistribution) {

	std::map<std::string, std::map<std::string, ConnectionDataDistribution> > map;

	std::vector<Node *> nodes;
	std::vector<Edge *> edges;
	std::map<std::string, KruskalNode*> nodesMap; // Just to have an easy way of searching them

	std::map<std::string, std::map<std::string, double> >::const_iterator it1;
	std::map<std::string, double>::const_iterator it2;

	// Create nodes
	// The way we built the connection map guarantees the first node is connected with all the rest.
	it1 = connEnergy.begin();
	KruskalNode *n = new KruskalNode(it1->first);
	nodes.push_back(n);
	nodesMap[it1->first] = n;
	for (it2 = it1->second.begin(); it2 != it1->second.end(); ++it2) {
		KruskalNode *n = new KruskalNode(it2->first);
		nodes.push_back(n);
		nodesMap[it2->first] = n;
	}

	// Create edges
	for (it1 = connEnergy.begin(); it1 != connEnergy.end(); ++it1) {
		for (it2 = it1->second.begin(); it2 != it1->second.end(); ++it2) {
			KruskalNode *a = nodesMap[it1->first];
			KruskalNode *b = nodesMap[it2->first];
			Edge *e = new KruskalEdge(a, b, it2->second);
			edges.push_back(e);
		}
	}

	Kruskal k(nodes, edges);
	k.solve();

	/*
	 * Kruskal's algorithm works in a undirected graph.
	 * Because of that, we have to build a tree from Kruskal's edges.
	 * Note that a naive implementation of the following could return a tree with
	 * more than one root.
	 * The pictorial structures algorithm works on trees with a single root;
	 * the following algorithm will have that in mind.
	 */
	std::vector<Edge *> mst = k.getEdges();
	std::vector<Edge *>::iterator it = mst.begin();
	KruskalNode *a = (KruskalNode *) (*it)->a;
	KruskalNode *b = (KruskalNode *) (*it)->b;
	map[a->name][b->name] = connDistribution[a->name][b->name];
	mst.erase(mst.begin());
	while (!mst.empty()) {
		for (it = mst.begin(); it != mst.end(); it++) {
			bool mstVectorChanged = false;
			KruskalNode *a = (KruskalNode *) (*it)->a;
			KruskalNode *b = (KruskalNode *) (*it)->b;

			std::map<std::string,
					std::map<std::string, ConnectionDataDistribution> >::iterator mapIterator;
			std::map<std::string, ConnectionDataDistribution>::iterator subMapIterator;

			for (mapIterator = map.begin(); mapIterator != map.end();
					++mapIterator) {
				for (subMapIterator = mapIterator->second.begin();
						subMapIterator != mapIterator->second.end();
						++subMapIterator) {

					// Adds a branch from the root of the segment
					if (a->name == mapIterator->first) {
						map[a->name][b->name] =
								connDistribution[a->name][b->name];
						mst.erase(it);
						mstVectorChanged = true;
						break;
					}

					// Adds a branch from the child of the segment
					if (a->name == subMapIterator->first) {
						map[a->name][b->name] =
								connDistribution[a->name][b->name];
						mst.erase(it);
						mstVectorChanged = true;
						break;
					}

					// Adds a node on top of the root
					if (b->name == mapIterator->first) {
						map[a->name][b->name] =
								connDistribution[a->name][b->name];
						mst.erase(it);
						mstVectorChanged = true;
						break;
					}

					// This is the case when multiple roots would be created
					if (b->name == subMapIterator->first) {
						map[b->name][a->name] = invertDistribution(
								connDistribution[a->name][b->name]);
						mst.erase(it);
						mstVectorChanged = true;
						break;
					}

				}
				if (mstVectorChanged) {
					break;
				}
			}
			if (mstVectorChanged) {
				break;
			}
		}
	}

	// Free memory
	for (std::vector<Node *>::iterator it = nodes.begin(); it != nodes.end();
			++it) {
		KruskalNode *kn = (KruskalNode *) (*it);
		free(kn);
	}

	for (std::vector<Edge *>::iterator it = edges.begin(); it != edges.end();
			++it) {
		KruskalEdge *ke = (KruskalEdge *) (*it);
		free(ke);

	}

	return map;
}

PSNode *PictorialStructure::searchTree(PSNode *psTree, std::string key) {
	if (psTree->getID() == key) {
		return psTree;
	} else {
		std::vector<PSNode> &children = psTree->children;
		std::vector<PSNode>::iterator it;
		for (it = children.begin(); it != children.end(); ++it) {
			PSNode &currentChild = *it;
			PSNode *found = searchTree(&currentChild, key);
			if (found != 0) {
				return found;
			}
		}
		return 0;
	}
}

void PictorialStructure::setNodeOperations(PSNode &newNode) {

	newNode.dimensionIOp.clear();
	newNode.dimensionIOp.push_back(&PSNode::divide);
	newNode.dimensionIOp.push_back(&PSNode::logarithm);
	newNode.dimensionIOp.push_back(&PSNode::divide);
	newNode.dimensionIOp.push_back(&PSNode::divide);

	newNode.dimensionOp.clear();
	newNode.dimensionOp.push_back(&PSNode::multiply);
	newNode.dimensionOp.push_back(&PSNode::raise);
	newNode.dimensionOp.push_back(&PSNode::multiply);
	newNode.dimensionOp.push_back(&PSNode::multiply);

}

void PictorialStructure::setNodeConnectionData(PSNode &newNode,
		const std::vector<ConnectionData> &connectionData) {

	newNode.setConnectionData(connectionData);

}

std::vector<ConnectionData> PictorialStructure::invertConnectionData(
		std::vector<ConnectionData> d) {
	std::vector<ConnectionData> id;

	std::vector<ConnectionData>::const_iterator it;
	for (it = d.begin(); it != d.end(); ++it) {
		ConnectionData cd;
		cd.dX = -it->dX;
		cd.dY = -it->dY;
		cd.dScale = 1.0 / it->dScale;
		cd.dAngle = -it->dAngle;
		id.push_back(cd);
	}

	return id;
}

PSNode PictorialStructure::buildPSTree(
		std::map<std::string, std::map<std::string, ConnectionDataDistribution> > mstConnnDistribution,
		std::map<std::string,
				std::map<std::string, std::vector<ConnectionData> > > connectionMap) {

	PSNode psTree;
	std::vector<std::string> insertedNodes;

	// The root doesn't store the relative position to its parent (it doesn't have parent)
	psTree.setID(mstConnnDistribution.begin()->first);
	setNodeOperations(psTree);
	insertedNodes.push_back(mstConnnDistribution.begin()->first);

	std::map<std::string, std::map<std::string, ConnectionDataDistribution> >::const_iterator it1;
	std::map<std::string, ConnectionDataDistribution>::const_iterator it2;

	/*
	 * Nodes in the second level of the map are children of those in the first level.
	 * Hang nodes with "it2->second" as data from each it1 (parent).
	 */
	bool treeChanged = true;
	while (treeChanged) {
		treeChanged = false;
		for (it1 = mstConnnDistribution.begin();
				it1 != mstConnnDistribution.end(); ++it1) {
			for (it2 = it1->second.begin(); it2 != it1->second.end(); ++it2) {
				bool missingChild = std::find(insertedNodes.begin(),
						insertedNodes.end(), it1->first) != insertedNodes.end()
						&& std::find(insertedNodes.begin(), insertedNodes.end(),
								it2->first) == insertedNodes.end();
				bool missingParent = std::find(insertedNodes.begin(),
						insertedNodes.end(), it1->first) == insertedNodes.end()
						&& std::find(insertedNodes.begin(), insertedNodes.end(),
								it2->first) != insertedNodes.end();

				if (missingChild) {
					PSNode child;
					child.setID(it2->first);
					setNodeOperations(child);

					std::map<std::string,
							std::map<std::string, std::vector<ConnectionData> > >::const_iterator searchIt1;
					std::map<std::string, std::vector<ConnectionData> >::const_iterator searchIt2;
					searchIt1 = connectionMap.find(it1->first);
					searchIt2 = connectionMap[it1->first].find(it2->first);
					if (searchIt1 != connectionMap.end()
							&& searchIt2 != connectionMap[it1->first].end()) {
						setNodeConnectionData(child,
								connectionMap[it1->first][it2->first]);
					} else {
						// The distribution has been inverted
						std::vector<ConnectionData> invertedData =
								invertConnectionData(
										connectionMap[it2->first][it1->first]);
						setNodeConnectionData(child, invertedData);
					}

					// Look for the parent node, it shouldn't be null
					PSNode *parentNode = searchTree(&psTree, it1->first);
					parentNode->children.push_back(child);

					insertedNodes.push_back(it2->first);
					treeChanged = true;
				} else if (missingParent) {
					PSNode parent;
					parent.setID(it1->first);

					setNodeOperations(parent);

					// Look for the child node, it shouldn't be null
					PSNode *childNode = searchTree(&psTree, it2->first);

					// Find out if we really are creating a new root
					if (childNode->getID() == psTree.getID()) {
						setNodeOperations(*childNode);

						std::map<std::string,
								std::map<std::string,
										std::vector<ConnectionData> > >::const_iterator searchIt1;
						std::map<std::string, std::vector<ConnectionData> >::const_iterator searchIt2;
						searchIt1 = connectionMap.find(it1->first);
						searchIt2 = connectionMap[it1->first].find(it2->first);
						if (searchIt1 != connectionMap.end()
								&& searchIt2
										!= connectionMap[it1->first].end()) {
							setNodeConnectionData(*childNode,
									connectionMap[it1->first][it2->first]);
						} else {
							// The distribution has been inverted
							std::vector<ConnectionData> invertedData =
									invertConnectionData(
											connectionMap[it2->first][it1->first]);
							setNodeConnectionData(*childNode, invertedData);
						}

						parent.children.push_back(*childNode);

						psTree = parent;
					} else { // This node wont be the parent in the end

						std::map<std::string,
								std::map<std::string,
										std::vector<ConnectionData> > >::const_iterator searchIt1;
						std::map<std::string, std::vector<ConnectionData> >::const_iterator searchIt2;
						searchIt1 = connectionMap.find(it1->first);
						searchIt2 = connectionMap[it1->first].find(it2->first);
						if (searchIt1 != connectionMap.end()
								&& searchIt2
										!= connectionMap[it1->first].end()) {
							std::vector<ConnectionData> invertedData =
									invertConnectionData(
											connectionMap[it1->first][it2->first]);
							setNodeConnectionData(parent, invertedData);
						} else {
							// The distribution has been inverted
							setNodeConnectionData(parent,
									connectionMap[it2->first][it1->first]);
						}

						childNode->children.push_back(parent);
					}

					insertedNodes.push_back(it1->first);
					treeChanged = true;
				}

			}
		}

	}

	return psTree;
}

void PictorialStructure::trainNodes(PSNode &psTree,
		std::map<std::string, std::vector<cv::Mat> > &trainingImages) {

	/*
	 * trainingImages stores two vectors per part.
	 * One vector contains positive images, and the other negative ones.
	 * This is because the size of the samples
	 * must be the same for all the samples of the same class.
	 */

	PartFilter *filter = instantiateFilter(filterType);
	/*
	 * All images should have the same size at this point.
	 * Take the descriptor size from the first image.
	 */
	cv::Size sz(trainingImages[psTree.getID()][0].cols,
			trainingImages[psTree.getID()][0].rows);
	filter->setWindowSize(sz);
	psTree.setPartFilter(std::shared_ptr<PartFilter>(filter));

	psTree.trainPartFilter(trainingImages[psTree.getID()],
			trainingImages[psTree.getID() + "_negative"]);
	for (unsigned i = 0; i < psTree.children.size(); ++i) {
		trainNodes(psTree.children[i], trainingImages);
	}
}

void PictorialStructure::trainAppearanceFilters(PSNode &psTree,
		Dataset dataset) {

	std::map<std::string, std::vector<cv::Mat> > trainingImages;
	std::vector<std::string> parts;

	std::map<std::string, AnnotationRect> firstAnnotation =
			dataset[0].annotationRects;
	std::map<std::string, AnnotationRect>::const_iterator it;

	for (it = firstAnnotation.begin(); it != firstAnnotation.end(); ++it) {
		parts.push_back(it->first);
	}

	std::vector<std::string>::const_iterator partNameIterator;

	for (unsigned i = 0; i < dataset.size(); ++i) {
		for (partNameIterator = parts.begin(); partNameIterator != parts.end();
				++partNameIterator) {
			// Scale the image to the canonical size and store it in the map
			cv::Mat partROI = dataset[i].getPartROI(*partNameIterator);
			cv::Size2f targetSize(dataset[i][*partNameIterator].targetWidth,
					dataset[i][*partNameIterator].targetHeight);
			cv::resize(partROI, partROI, targetSize);

			trainingImages[*partNameIterator].push_back(partROI);

			cv::Mat partROINegative = dataset[i].getNegativeSample(
					*partNameIterator);

			cv::resize(partROINegative, partROINegative, targetSize);
			trainingImages[(*partNameIterator) + "_negative"].push_back(
					partROINegative);

		}
	}

	trainNodes(psTree, trainingImages);

}

void PictorialStructure::train(Dataset dataset) {

// Find out which parts are connected.
	std::map<std::string, std::map<std::string, std::vector<ConnectionData> > > connectionMap =
			buildConnectionMap(dataset);

// Get multivariate normal distribution of each connection
	std::map<std::string, std::map<std::string, ConnectionDataDistribution> > connectionDistribution =
			calcConnDistribution(connectionMap);

// Calculate the energy of each connection as the sum of the -log(quality(edge))
	std::map<std::string, std::map<std::string, double> > connectionEnergy =
			calcConnEnergy(connectionMap, connectionDistribution);

// Use the Kruskal algorithm to find out which parts should be connected
	std::map<std::string, std::map<std::string, ConnectionDataDistribution> > mstConnnDistribution =
			findConnectedParts(connectionEnergy, connectionDistribution);

// Build the Pictorial structure tree
	psTree = buildPSTree(mstConnnDistribution, connectionMap);

// Train the appearance filters
	trainAppearanceFilters(psTree, dataset);

// Save the dataset, it will be useful and be used to know that the structure has been trained
	this->dataset = dataset;

	trained = true;

}

void PictorialStructure::retrieveRect(PSNode &psTree, AnnotationRect &ar) {
	ar.id = psTree.getID();
	std::vector<float (PSNode::*)(float a, float b)> dimensionOp =
			psTree.dimensionOp;
	float x = (psTree.*dimensionOp[3])(psTree.getPartLocation()[3],
			psTree.getDimensionScales()[3]);
	float y = (psTree.*dimensionOp[2])(psTree.getPartLocation()[2],
			psTree.getDimensionScales()[2]);
	float scale = (psTree.*dimensionOp[1])(psTree.getDimensionScales()[1],
			psTree.getPartLocation()[1]);
	float angle = (psTree.*dimensionOp[0])(psTree.getPartLocation()[0],
			psTree.getDimensionScales()[0]);

	float sizeX = dataset[0][ar.id].targetWidth;
	float sizeY = dataset[0][ar.id].targetHeight;

	ar.annotation.center = cv::Point2f(x, y);
	ar.annotation.size = cv::Size2f(sizeX * scale, sizeY * scale);
	ar.annotation.angle = angle;
}

void PictorialStructure::retrieveConfiguration(PSNode &psTree,
		std::vector<AnnotationRect> &annotationRects) {

	AnnotationRect ar;
	retrieveRect(psTree, ar);
	annotationRects.push_back(ar);

	for (unsigned i = 0; i < psTree.children.size(); ++i) {
		retrieveConfiguration(psTree.children[i], annotationRects);
	}

}

std::vector<AnnotationRect> PictorialStructure::detect(cv::Mat inputImage,
		double *score) {

	CV_Assert(trained);

	std::vector<AnnotationRect> annotationRects;

// Solve the tree
	psTree.solve(inputImage);

// Go through the tree nodes retrieving the configuration (location) parameters
	retrieveConfiguration(psTree, annotationRects);

	if (score != 0) {
		*score = psTree.getScore();
	}

	return annotationRects;
}

int PictorialStructure::getFilterType() {
	return filterType;
}

void PictorialStructure::setFilterType(int filterType) {
	if (this->filterType != filterType) {
		trained = false;
		this->filterType = filterType;
	}
}

int PictorialStructure::getNScaleLevels() {
	return psTree.getNScaleLevels();
}

void PictorialStructure::setNScaleLevels(int nScaleLevels) {
	psTree.setNScaleLevels(nScaleLevels);
}

double PictorialStructure::getScaleFactor() {
	return psTree.getScaleFactor();
}

void PictorialStructure::setScaleFactor(double scaleFactor) {
	psTree.setScaleFactor(scaleFactor);
}

int PictorialStructure::getNRotationLevels() {
	return psTree.getNRotationLevels();
}

void PictorialStructure::setNRotationLevels(int nRotationLevels) {
	psTree.setNRotationLevels(nRotationLevels);
}

bool PictorialStructure::read(cv::FileNode& obj) {
	if (!obj.isMap())
		return false;

	cv::FileNode fn = obj["dataset"];
	dataset.read(fn);

	fn = obj["psTree"];
	psTree.read(fn);

	obj["trained"] >> trained;

	return true;
}

void PictorialStructure::write(cv::FileStorage& fs,
		const std::string& objName) const {

	if (!objName.empty())
		fs << objName;

	fs << "{";

	dataset.write(fs, "dataset");

	psTree.write(fs, "psTree");

	fs << "trained" << trained;

	fs << "}";
}

bool PictorialStructure::load(const std::string& filename,
		const std::string& objname) {
	cv::FileStorage fs(filename, cv::FileStorage::READ);
	cv::FileNode obj =
			!objname.empty() ? fs[objname] : fs.getFirstTopLevelNode();
	return read(obj);
}

void PictorialStructure::save(const std::string& filename,
		const std::string& objName) const {
	cv::FileStorage fs(filename, cv::FileStorage::WRITE);
	// This will be the root element; it must have a name.
	write(fs,
			!objName.empty() ?
					objName : cv::FileStorage::getDefaultObjectName(filename));
}

