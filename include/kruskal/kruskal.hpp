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

#ifndef KRUSKAL_H
#define KRUSKAL_H

#include <vector>
#include <map>

//Kruskal's algorithm implementation using an union-find data structure

// Graph node interface, to be extended for use
class Node {

public:
	Node() :
			rank(1) {
	}
	virtual ~Node() {
	}
	int rank;

};

// Graph edge interface, to be extended for use
class Edge {

public:

	virtual ~Edge() {
	}

	/*
	 * An edge should, at last, have 2 end points and a weight
	 */
	Node *a;
	Node *b;
	virtual float getWeight() = 0;

};

// Union-find data structure
class UnionFind {

private:
	std::vector<Node *> nodes;
	std::vector<Edge *> edges;
	/*
	 * Union-Find table; for each node, store its parent
	 * It's not as efficient as an array, but flexibility comes with a price
	 */
	std::map<Node *, Node *> ufTable;

public:
	UnionFind() {
	}
	UnionFind(std::vector<Node *> &nodes, std::vector<Edge *> &edges);
	virtual ~UnionFind();
	bool join(Node *nodeA, Node *nodeB);
	Node *find(Node *node);

};

// Implementation of the Kruskal's algorithm
class Kruskal {

private:
	UnionFind uf;
	std::vector<Node *> nodes;
	std::vector<Edge *> edges;
	std::vector<Edge *> mst;
	bool solved;

protected:
	/*
	 * Edges will be sorted this way
	 */
	static bool sortFunction(Edge* first, Edge* second);

public:
	Kruskal(std::vector<Node *> &nodes, std::vector<Edge *> &edges);
	// Get the edges that belong to the minimum spanning tree
	std::vector<Edge *> getEdges();
	// Performs the Kruksal's algorithm and returns the weight of the minimum spanning tree
	float solve();

};

#endif
