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

#include <algorithm>
#include <iostream>
#include "kruskal/kruskal.hpp"

UnionFind::UnionFind(std::vector<Node *> &nodes, std::vector<Edge *> &edges) {
	this->nodes = nodes;
	this->edges = edges;
	for (std::vector<Node *>::iterator it = nodes.begin(); it != nodes.end();
			it++) {
		ufTable[(*it)] = *it;
	}
}

UnionFind::~UnionFind() {

}

bool UnionFind::join(Node *nodeA, Node *nodeB) {
	Node *a = find(nodeA);
	Node *b = find(nodeB);

	// Reference comparison, we want to know if they are the same object
	if (a == b)
		return false;

	if (a->rank > b->rank) {
		std::swap(a, b);
	}

	// if both are equal, the combined tree becomes 1 deeper
	if (a->rank == b->rank)
		b->rank++;

	ufTable[a] = b;
	return true;
}

Node *UnionFind::find(Node *node) {
	if (ufTable[node] != ufTable[ufTable[node]]) {
		ufTable[node] = find(ufTable[node]);
	}
	return ufTable[node];
}

Kruskal::Kruskal(std::vector<Node *> &nodes, std::vector<Edge *> &edges) :
		uf(nodes, edges), nodes(nodes), edges(edges) {
	solved = false;
}

bool Kruskal::sortFunction(Edge* first, Edge* second) {
	return first->getWeight() < second->getWeight();
}

std::vector<Edge *> Kruskal::getEdges() {
	if (!solved) {
		solve();
	}
	return mst;
}

float Kruskal::solve() {
	solved = true;
	std::sort(edges.begin(), edges.end(), sortFunction);

	int treeCount = nodes.size();
	float weight = 0;

	for (unsigned int i = 0; i < edges.size() && treeCount > 1; i++) {
		if (uf.join(edges[i]->a, edges[i]->b)) {
			treeCount--;
			weight += edges[i]->getWeight();

			// Save the edge, it may be useful for the user
			Edge *e = edges[i];
			mst.push_back(e);
		}
	}

	return weight;
}
