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

#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/opencv_modules.hpp"

#include "pictorialStructure/PSNode.hpp"
#include "dt/dt.hpp"
#include "util/util.hpp"
#include <cmath>

class MatchCostInvoker: public cv::ParallelLoopBody {
public:
    MatchCostInvoker(PartFilter *filter, const cv::Mat& inputImage,
            const std::vector<double> &scaleLevels,
            const std::vector<double> &rotationLevels, cv::Mat *energyMatrix) {
        this->filter = filter;
        this->inputImage = inputImage;
        this->scaleLevels = scaleLevels;
        this->rotationLevels = rotationLevels;
        this->energyMatrix = energyMatrix;
    }

    /*
     * Rotates image without cropping it
     */
    void rotate(const cv::Mat inputImage, cv::Mat &outputImage,
            double angle) const {

        outputImage = inputImage.clone();

        cv::Point2f center(outputImage.cols / 2., outputImage.rows / 2.);

        cv::RotatedRect rotatedRect(center, outputImage.size(), angle);
        cv::Rect boundingRect = rotatedRect.boundingRect();

        angle *= CV_PI / 180;
        double alpha = cos(angle);
        double beta = sin(angle);

        cv::Mat rotationTransform = cv::Mat(2, 3, CV_64F);
        double *m = (double *) rotationTransform.data;

        m[0] = alpha;
        m[1] = beta;
        m[2] = (1 - alpha) * center.x - beta * center.y
                + (boundingRect.width - outputImage.cols) / 2.;
        m[3] = -beta;
        m[4] = alpha;
        m[5] = beta * center.x + (1 - alpha) * center.y
                + (boundingRect.height - outputImage.rows) / 2.;

        cv::warpAffine(outputImage, outputImage, rotationTransform,
                boundingRect.size(), cv::INTER_CUBIC);

    }

    void unrotate(const cv::Mat inputImage, cv::Mat &outputImage, double angle,
            cv::Size sz) const {

        outputImage = inputImage.clone();

        cv::Mat inverseRotationTransform(2, 3, CV_64F);

        double *m = (double *) inverseRotationTransform.data;

        cv::Point2f center = cv::Point2f(outputImage.cols / 2.,
                outputImage.rows / 2.);
        angle *= CV_PI / 180;
        double alpha = cos(-angle);
        double beta = sin(-angle);
        m[0] = alpha;
        m[1] = beta;
        m[2] = (1 - alpha) * center.x - beta * center.y
                - (outputImage.cols - sz.width) / 2.;
        m[3] = -beta;
        m[4] = alpha;
        m[5] = beta * center.x + (1 - alpha) * center.y
                - (outputImage.rows - sz.height) / 2.;

        cv::warpAffine(outputImage, outputImage, inverseRotationTransform, sz,
                cv::INTER_CUBIC);

    }

    void operator()(const cv::Range& range) const {
        int i, i1 = range.start, i2 = range.end;

        for (i = i1; i < i2; i++) { // Process current range of scales

            double scale = scaleLevels[i];

            cv::Size sz(cvRound(inputImage.cols / scale),
                    cvRound(inputImage.rows / scale));
            cv::Mat scaledImage;
            if (sz == inputImage.size()) {
                scaledImage = cv::Mat(sz, inputImage.type(), inputImage.data,
                        inputImage.step);
            } else {
                resize(inputImage, scaledImage, sz);
            }

            for (unsigned j = 0; j < rotationLevels.size(); j++) { // Process all desired orientations
                double angle = rotationLevels[j];
                cv::Mat scaledRotatedImage;
                rotate(scaledImage, scaledRotatedImage, angle);

                cv::Mat energyImage;
                filter->filter(scaledRotatedImage, energyImage);

                // Undo rotation on the energy image
                unrotate(energyImage, energyImage, angle, sz);

                // Undo scale (the energy image will be the same size as the input image

                cv::resize(energyImage, energyImage, inputImage.size());

//				std::vector<cv::Range> ranges;
//				ranges.push_back(cv::Range(j, j + 1));
//				ranges.push_back(cv::Range(i, i + 1));
//				ranges.push_back(cv::Range(0, energyImage.rows));
//				ranges.push_back(cv::Range(0, energyImage.cols));
//
//				int sizes[] = { 1, 1, inputImage.rows, inputImage.cols };
//				cv::Mat tmp = cv::Mat(4, sizes, CV_8UC1, energyImage.data);
//
//				tmp.copyTo((*energyMatrix)(&ranges[0]));

                /*
                 * Might be dangerous to memcpy instead of
                 * using the above commented code; but its faster.
                 * Using a mutex is be not necessary since it writes on different regions.
                 */
                memcpy(
                        energyMatrix->data + i * energyMatrix->step[1]
                                + j * energyMatrix->step[0], energyImage.data,
                        energyImage.cols * energyImage.rows * sizeof(uchar));

            }

        }
    }

private:
    PartFilter *filter;
    cv::Mat inputImage;
    std::vector<double> scaleLevels;
    std::vector<double> rotationLevels;
    cv::Mat *energyMatrix;
};

PSNode::PSNode() :
        filter(0), id(""), nScaleLevels(4), scaleFactor(1.2), nRotationLevels(
                1), scaleChanged(true) {
    dimensionScales.push_back(360 / nRotationLevels); // Rotation
    dimensionScales.push_back(scaleFactor); // Scale
    dimensionScales.push_back(1); // Y
    dimensionScales.push_back(1); // X
    score = 0;
}

PSNode::~PSNode() {

}

void PSNode::calcMatchCost(const cv::Mat inputImage, cv::Mat &energyMatrix) {
    std::vector<double> scaleLevels;
    std::vector<double> rotationLevels;

    double scale = 1.;
    double rotation = 0.;
    for (int levels = 0; levels < nScaleLevels; levels++) {

        if (cvRound(inputImage.cols / scale) < filter->getWindowSize().width
                || cvRound(inputImage.rows / scale)
                        < filter->getWindowSize().height || scaleFactor <= 1)
            break;
        scaleLevels.push_back(scale);
        scale *= scaleFactor;
    }

    double rotationFactor = 360. / (double) nRotationLevels;
    for (int levels = 0; levels < nRotationLevels; levels++) {
        rotationLevels.push_back(rotation);
        rotation -= rotationFactor;
    }

    int sizes[] = { (int) rotationLevels.size(), (int) scaleLevels.size(),
            inputImage.rows, inputImage.cols };
    energyMatrix = cv::Mat(4, sizes, CV_8UC1);

    cv::Range range(0, (int) scaleLevels.size());

    MatchCostInvoker invoker(filter.get(), inputImage, scaleLevels,
            rotationLevels, &energyMatrix);

    cv::parallel_for_(range, invoker);

}

void PSNode::translateResponse(const cv::Mat energyMatrix,
        cv::Mat &translatedEnergyMatrix) {

    CV_Assert(energyMatrix.dims == (int ) means.size());
    const float inf = 1e15f;

    std::vector<int> paddedSizes;
    std::vector<cv::Range> originalRanges;

    // Orientations are cyclic, so pad that dimension twice
    {
        int padding = 2 * energyMatrix.size[0];
        int paddedSize = energyMatrix.size[0] + padding;

        paddedSizes.push_back(paddedSize);
        originalRanges.push_back(
                cv::Range(padding / 2, energyMatrix.size[0] + padding / 2));
    }
    {
        int paddedSize = energyMatrix.size[1];

        paddedSizes.push_back(paddedSize);
        originalRanges.push_back(cv::Range(0, energyMatrix.size[1]));
    }
    for (int d = 2; d < energyMatrix.dims; ++d) {

        int padding = 2 * std::abs(means[d]);
        int paddedSize = energyMatrix.size[d] + padding;

        paddedSizes.push_back(paddedSize);
        originalRanges.push_back(
                cv::Range(padding / 2, energyMatrix.size[d] + padding / 2));
    }

    cv::Mat paddedEnergyMatrix(energyMatrix.dims, &paddedSizes[0], CV_32FC1,
            inf);

    std::vector<cv::Range> translatedRanges;

    {
        int offset = energyMatrix.size[0] + means[0];
        translatedRanges.push_back(
                cv::Range(offset, energyMatrix.size[0] + offset));
    }
    {
        translatedRanges.push_back(cv::Range(0, energyMatrix.size[1]));
    }
    for (int d = 2; d < energyMatrix.dims; ++d) {
        int offset = std::abs(means[d]) + means[d];
        translatedRanges.push_back(
                cv::Range(offset, energyMatrix.size[d] + offset));
    }

    std::vector<cv::Range> tmpRanges;
    tmpRanges = originalRanges;
    energyMatrix.copyTo(paddedEnergyMatrix(&tmpRanges[0]));

    tmpRanges[0].start -= energyMatrix.size[0];
    tmpRanges[0].end -= energyMatrix.size[0];
    energyMatrix.copyTo(paddedEnergyMatrix(&tmpRanges[0]));

    tmpRanges = originalRanges;
    tmpRanges[0].start += energyMatrix.size[0];
    tmpRanges[0].end += energyMatrix.size[0];
    energyMatrix.copyTo(paddedEnergyMatrix(&tmpRanges[0]));

    translatedEnergyMatrix = paddedEnergyMatrix(&translatedRanges[0]).clone();

}

void PSNode::mahalanobisDistanceTransform(const cv::Mat translatedEnergyMatrix,
        cv::Mat &outputEnergyMatrix, cv::Mat &locations) {
    outputEnergyMatrix = translatedEnergyMatrix.clone();
    distanceTransform(outputEnergyMatrix, outputEnergyMatrix, locations, sDevs);
}

void PSNode::solve(const cv::Mat &inputImage, cv::Mat &partEnergyMatrix,
        bool root) {

    childrenTranslatedEnergyMatrices.clear();
    childrenTranslatedEnergyLocations.clear();

    if (scaleChanged) {
        scaleChanged = false;
        // Update means and sDevs to work with current scales
        updateMoments();
    }

    // Get part response
    calcMatchCost(inputImage, partEnergyMatrix);

    cv::Mat partEnergyMatrixF;

    partEnergyMatrix.convertTo(partEnergyMatrixF, CV_32FC1);

    /*
     * Get children response, for each children:
     * Translate;
     * Perform a distance transform;
     * Add the result to the current part matrix.
     * If the parts are visible, low energy locations should match.
     */
    for (std::vector<PSNode>::iterator it = children.begin();
            it != children.end(); ++it) {
        cv::Mat childEnergyMatrix;
        cv::Mat childEnergyLocations;
        cv::Mat childEnergyMatrixF;
        cv::Mat childEnergyMatrixFT;
        cv::Mat childEnergyMatrixFTDT;
        it->solve(inputImage, childEnergyMatrix, false);
        childEnergyMatrix.convertTo(childEnergyMatrixF, CV_32FC1);

        it->translateResponse(childEnergyMatrixF, childEnergyMatrixFT);

        it->mahalanobisDistanceTransform(childEnergyMatrixFT,
                childEnergyMatrixFTDT, childEnergyLocations);

        childrenTranslatedEnergyMatrices.push_back(childEnergyMatrixFTDT);
        childrenTranslatedEnergyLocations.push_back(childEnergyLocations);

        /*
         * Check if both have the same size on the scales dimension.
         * On a tree which nodes have filters with different sizes this might happen.
         * Pad the smaller one with inf.
         */
        if (partEnergyMatrixF.size[1] != childEnergyMatrixFTDT.size[1]) {
            const float inf = 1e15f;
            std::vector<int> paddedSizes;
            std::vector<cv::Range> ranges;
            for (int i = 0; i < partEnergyMatrixF.dims; ++i) {
                paddedSizes.push_back(partEnergyMatrixF.size[i]);
                ranges.push_back(cv::Range(0, partEnergyMatrixF.size[i]));
            }
            if (partEnergyMatrixF.size[1] > childEnergyMatrixFTDT.size[1]) {
                paddedSizes[1] = partEnergyMatrixF.size[1];
                ranges[1] = cv::Range(0, childEnergyMatrixFTDT.size[1]);
                cv::Mat paddedEnergyMatrix(partEnergyMatrixF.dims, &paddedSizes[0],
                        CV_32FC1, inf);
                childEnergyMatrixFTDT.copyTo(paddedEnergyMatrix(&ranges[0]));
                childEnergyMatrixFTDT = paddedEnergyMatrix;

            } else {
                paddedSizes[1] = childEnergyMatrixFTDT.size[1];
                ranges[1] = cv::Range(0, partEnergyMatrixF.size[1]);
                cv::Mat paddedEnergyMatrix(partEnergyMatrixF.dims, &paddedSizes[0],
                        CV_32FC1, inf);
                partEnergyMatrixF.copyTo(paddedEnergyMatrix(&ranges[0]));
                partEnergyMatrixF = paddedEnergyMatrix;
            }
        }

        partEnergyMatrixF += childEnergyMatrixFTDT;

    }
    this->partEnergyMatrix = partEnergyMatrixF.clone();
    /*
     * If this is the root part (maybe not the whole model root, but just the part the user wants to solve)
     * find the minimum value (location of the current part), and iterate through the children,
     * calculating the children locations using the current one as reference.
     */
    if (root) {

        // For each children get the location of the parabola and translate it back to its original position
        int minIdx[this->partEnergyMatrix.dims];
        minMaxIdx(this->partEnergyMatrix, &score, 0, minIdx, 0);
        score = -score;
        this->partLocation = std::vector<int>(minIdx,
                minIdx + sizeof(minIdx) / sizeof(minIdx[0]));

        for (int childIdx = 0; childIdx < (int) children.size(); ++childIdx) {
            cv::Mat childTranslatedLocationsMatrix =
                    childrenTranslatedEnergyLocations[childIdx];
            int index = 0;
            for (int d = 0; d < this->partEnergyMatrix.dims; ++d) {
                index += minIdx[d] * childTranslatedLocationsMatrix.step[d + 1]; // The childLocationsMatrix first dim stores the minimum location
            }

            std::vector<int> location;
            for (int d = 0; d < this->partEnergyMatrix.dims; ++d) {
                int l = childTranslatedLocationsMatrix.data[index]; // childTranslatedLocationsMatrix stores translated responses

                //l = this->partLocation[d];
                if (d != 1) {
                    l += children[childIdx].means[d]; // translate the location to its original position
                }
                if (l < 0) {
                    l = 0;
                }
                if (l > this->partEnergyMatrix.size[d] - 1) {
                    l = this->partEnergyMatrix.size[d] - 1;
                }

                location.push_back(l);
                index += childTranslatedLocationsMatrix.step[0]; // Go to the next coordinate index
            }

            // Now we have children location, propagate the solution
            children[childIdx].propagateSolution(location);

        }
    }
}

void PSNode::solve(const cv::Mat &inputImage) {
    cv::Mat emptyMatrix;
    solve(inputImage, emptyMatrix, true);
}

void PSNode::updateMoments() {
    std::vector<int> dX;
    std::vector<int> dY;
    std::vector<float> dScale;
    std::vector<float> dAngle;
    std::vector<ConnectionData>::const_iterator it;
    for (it = connectionData.begin(); it != connectionData.end(); ++it) {
        dAngle.push_back(
                (this->*dimensionIOp[0])(it->dAngle, dimensionScales[0]));
        dScale.push_back(
                (this->*dimensionIOp[1])(it->dScale, dimensionScales[1]));
        dY.push_back((this->*dimensionIOp[2])(it->dY, dimensionScales[2]));
        dX.push_back((this->*dimensionIOp[3])(it->dX, dimensionScales[3]));
    }

    means.clear();
    sDevs.clear();

    cv::Scalar mAngle;
    cv::Scalar sdAngle;
    cv::meanStdDev(dAngle, mAngle, sdAngle);
    means.push_back(mAngle[0]);
    sDevs.push_back(sdAngle[0]);

    cv::Scalar mScale;
    cv::Scalar sdScale;
    cv::meanStdDev(dScale, mScale, sdScale);
    means.push_back(mScale[0]);
    sDevs.push_back(sdScale[0]);

    cv::Scalar mY;
    cv::Scalar sdY;
    cv::meanStdDev(dY, mY, sdY);
    means.push_back(mY[0]);
    sDevs.push_back(sdY[0]);

    cv::Scalar mX;
    cv::Scalar sdX;
    cv::meanStdDev(dX, mX, sdX);
    means.push_back(mX[0]);
    sDevs.push_back(sdX[0]);

}

void PSNode::propagateSolution(std::vector<int> partLocation) {

    this->partLocation = partLocation;
    for (int childIdx = 0; childIdx < (int) children.size(); ++childIdx) {
        cv::Mat childTranslatedLocationsMatrix =
                childrenTranslatedEnergyLocations[childIdx];
        int index = 0;
        for (int d = 0; d < this->partEnergyMatrix.dims; ++d) {
            index += partLocation[d]
                    * childTranslatedLocationsMatrix.step[d + 1]; // The childLocationsMatrix first dim stores the minimum location
        }

        std::vector<int> location;
        for (int d = 0; d < this->partEnergyMatrix.dims; ++d) {
            int l = childTranslatedLocationsMatrix.data[index]; // childTranslatedLocationsMatrix stores translated responses

            //l = this->partLocation[d];
            if (d != 1) {
                l += children[childIdx].means[d]; // translate the location to its original position
            }

            if (l < 0) {
                l = 0;
            }
            if (l > this->partEnergyMatrix.size[d] - 1) {
                l = this->partEnergyMatrix.size[d] - 1;
            }

            location.push_back(l);
            index += childTranslatedLocationsMatrix.step[0]; // Go to the next coordinate index
        }

        // Now we have children location, propagate the solution
        children[childIdx].propagateSolution(location);

    }
}

int PSNode::getNScaleLevels() {
    return nScaleLevels;
}
void PSNode::setNScaleLevels(int nScaleLevels) {
    if (this->nScaleLevels != nScaleLevels) {
        scaleChanged = true;
        this->nScaleLevels = nScaleLevels;
        for (std::vector<PSNode>::iterator it = children.begin();
                it != children.end(); ++it) {
            it->setNScaleLevels(nScaleLevels);
        }
    }
}

double PSNode::getScaleFactor() {
    return scaleFactor;
}

void PSNode::setScaleFactor(double scaleFactor) {
    if (this->scaleFactor != scaleFactor) {
        scaleChanged = true;
        this->scaleFactor = scaleFactor;
        for (std::vector<PSNode>::iterator it = children.begin();
                it != children.end(); ++it) {
            it->setScaleFactor(scaleFactor);
        }
    }
}

int PSNode::getNRotationLevels() {
    return nRotationLevels;
}

void PSNode::setNRotationLevels(int nRotationLevels) {
    if (this->nRotationLevels != nRotationLevels) {
        scaleChanged = true;
        this->nRotationLevels = nRotationLevels;
        for (std::vector<PSNode>::iterator it = children.begin();
                it != children.end(); ++it) {
            it->setNRotationLevels(nRotationLevels);
        }
    }
}

double PSNode::getScore() {
    return score;
}

bool PSNode::read(cv::FileNode& obj) {
    if (!obj.isMap())
        return false;

    cv::FileNode fn = obj["filter"];
    int filterType;
    obj["filterType"] >> filterType;

    filter = std::shared_ptr<PartFilter>(instantiateFilter(filterType));
    filter->read(fn);

    cv::FileNodeIterator it = obj["children"].begin(), it_end =
            obj["children"].end();
    for (; it != it_end; ++it) {
        cv::FileNode fn = *it;
        PSNode newChild;
        newChild.read(fn);
        children.push_back(newChild);
    }

    obj["nScaleLevels"] >> nScaleLevels;
    obj["nRotationLevels"] >> nRotationLevels;
    obj["scaleFactor"] >> scaleFactor;

    it = obj["connectionData"].begin(), it_end = obj["connectionData"].end();
    for (; it != it_end; ++it) {
        cv::FileNode fn = *it;
        ConnectionData cd;
        fn["dX"] >> cd.dX;
        fn["dY"] >> cd.dY;
        fn["dScale"] >> cd.dScale;
        fn["dAngle"] >> cd.dAngle;
        connectionData.push_back(cd);
    }

    obj["id"] >> id;

    dimensionIOp.clear();
    dimensionIOp.push_back(&PSNode::divide);
    dimensionIOp.push_back(&PSNode::logarithm);
    dimensionIOp.push_back(&PSNode::divide);
    dimensionIOp.push_back(&PSNode::divide);

    dimensionOp.clear();
    dimensionOp.push_back(&PSNode::multiply);
    dimensionOp.push_back(&PSNode::raise);
    dimensionOp.push_back(&PSNode::multiply);
    dimensionOp.push_back(&PSNode::multiply);

    return true;
}

void PSNode::write(cv::FileStorage& fs, const std::string& objName) const {

    if (!objName.empty())
        fs << objName;

    fs << "{";

    fs << "filterType" << filter->filterType;

    filter->write(fs, "filter");

    fs << "children" << "[";

    std::vector<PSNode>::const_iterator childrenIterator;
    for (childrenIterator = children.begin();
            childrenIterator != children.end(); ++childrenIterator) {
        childrenIterator->write(fs);
    }

    fs << "]";

    fs << "nScaleLevels" << nScaleLevels;
    fs << "nRotationLevels" << nRotationLevels;
    fs << "scaleFactor" << scaleFactor;

    fs << "connectionData" << "[";

    std::vector<ConnectionData>::const_iterator connectionDataterator;
    for (connectionDataterator = connectionData.begin();
            connectionDataterator != connectionData.end();
            ++connectionDataterator) {
        fs << "{";
        fs << "dX" << connectionDataterator->dX;
        fs << "dY" << connectionDataterator->dY;
        fs << "dScale" << connectionDataterator->dScale;
        fs << "dAngle" << connectionDataterator->dAngle;
        fs << "}";
    }

    fs << "]";

    fs << "id" << id;

    fs << "}";
}

bool PSNode::load(const std::string& filename, const std::string& objname) {
    cv::FileStorage fs(filename, cv::FileStorage::READ);
    cv::FileNode obj =
            !objname.empty() ? fs[objname] : fs.getFirstTopLevelNode();
    return read(obj);
}

void PSNode::save(const std::string& filename,
        const std::string& objName) const {
    cv::FileStorage fs(filename, cv::FileStorage::WRITE);
    // This will be the root element; it must have a name.
    write(fs,
            !objName.empty() ?
                    objName : cv::FileStorage::getDefaultObjectName(filename));
}

