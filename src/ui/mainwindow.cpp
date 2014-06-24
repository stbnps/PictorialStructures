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

#include "include/ui/mainwindow.h"
#include "ui_mainwindow.h"
#include "include/ui/filterconfig.h"
#include "include/ui/resultdialog.h"

#include <iostream>
#include <QFileDialog>
#include <QRunnable>
#include <QThreadPool>
#include <QObject>
#include <QMessageBox>

#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/opencv_modules.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include "include/util/util.hpp"

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{    
    ui->setupUi(this);
    installEventFilter(this);
    QObject::connect(&work, &TrainerThread::finished,
                         this, &MainWindow::trainingFinished);
    state = AppState::Ready;
}

MainWindow::~MainWindow()
{
    delete ui;
}

bool MainWindow::eventFilter(QObject* watched, QEvent* event) {
    return event->type() == QEvent::StatusTip;
}

void MainWindow::trainingFinished(){
    ui->statusBar->showMessage("Finished.");
    state = AppState::Ready;
}

void MainWindow::trainPictorialStructure(){

    if (state == AppState::Ready){

        QString fileName = QFileDialog::getOpenFileName(this, tr("Select dataset"),
                                                         "",
                                                         tr("Datasets (*.ds);;All files (*.*)"));

        if (fileName.isEmpty()){
            return;
        }

        // Create new objects, just in case
        dataset = Dataset();
        int filterType = ps.getFilterType();
        ps = PictorialStructure();
        ps.setFilterType(filterType);

        dataset.load(fileName.toStdString());

        work.ds = &dataset;
        work.ps = &ps;
        work.setAutoDelete(false);
        QThreadPool *threadPool = QThreadPool::globalInstance();
        threadPool->start(&work);
        ui->statusBar->showMessage("Training, please wait.");
        state = AppState::Training;

    } else {
        QMessageBox messageBox;
        messageBox.critical(0, "Error", "The training session has not finished.");
    }

}

void MainWindow::loadPictorialStructure(){

    if (state == AppState::Ready){

        QString fileName = QFileDialog::getOpenFileName(this, tr("Select Pictorial Structure"),
                                                         "",
                                                         tr("Structures (*.ps);;All files (*.*)"));
        if (fileName.isEmpty()){
            return;
        }

        ps = PictorialStructure();
        ps.load(fileName.toStdString());
    } else {
        QMessageBox messageBox;
        messageBox.critical(0, "Error", "The training session has not finished.");
    }
}

void MainWindow::savePictorialStructure(){    

    if (state == AppState::Ready){
        QString defaultExtension = "Structures (*.ps)";
        QString fileName = QFileDialog::getSaveFileName(this, tr("Select file to save"), "", tr("Structures (*.ps);;All files (*.*)"), &defaultExtension);
        if (fileName.isEmpty()){
            return;
        }

        QFile file(fileName);
        QFileInfo fi(file);
        if(fi.suffix().isEmpty())
        {
            fileName.append(".ps");
        }

        ps.save(fileName.toStdString());
    } else {
        QMessageBox messageBox;
        messageBox.critical(0, "Error", "The training session has not finished.");
    }
}


void MainWindow::detect(){

    if (state == AppState::Ready){

        if (ps.isTrained()){
            QString fileName = QFileDialog::getOpenFileName(this, tr("Open File"),
                                                             "",
                                                             tr("Images (*.jpg *.jpg *.gif *.jpg *.pgm);;All files (*.*)"));
            if (fileName.isEmpty()){
                return;
            }


            cv::Mat test = cv::imread(fileName.toStdString());

            cv::Mat testGray;
            cv::cvtColor(test, testGray, CV_BGR2GRAY);

            std::vector<AnnotationRect> detection = ps.detect(testGray);

            overlayAnnotations(test, detection);

            cv::Mat testRGB;
            cv::cvtColor(test, testRGB, CV_BGR2RGB);

            QImage displayedResult = QImage((uchar*)testRGB.data, testRGB.cols, testRGB.rows, testRGB.step, QImage::Format_RGB888);

            ResultDialog *rd = new ResultDialog(this, displayedResult);

            rd->show();
        } else {
            QMessageBox messageBox;
            messageBox.critical(0, "Error", "The pictorial structure has not been trained.");
        }

    } else {
        QMessageBox messageBox;
        messageBox.critical(0, "Error", "The training session has not finished.");
    }

}


void MainWindow::configureFiltering(){

    if (state == AppState::Ready){
        FilterConfig *fc = new FilterConfig(this, ps.getFilterType(), ps.getNScaleLevels(), ps.getScaleFactor(), ps.getNRotationLevels());
        int returnCode = fc->exec();
        if (returnCode) {
            int filterType = fc->getFilterType();
            int scaleCount = fc->getScaleCount();
            float scaleFactor = fc->getScaleFactor();
            int orientationCount = fc->getOrientationCount();

            ps.setFilterType(filterType);
            ps.setNScaleLevels(scaleCount);
            ps.setScaleFactor(scaleFactor);
            ps.setNRotationLevels(orientationCount);
        }
    } else {
        QMessageBox messageBox;
        messageBox.critical(0, "Error", "The training session has not finished.");
    }
}





