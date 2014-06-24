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

#include <sstream>
#include <iomanip>
#include "include/ui/filterconfig.h"
#include "ui_filterconfig.h"


FilterConfig::FilterConfig(QWidget *parent, int filterType, int scaleCount, float scaleFactor, int orientationCount) :
    QDialog(parent),
    ui(new Ui::FilterConfig)
{
    ui->setupUi(this);
    ui->filterTypeComboBox->setCurrentIndex(filterType);
    ui->scaleSlider->setValue(scaleCount);
    updateScaleLabel(scaleCount);
    ui->scaleFactorSlider->setValue(scaleFactor * 100);
    updateScaleFactorLabel(scaleFactor * 100);
    ui->orientationSlider->setValue(orientationCount);
    updateOrientationLabel(orientationCount);
}

FilterConfig::~FilterConfig()
{
    delete ui;
}

void FilterConfig::updateScaleLabel(int value){
    ui->scaleLabel->setText(QString::number(value));
}

void FilterConfig::updateScaleFactorLabel(int value){
    std::stringstream ss;
    ss << std::fixed << std::setprecision(2) << std::showpoint << (float)value / 100.0;
    ui->scaleFactorLabel->setText(QString(ss.str().data()));
}

void FilterConfig::updateOrientationLabel(int value){
    ui->orientationsLabel->setText(QString::number(value));
}

int FilterConfig::getFilterType(){
    return ui->filterTypeComboBox->currentIndex();
}

int FilterConfig::getScaleCount(){
    return ui->scaleSlider->value();
}

float FilterConfig::getScaleFactor(){
    return ((float) ui->scaleFactorSlider->value() / 100.0);
}

int FilterConfig::getOrientationCount(){
    return ui->orientationSlider->value();
}

