#-------------------------------------------------
#
# Project created by QtCreator 2014-06-10T18:46:28
#
#-------------------------------------------------

QT       += core gui

CONFIG += c++11

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = PSDemo
TEMPLATE = app

INCLUDEPATH += include/

SOURCES += main.cpp\
        src/ui/mainwindow.cpp \
    src/ui/filterconfig.cpp \
    src/ui/resultdialog.cpp \
    src/filter/adaboostHogFilter.cpp \
    src/util/util.cpp \
    src/pictorialStructure/PSNode.cpp \
    src/pictorialStructure/pictorialStructure.cpp \
    src/filter/nCFilter.cpp \
    src/kruskal/kruskal.cpp \
    src/dt/dt.cpp \
    src/dataset/dataset.cpp \
    src/dataset/annotationRect.cpp \
    src/dataset/annotation.cpp

HEADERS  += include/ui/mainwindow.h \
    include/ui/filterconfig.h \
    include/ui/resultdialog.h \
    include/filter/adaboostHogFilter.hpp \
    include/util/util.hpp \
    include/pictorialStructure/PSNode.hpp \
    include/pictorialStructure/pictorialStructure.hpp \
    include/filter/partFilter.hpp \
    include/filter/nCFilter.hpp \
    include/kruskal/kruskal.hpp \
    include/dt/dt.hpp \
    include/dataset/dataset.hpp \
    include/pictorialStructure/connection.hpp \
    include/dataset/annotationRect.hpp \
    include/dataset/annotation.hpp

FORMS    += include/ui/mainwindow.ui \
    include/ui/filterconfig.ui \
    include/ui/resultdialog.ui

LIBS += -lopencv_core \
    -lopencv_imgproc \
    -lopencv_highgui \
    -lopencv_ml \
    -lopencv_video \
    -lopencv_features2d \
    -lopencv_calib3d \
    -lopencv_objdetect \
    -lopencv_contrib \
    -lopencv_legacy \
    -lopencv_flann

OTHER_FILES += \
    PSDemo.pro.user
