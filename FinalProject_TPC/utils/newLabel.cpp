/*
 Author:

  (1) Sudiro 
       [at] SudiroEEN@gmail.com

*/

#include <opencv2/opencv.hpp>
#include <iostream>
#include <dirent.h>
#include <cstring> 

#include "../LIB_SOURCE/HOG_feature.hpp"


using namespace std;
using namespace cv;

Mat gambar;
Mat gambarClone;
Point titikStart;
bool afterDownBeforeUp = false;
Rect rectROI;
bool _EV;

static void readMouse(int event, int x, int y, int, void*){
    int xrs, yrs, lx, ly;

    _EV = false;

    if(afterDownBeforeUp){
        gambar = gambarClone.clone();
        xrs = min(titikStart.x, x);
        yrs = min(titikStart.y, y);
        lx = max(titikStart.x, x) - min(titikStart.x, x);
        ly = max(titikStart.y, y) - min(titikStart.y, y);
        rectROI = Rect(xrs, yrs, lx+1, ly+1);

        rectangle(gambar, rectROI,Scalar(255, 0, 0), 1);
    }
    if(event == EVENT_LBUTTONDOWN){
        titikStart = Point(x,y);
        rectROI = Rect(x,y,0,0);
        afterDownBeforeUp = true;

    }else if(event == EVENT_LBUTTONUP){
        Mat roi(gambarClone.clone(), rectROI);
        imshow("roi", roi);

        _EV = true;

        afterDownBeforeUp = false;
    }

}


string SourceMataUtuh = "../SourceMataUtuh/";
string DirOfCropped = "../datasetGMM/";

int main(){
  int count  = 0;

	for(int i=0; i<190; i++){
		stringstream ss;
		ss << "../SourceMataUtuh/mata_" << i << ".jpg";

   		namedWindow(ss.str().c_str(), CV_WINDOW_NORMAL);
	   	setMouseCallback(ss.str().c_str(), readMouse);

   		Mat img = cv::imread(ss.str().c_str());
      resize(img, img, Size(), 640.0/img.cols, 480.0/img.rows);

   		gambar = img;
      gambarClone = gambar.clone();

   		while(true){
        cv::imshow(ss.str().c_str(), gambar);
        int inkey =  waitKey(10);
   			if((rectROI.width != 0) || (rectROI.height != 0) ){

            Mat _savekah(img, rectROI);
            imshow("_savekah", _savekah);

            if((char)inkey == 's'){
              stringstream _namess;
              _namess << DirOfCropped << "images/" << "dataset_" << count << ".jpg";

              imwrite(_namess.str().c_str(), _savekah);

              stringstream _nameYAMl;
              _nameYAMl << DirOfCropped  << "labels/" << "dataset_" << count << ".yaml";

              HOG_feature deklarHOG(_savekah, _nameYAMl.str().c_str());
              deklarHOG.getHOG_feature();
              deklarHOG.saveFeature();
              rectROI = Rect();

              count ++;
            }else if( (char)inkey == 'd'){
              stringstream _namess;
              _namess << DirOfCropped << "images/" << "dataset_" << count << ".jpg";

              imwrite(_namess.str().c_str(), _savekah);

              stringstream _nameYAMl;
              _nameYAMl << DirOfCropped << "labels/" << "dataset_" << count << ".yaml";

              HOG_feature deklarHOG(_savekah, _nameYAMl.str().c_str());
              deklarHOG.getHOG_feature();
              deklarHOG.saveFeature();

              rectROI = Rect();

              count ++;
              destroyAllWindows();
              break;
            }
        }
		}
	}
}


