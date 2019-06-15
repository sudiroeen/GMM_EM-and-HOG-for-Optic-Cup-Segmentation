/*
 Author:

 	(1) Sudiro 
 		   [at] SudiroEEN@gmail.com

 	(2) M Izzarrasyadi Wachid
 		[at] izzarrasyadi@gmail.com

*/


#include "../LIB_SOURCE/GMM_EM.hpp"
#include "../LIB_SOURCE/HOG_feature.hpp"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <cstdlib>

using namespace std;
using namespace cv;

vector<Mat> masukanMatrix(Mat gambar, Rect kotak);
static void onMouse(int event, int x, int y, int, void*);

Mat gambar;
Mat gambarClone;
Rect rectROI;
Point titikStart;
bool afterDownBeforeUp = false;

// #define DATASET_CROP
// #define DATASET_FRAME
#define DATASET_LOAD

int main(){	
	vector<Mat> _datasetWarna;

	int state = GMM::STATE_COLLECT;

#ifdef DATASET_CROP
		if(state == STATE_COLLECT){
			if((rectROI.width != 0) || (rectROI.height != 0) ){
                vector<Mat> dataTemp;

                if((char)inkey == 's'){
                	dataTemp = masukanMatrix(gambar, rectROI);
                	cout << "data saved !!!" << endl;
                }

	            if(dataTemp.size()){
	            	if(_datasetWarna.size() > 1000){
						cout << "Dataset udah banyak brooo" << endl;

						state = GMM::STATE_TRAIN;

						char simpan;
						cin >> simpan;
						if(simpan == 'b'){
							FileStorage __fs("../../posNeg/", FileStorage::WRITE);
							__fs << "_dimensi" << _datasetWarna[0].rows;
							__fs << "size" << (int)_datasetWarna.size();
							for(int zz=0; zz<_datasetWarna.size(); zz++){
								stringstream ssm;
								ssm << zz;
								
								__fs << "data_"+ssm.str() << _datasetWarna[zz];

								cout << "saved-" << zz << endl;
							}
							__fs.release();
						}
					 }else{
					 	int tambah = dataTemp.size() < 200 ? dataTemp.size()-1 : 200;
	            		_datasetWarna.insert(_datasetWarna.end(), dataTemp.begin(), dataTemp.begin() + tambah);
	            	 }
	            	cout << _datasetWarna.size() << endl;
	            }
	        }
	    }
#elif defined(DATASET_FRAME)
		if(state == GMM::STATE_COLLECT){
            if((char) inkey == 'y'){
            	_frameAsDataSet = frame;
            	state = GMM::STATE_TRAIN;
            	destroyAllWindows();
            }
        }
#elif defined(DATASET_LOAD)
        if(state == GMM::STATE_COLLECT)
        	state = GMM::STATE_TRAIN;
#endif


#ifdef DATASET_CROP
		if(state == GMM::STATE_TRAIN){
			if(! _datasetWarna.size()){
				cout << "Harus ada dataset BROO" << endl;
			}else if(_datasetWarna.size() > 1000){
				int banyakKluster = 3;
				vector<Mat> awal;
				awal.push_back(Mat::ones(3780,1,CV_64FC1));
				awal.push_back(Mat::ones(3780,1,CV_64FC1));

				GMM _gausCROP(banyakKluster, _datasetWarna, awal);
				_gausCROP.train(2000, "../configF/tr_fHOG.yaml");
				state = GMM::STATE_PREDICT;
			}
		}
#elif defined(DATASET_LOAD)
		if(state == GMM::STATE_TRAIN){
				int banyakKluster = 3;		        

		        vector<Mat> awal;
				awal.push_back(Mat::ones(3780,1,CV_64FC1));
				awal.push_back(Mat::ones(3780,1,CV_64FC1));

				GMM _gausLOAD(banyakKluster, "../datasetGMM/labels/", GMM::STORAGE_FOLDER , awal);
				_gausLOAD.train(2000, "../configF/tr_fHOG.yaml");
				state = GMM::STATE_PREDICT;
				exit(0);
			}
#elif defined(DATASET_FRAME)
		if(state == GMM::STATE_TRAIN){
				int banyakKluster = 3;
				vector<Mat> awal;
				awal.push_back(Mat::ones(3780,1,CV_64FC1));
				awal.push_back(Mat::ones(3780,1,CV_64FC1));

				GMM _gausFRAME(banyakKluster, _frameAsDataSet, awal);
				_gausFRAME.train(2000, "../configF/tr_fHOG.yaml");
				state = GMM::STATE_PREDICT;
			}
#endif

		if(state == GMM::STATE_PREDICT){

		}
#if 0
		if(state == GMM::STATE_PREDICT){
			Mat blank = Mat::zeros(frame.size(), CV_8UC3);

			GMM _gaus_predict;
			_gaus_predict.loadConfig("../configF/tr_fHOG.yaml");
			for(int r=0;r<frame.rows; r++){
				Vec3b* ptr_ = frame.ptr<Vec3b>(r);
				Vec3b* _ptr_blank = blank.ptr<Vec3b>(r);

				for(int c=0; c<frame.cols; c++){					
					double B = (double)ptr_[c][2];
					double G = (double)ptr_[c][1];
					double R = (double)ptr_[c][0];

					Mat _pixel = (Mat_<double>(3,1) << B, G, R);

					int predicted = _gaus_predict.predict(_pixel);

					switch(predicted){
						case 0:
							_ptr_blank[c] = Vec3b(0, 0, 255);
							break;
						case 1: 
							_ptr_blank[c] = Vec3b(0, 255, 0);
							break;
						case 2: 
							_ptr_blank[c] = Vec3b(255, 0, 0);
							break;
					}
				}
			}
			namedWindow("labeled", CV_WINDOW_NORMAL);
			imshow("labeled", blank);
		}
#endif
}



static void onMouse(int event, int x, int y, int, void*){
    int xrs, yrs, lx, ly;

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

        afterDownBeforeUp = false;
    }
}

vector<Mat> masukanMatrix(Mat gambar, Rect kotak){
    int xrs, yrs, xrf, yrf;
    xrs = kotak.x;
    yrs = kotak.y;
    xrf = xrs + kotak.width;
    yrf = yrs + kotak.height;

    vector<Mat> RGB;
    for(int xx=xrs+1; xx<xrf; xx++){
        for(int yy=yrs+1; yy<yrf; yy++){
            Vec3b pixel = gambar.at<Vec3b>(yy,xx);

            double R = (double)pixel[2];
            double G = (double)pixel[1];
            double B = (double)pixel[0];

            RGB.push_back((Mat_<double>(3,1) << B, G, R));
        }
    }

    return RGB;
}




