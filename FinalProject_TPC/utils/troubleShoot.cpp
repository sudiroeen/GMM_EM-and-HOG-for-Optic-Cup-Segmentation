/*
 Author:

 	(1) Sudiro 
 	   [at] SudiroEEN@gmail.com
*/


#include<opencv2/opencv.hpp>
#include<iostream>
#include<dirent.h>

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


int main(){
	for(int i=0; i<222; i++){
		stringstream ss;
		ss <<  "../datasetGMM/labels/" << "dataset_" << i << ".yaml";

		FileStorage mfs(ss.str().c_str(), FileStorage::READ);
		Mat _temp;

		mfs["feature__oc"] >> _temp;
		mfs.release();

		FileStorage mmf(ss.str().c_str(), FileStorage::WRITE);

		mmf << "feature" << _temp;
		mmf.release();
	}
	
}
	// Mat _pos;
	// 			FileStorage _fpos("/home/udiro/Videos/Images_optic_cup/8 juni 2019/posNeg/pos_mata_14.yaml",
	// 							FileStorage::READ);
	// _fpos["feature"] >> _pos;
	// _pos.convertTo(_pos, CV_64F);
	// _fpos.release();

	// resize(_pos, _pos, Size(), (double)_pos.size().area()/(double)_pos.cols
	// 							, 1.0/(double)_pos.rows);
	// cout << _pos.rows << endl;

	// string DatasetPack = "feature_pos_oc/";
	// DIR* direktori;
	// direktori = opendir(DatasetPack.c_str());

	// struct dirent *str_dr;
	// str_dr = readdir(direktori);

	// while(str_dr != NULL){
	// 	stringstream ss(str_dr->d_name);
	// 	if(ss.str().find("mata_") == std::string::npos){
	// 		str_dr = readdir(direktori);
	// 		continue;
	// 	}

	// 	stringstream namaFile;
	// 	namaFile << "feature_pos_oc/" << (str_dr->d_name);

	// 	// cout << namaFile.str().c_str() << endl;
		
	// 	FileStorage _fsDir(namaFile.str().c_str(), FileStorage::READ);
	// 	Mat _tempData;
	// 	_fsDir["feature"] >> _tempData;
	// 	_fsDir.release();

	// 	stringstream _s;
	// 	_s << "feature_pos_oc/" << (str_dr->d_name);

	// 	Mat _tempData1;
	// 	_tempData.convertTo(_tempData1, CV_64F);
		
	// 	FileStorage _fswrite(_s.str().c_str(), FileStorage::WRITE);
	// 	_fswrite << "feature" << _tempData1;
	// 	_fswrite.release();

	// 	str_dr = readdir(direktori);
	// }



// #include<iostream> 
// using namespace std; 

// class Test 
// { 
// private: 
// int x; 
// int y; 
// int z;
// public: 
// Test(){};
// Test(int x, int y, char ToZ) { 
//   int oZ = 0;
//   if(ToZ == 'o') oZ = 17;
//   loadZ(oZ-3);
//   Test _t(x,y,z);
//   *this = _t;} 

// Test(int _x, int _y, int _z) { x = _x; _y = y; } 
// void loadZ(int _z){z = _z;}

// void destroy() { delete this; } 
// void print() { cout << "x = " << x << endl;
//              	cout << "y = " << y << endl;
//                cout << "z = " << z << endl;} 
// }; 

// int main() 
// { 
// Test _mt(2,3,'o');
//   _mt.print();
// return 0; 
// } 



// Mat _pos;
// FileStorage _fpos("/home/udiro/Videos/Images_optic_cup/8 juni 2019/posNeg/pos_mata_14.yaml",
// 				FileStorage::READ);
// _fpos["feature"] >> _pos;
// _pos.convertTo(_pos, CV_64F);
// _fpos.release();

// resize(_pos, _pos, Size(), 1.0/(double)_pos.cols
// 							, (double)_pos.size().area()/(double)_pos.rows);

// Mat _neg;
// FileStorage _fneg("/home/udiro/Videos/Images_optic_cup/8 juni 2019/posNeg/neg_mata_13.yaml",
// 				FileStorage::READ);
// _fneg["feature"] >> _pos;
// _neg.convertTo(_neg, CV_64F);
// _fneg.release();

// resize(_neg, _neg, Size(), 1.0/(double)_neg.cols
// 							, (double)_neg.size().area()/(double)_neg.rows);
// awal.push_back(_pos);
// awal.push_back(_neg);


// int main(){
// 	std::pair<vector<Mat>, vector<Mat> > croppedImagePN;
// 	std::pair<vector<string>, vector<string> > _indexCroppedPN;
// 	takeYAMLdata(croppedImagePN.first, _indexCroppedPN.first, "pos_oc.yaml", "feature_pos_oc/", "pos");
// 	takeYAMLdata(croppedImagePN.second, _indexCroppedPN.second, "neg_oc.yaml", "feature_neg_oc/", "neg");

// #ifdef POSITIVE
// 	for(int s=0; s<croppedImagePN.first.size(); s++){
// 		imshow(_indexCroppedPN.first[s], croppedImagePN.first[s]);
// 		HOG_feature(croppedImagePN.first[s], _indexCroppedPN.first[s], "pos");
// #ifdef DEBUG	
// 		while(true){
// 			if(waitKey(10) == 27){
// 				destroyWindow(_indexCroppedPN.first[s]);
// 				break;
// 			}
// 		}
// #endif
// 	}
// #endif


// #ifdef NEGATIVE
// 	for(int s=0; s<croppedImagePN.second.size(); s++){
// 		imshow(_indexCroppedPN.second[s], croppedImagePN.second[s]);
// 		HOG_feature(croppedImagePN.second[s], _indexCroppedPN.second[s], "neg");
// #ifdef DEBUG	
// 		while(true){
// 			if(waitKey(10) == 27){
// 				destroyWindow(_indexCroppedPN.second[s]);
// 				break;
// 			}
// 		}
// #endif
// 	}
// #endif
// }
