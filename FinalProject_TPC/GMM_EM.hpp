/*
 Author:

 	(1) Sudiro 
 	   [at] SudiroEEN@gmail.com

*/


#ifndef GMM_EM_HPP
#define GMM_EM_HPP

#include<opencv2/opencv.hpp>
#include<iostream>
#include<dirent.h>

using namespace std;
using namespace cv;

class GMM{
private:
	int nKluster;
	int nData;
	int _dimensi;

	vector<Mat> _data_training;
	vector<Mat> mu_k;
	vector<Mat> Sigma_k;
	vector<Mat> _w_i_k;
	vector<Mat> _pdf_k;

	vector<double> _alpha_k;
public:
   GMM();
	GMM(int nKluster, vector<Mat> dataset, vector<Mat> initialR);
	GMM(int jmlKluster, string DatasetPack, int storage, vector<Mat> initialR);
   GMM(int jmlKluster, Mat frameToBeDataset, vector<Mat> initialR);
   GMM(int jmlKluster, Mat frameToBeDataset, int partX, int partY, int strideX, int strideY , vector<Mat> initialR);
	
	bool isConvergence(vector<double> bobot_alpha_k, vector<Mat> _miyu, vector<Mat> _sigma, double& before_log);
	void train(int iterasi, string saveToYAML);
	double _PDF(Mat datum, Mat rerata, Mat covariance);
	
   void loadConfig(string configYAML);
   int predict(const Mat& datum_);
	
	void scanFrameAsDataset(Mat _frame, int type);
   void loadDatasetPack(string DatasetFile, int storage);
	
	
	enum STATE{
   	STATE_COLLECT = 0,
   	STATE_TRAIN = 1,
   	STATE_PREDICT = 2,
   	STATE_OTHER = 3
   };
   
   enum STORAGE{
      STORAGE_FILE = 0,
      STORAGE_FOLDER = 1
   };
   
   enum TYPE_DATASET{
      TYPE_PIXEL = 0,  // color segmentation
      TYPE_RECT = 1    // object detection
   };

};

#endif
