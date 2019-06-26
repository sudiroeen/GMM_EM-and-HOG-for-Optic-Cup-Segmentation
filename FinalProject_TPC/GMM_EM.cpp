/*
 Author:

 	(1) Sudiro 
 		   [at] SudiroEEN@gmail.com

 	(2) M Izzarrasyadi Wachid
 		[at] izzarrasyadi@gmail.com

*/

#include "GMM_EM.hpp"

GMM::GMM(int jmlKluster, vector<Mat> dataset, vector<Mat> initialR)
	:nKluster(jmlKluster), nData(dataset.size())
	, _data_training(dataset)
{
	_dimensi = initialR[0].rows;
	mu_k.resize(nKluster);
	Sigma_k.resize(nKluster);
	_alpha_k.resize(nKluster);

	_w_i_k.resize(nKluster);
	_pdf_k.resize(nKluster);

	for(int k=0; k<jmlKluster; k++){
		_w_i_k[k] = Mat::zeros(1, nData, CV_64FC1);
		_pdf_k[k] = Mat::zeros(1, nData, CV_64FC1);

		_alpha_k[k] = 1.0/jmlKluster;
	}

	mu_k = initialR;

	for(int k=0; k<nKluster; k++){
		Sigma_k[k] = Mat::zeros(initialR[0].rows, initialR[0].rows, CV_64FC1);
		for(int n=0; n<dataset.size(); n++){
			Sigma_k[k] += (dataset[n] - mu_k[k]) * (dataset[n] - mu_k[k]).t();
		}
		Sigma_k[k] /= dataset.size();
	}

	double sigma_alphaXpdf = 0.0;
	for(int k=0; k<nKluster; k++){
		double* _pix_wi_k = _w_i_k[k].ptr<double>(0);
		for(int n=0; n<nData; n++){
			_pix_wi_k[n] = _alpha_k[k] * _PDF(_data_training[n], mu_k[k], Sigma_k[k]);
			sigma_alphaXpdf += _pix_wi_k[n];
		}
	}

	for(int k=0; k<nKluster; k++){
		if(sigma_alphaXpdf != 0.0)
			_w_i_k[k] /= sigma_alphaXpdf;
	}

}

GMM::GMM(){
	cout << "GMM object created" << endl;
}

GMM::GMM(int jmlKluster, string DatasetPack, int storage, vector<Mat> initialR)
{
	switch(storage){
		case STORAGE_FILE:
			loadDatasetPack(DatasetPack, STORAGE_FILE);
			break;
		case STORAGE_FOLDER:
			loadDatasetPack(DatasetPack, STORAGE_FOLDER);
			break;
	}

	GMM _gausFile(jmlKluster, _data_training, initialR);
	*this = _gausFile;
}

GMM::GMM(int jmlKluster, Mat frameToBeDataset, vector<Mat> initialR){
	scanFrameAsDataset(frameToBeDataset, TYPE_PIXEL);
	GMM _gaussFrame(jmlKluster, _data_training, initialR);
	*this = _gaussFrame;
}

GMM::GMM(int jmlKluster, Mat frameToBeDataset, int partX, int partY,
		 int strideX, int strideY , vector<Mat> initialR)
{

}


// prefered now to object detection project
void GMM::loadDatasetPack(string DatasetPack, int storage){
	switch(storage){
		case STORAGE_FILE:{
			FileStorage dfs(DatasetPack, FileStorage::READ);
			dfs["size"] >> nData;
			dfs["_dimensi"] >> _dimensi;
			
			for(int c=0; c<nData; c++){
				stringstream ss;
				ss << c;

				Mat _mat;
				dfs["dataset_"+ss.str()] >> _mat;
				_data_training.push_back(_mat);
			}
			dfs.release();
			break;
		}
		case STORAGE_FOLDER:{
			DIR* direktori;
			direktori = opendir(DatasetPack.c_str());

			struct dirent *str_dr;
			str_dr = readdir(direktori);

			while(str_dr != NULL){
				stringstream ss(str_dr->d_name);
				if(ss.str().find("dataset_") == std::string::npos){
					str_dr = readdir(direktori);
					continue;
				}

				stringstream namaFile;
				namaFile << DatasetPack << (str_dr->d_name);

				// cout << namaFile.str().c_str() << endl;
				
				FileStorage _fsDir(namaFile.str().c_str(), FileStorage::READ);
				Mat _tempData;
				_fsDir["feature"] >> _tempData;
				_fsDir.release();

				_tempData.convertTo(_tempData, CV_64F);
				resize(_tempData, _tempData, Size(), 1.0/(double)_tempData.cols
											, (double)_tempData.size().area()/(double)_tempData.rows);
				_data_training.push_back(_tempData);

				str_dr = readdir(direktori);
			}
			// cout << _data_training.size() << endl;
			break;
		}
	}
}

void GMM::scanFrameAsDataset(Mat _frame, int type){
	switch(type){
		case TYPE_PIXEL:{
			for(int r=0; r<_frame.rows; r++){
				Vec3b* p_ixel = _frame.ptr<Vec3b>(r);
				for(int c=0; c<_frame.cols; c++){
					double B = (double)p_ixel[c][2];
					double G = (double)p_ixel[c][1];
					double R = (double)p_ixel[c][0];

					_data_training.push_back((Mat_<double>(int(_frame.channels()),1) << B, G, R));
				}
			}
		break;
		}

		case TYPE_RECT:{
		break;
	    }
	}
}

double GMM::_PDF(Mat datum, Mat rerata, Mat covariance){
	if(!determinant(covariance)){
		cout << "covariance matrix singular !!!" << endl;
		covariance = Mat::eye(covariance.size(), CV_64FC1);
		// return -1.0f;
	}
	double PI_detSIGMA = pow(pow(M_PI, _dimensi) * determinant(covariance), 0.5);

	Mat mPANGKAT = -0.5*(datum - rerata).t() * covariance.inv() * (datum - rerata);
	double PANGKAT = mPANGKAT.at<double>(0,0);

	return pow(M_E, PANGKAT)/PI_detSIGMA;
}

bool GMM::isConvergence(vector<double> bobot_alpha_k, vector<Mat> _miyu, vector<Mat> _sigma, double& before_log){
	double current_log = 0.0;
	for(int n=0; n<nData; n++){
		double jmlPDFxAlpha = 0.0;
		for(int k=0; k<nKluster; k++){
			jmlPDFxAlpha += bobot_alpha_k[k] * _PDF(_data_training[n], _miyu[k], _sigma[k]);
		}
		current_log += log10(jmlPDFxAlpha);
	}

	cout << fabs(current_log - before_log) << endl;

	if(fabs(current_log - before_log) < 1e-8)
		return true;

	before_log = current_log;
	return false;
}

void GMM::train(int iterasi, string saveToYAML){
	double log_likely_hood = 0.0;
	for(int i=0; i<iterasi; i++){
		cout << "step " << i << ": ";
		if(isConvergence(_alpha_k, mu_k, Sigma_k, log_likely_hood))
			break;

		double* Nk = new double[nKluster];
		for(int k=0; k<nKluster; k++){
			Nk[k] = 0;
			double* _pix_wi_k = _w_i_k[k].ptr<double>(0);
			Mat wX = Mat::zeros(_dimensi, 1, CV_64FC1);
			Mat gtSigma = Mat::zeros(Sigma_k[k].size(), CV_64FC1);

			for(int n=0; n<nData; n++){
				Nk[k] += _pix_wi_k[n];
				wX += _pix_wi_k[n] * _data_training[n];
			}
			_alpha_k[k] = Nk[k]/nData;
			mu_k[k] = wX/Nk[k];

			for(int n=0; n<nData; n++){
				gtSigma += _pix_wi_k[n] * (_data_training[n] - mu_k[k]) * (_data_training[n] - mu_k[k]).t();
			}
			Sigma_k[k] = gtSigma;

			double sigma_alphaXpdf = 0.0;
			for(int l=0; l<nKluster; l++){
				double* _pix_wi_k = _w_i_k[l].ptr<double>(0);
				for(int n=0; n<nData; n++){
					_pix_wi_k[l] = _alpha_k[l] * _PDF(_data_training[n], mu_k[l], Sigma_k[l]);
					sigma_alphaXpdf += _pix_wi_k[n];
				}
			}
			if(sigma_alphaXpdf != 0.0)
				_w_i_k[k] /= sigma_alphaXpdf;
		}
	}


	FileStorage fs(saveToYAML, FileStorage::WRITE);
	fs << "nKluster" << nKluster;
	fs << "_dimensi" << _dimensi;
	for(int k=0; k<nKluster;k++){
		stringstream ss;
		ss << k;
		fs << "Mean_"+ss.str() << mu_k[k];
		fs << "Cov_"+ss.str() << Sigma_k[k];
		fs << "Alpha_"+ss.str() << _alpha_k[k];
	}
	fs.release();
}

void GMM::loadConfig(string configYAML){
	FileStorage fs(configYAML, FileStorage::READ);
	fs["nKluster"] >> nKluster;

	fs["_dimensi"] >> _dimensi;

	mu_k.resize(nKluster);
	Sigma_k.resize(nKluster);
	_alpha_k.resize(nKluster);

	for(int k=0; k<nKluster; k++){
		stringstream _ss;
		_ss << k;

		fs["Mean_" + _ss.str()] >> mu_k[k];
		fs["Cov_" + _ss.str()] >> Sigma_k[k];
		fs["Alpha_" + _ss.str()] >> _alpha_k[k];
	}
	fs.release();
}

int GMM::predict(const Mat& datum_){
	int kluster = 0;
	Mat _x = datum_;
	double _maksimum = 0.0;
	for(int k=0; k<nKluster; k++){
		double alphaXpdf = _alpha_k[k] * _PDF(_x, mu_k[k], Sigma_k[k]);

		if(alphaXpdf > _maksimum){
			_maksimum = alphaXpdf;
			kluster = k;
		}
	}
	return kluster;


#if 0	
	Mat _x = raw_pixel;
	double sumAlphaXpdf = 0.0;
	Mat xToKluster = Mat::zeros(1, nKluster, CV_64FC1);
	
	double* ptr_xToKluster = xToKluster.ptr<double>(0);

	for(int k=0; k<nKluster; k++){
		sumAlphaXpdf += _alpha_k[k]*_PDF(_x, mu_k[k], Sigma_k[k]);
		ptr_xToKluster[k] = _alpha_k[k]*_PDF(_x, mu_k[k], Sigma_k[k]);
	}

	double _max = 0.0;
	int kluster = -1;
	for(int k=0; k<nKluster; k++){
		double _batas = ptr_xToKluster[k] * log(ptr_xToKluster[k])/sumAlphaXpdf;
		if(_max < _batas){
			_max = _batas;
			kluster = k;
		}
	}
#endif
}

/*
	TODO:
	1. lengkapi semua fitur
	2. ubah more efficient

	#DATASET
		# Collect
			stride -> float, perbandingan antara ukuran image, dengan ukuran crop window

			a. crop di luar, extract HOG di luar
				@ bisa gambar
				@ bisa fitur
			b. input frame, crop langsung
				-> resizer
				-> ukuran window kernel, X dan Y
				-> stride X dan Y
				-> batas banyak dataset

				@ untuk warna, tanpa ukuran window, stride bisa
			c. input folder berisi banyak foto
				-> resizer
				-> ukuran window kernel, X dan Y
				-> stride X dan Y
				-> batas banyak dataset

				@ untuk warna, tanpa ukuran window, stride bisa
			d. input video
				-> resizer
				-> stride Frame
				-> ukuran window kernel, X dan Y
				-> stride X dan Y
				-> batas banyak dataset

				@ untuk warna, tanpa ukuran window, stride bisa
			e. input folder berisi beberapa video
				-> resizer
				-> stride Frame
				-> ukuran window kernel, X, dan Y
				-> stride X dan Y
				-> batas banyak dataset

				@ untuk warna, tanpa ukuran window, stride bisa
	#Predict
		$ Offline
			-> hanya valid untuk warna

			-> memakai LookUpTable
		$ Online
*/
