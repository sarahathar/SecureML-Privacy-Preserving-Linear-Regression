#include <Eigen/Dense>
#include <iostream>
#include <set>
#include <time.h>
#include <fstream>
#include <vector>
#include <math.h>

using namespace std;
using namespace Eigen;

// N * E = B * T
#define N 128*150	// number of training data samples
#define D 784	// number of features
#define T 150	// total number of iterations per epoch
#define B_size 128	// size of mini-batch
#define ALPHA 128	// 2^7
#define epoch 1
#define SCALING_FACTOR 8192

typedef Matrix<unsigned long long int, Dynamic, Dynamic> MATRIX;
typedef Matrix<double, Dynamic, Dynamic> MATRIXd;
typedef Matrix<unsigned long long int, Dynamic, 1, ColMajor> ColVectorXi64;
typedef Matrix<double, Dynamic, 1, ColMajor> ColVectorXd;
typedef Matrix<double, 1, Dynamic, RowMajor> RowVectorXd;
typedef Matrix<unsigned long long int, 1, Dynamic, RowMajor> RowVectorXi64;


// Simulates dditive shares created by the clients and distributed to the two servers.
void genDataShares(MATRIX& X0, MATRIX& X1, MATRIX& Y0, MATRIX& Y1);

// Initialize W0 and W1 to be random
void initializeW(MATRIX& W0, MATRIX& W1);

// Simulates generation of U, V, Z multiplication triplets
void genUVZmultTriplets(MATRIX& U0, MATRIX& U1, MATRIX& V0, MATRIX& V1, MATRIX& Z0, MATRIX& Z1);

// Simulates generation of V', Z' multiplication triplets
void genVZprimeMultTriplets(MATRIX& UB0, MATRIX& UB1, MATRIX& Vp0, MATRIX& Vp1, MATRIX& Zp0, MATRIX& Zp1);

// Returns set of random indices corresponding to training data for current minibatch(iteration)
set<int> createMiniBatch(set<int>& usedIndices);

// Reconstructs M, by adding MO and M1
void reconstruct(MATRIX M0, MATRIX M1, MATRIX& M);

void maskMatrix(MATRIX Data, MATRIX Mask, MATRIX& Final);

void computeDifference(MATRIX& Dif, MATRIX Y_, MATRIX Y);

int reverse_int(int i);

void read_MNIST_data(bool train, vector<vector<unsigned long long int>>& vec, int& number_of_images, int& number_of_features);

void read_MNIST_labels(bool train, vector<unsigned long long int>& vec);

void read_MNIST_data_test(bool train, vector<vector<double>>& vec, int& number_of_images, int& number_of_features);

void read_MNIST_labels_test(bool train, vector<double>& vec);

void secretShareData(MATRIX& X, MATRIX& Y, MATRIX& X0, MATRIX& X1, MATRIX& Y0, MATRIX& Y1);


int main() {

	cout << unsigned long int(-28145) % 4294967296 << endl;
	
	// READ TRAINING DATA
	vector<vector<unsigned long long int>> training_data;
	int param_n= N; int param_d= D;
	read_MNIST_data(true, training_data, param_n, param_d);
	MATRIX X(param_n, param_d);
	for(int i = 0; i < training_data.size(); i++){
        X.row(i) << Map<RowVectorXi64>(training_data[i].data(), training_data[i].size());
    }
	X *= SCALING_FACTOR;
	X /= 255;
	//cout << X << endl;
	//cout << X.rows() << " " << X.cols() << endl;
	/*for (int i = 0 ; i < 784 ; ++i) {
		cout << training_data.at(0).at(i) << " ";
	}
	cout << endl<<  X.row(0) << endl;*/

	// READ TRAINING LABEL
	vector<unsigned long long int> training_labels;
	read_MNIST_labels(true, training_labels);
	MATRIX Y(N, 1);
	//for (int i = 0; i < training_labels.size(); ++i) {
	//	Y(i) = training_labels.at(i);
	//}
	Y << Map<ColVectorXi64>(training_labels.data(), training_labels.size());
	Y *= SCALING_FACTOR;
	Y /= 10;

	cout << sizeof(unsigned long long int) << endl;
	cout << sizeof(uint64_t) << endl;

	// READ TESTING DATA
	vector<vector<double> > testing_data;
	int n_;
	//param_n= N; param_d= D;
	read_MNIST_data_test(false, testing_data, n_, param_d);
	MATRIXd Xt(n_, param_d);
	for(int i = 0; i < testing_data.size(); i++){
        Xt.row(i) << Map<RowVectorXd>(testing_data[i].data(), testing_data[i].size());
    }
	//Xt *= SCALING_FACTOR;		
	Xt /= 255;

	// READING TESTING LABEL
	vector<double> testing_labels;
	read_MNIST_labels_test(false, testing_labels);
	MATRIXd Yt(n_, 1);
	//for (int i = 0; i < training_labels.size(); ++i) {
	//	Y(i) = training_labels.at(i);
	//}
	Yt << Map<ColVectorXd>(testing_labels.data(), testing_labels.size());
	//Yt *= SCALING_FACTOR;		
	//Yt /= 10;


	/*for (long int n : training_labels) {
		cout << n << endl;
	}
	cout << training_labels.size() << endl;*/
	//cout << Yt << endl;
	//cout << Y.rows() << " " << Y.cols() << endl;

	//print out numbers
	/*int c = 0;
	for (int i = 0; i < 784; ++i) {
		if (c % 28 == 0) {
			cout << endl;
		}
		cout << X(2, i) << " ";
		++c;
	}
	cout << training_labels.at(2) << endl;*/
	//cout << Xt.row(0) << endl;



	/*START OF PROTOCOL*/

	// shares of training data
	clock_t start = clock();
	MATRIX X0(N, D);	MATRIX X1(N, D);
	MATRIX Y0(N, 1);	MATRIX Y1(N, 1);
	//genDataShares(X0, X1, Y0, Y1);
	secretShareData(X, Y, X0, X1, Y0, Y1);
	

	// tests:
	/*cout << "X0 MATRIX:\n" << X0 << endl;
	cout << "X1 MATRIX:\n" << X1 << endl;
	cout << "<X>:\n" << X0 + X1 << endl;*/
	/*cout << "Y0 MATRIX:\n" << Y0 << endl;
	cout << "Y1 MATRIX:\n" << Y1 << endl;
	cout << "<Y>:\n" << Y0 + Y1 << endl;*/

	// coefficient vertor w
	MATRIX W0(D, 1);	MATRIX W1(D, 1);
	initializeW(W0, W1);

	// tests:
	/*cout << "W0:\n" << W0 << endl;	
	cout << "W1:\n" << W1 << endl;*/

	// generate shared U, V, Z multiplication triplets
	MATRIX U0(N, D);	MATRIX U1(N, D);
	MATRIX V0(D, T);	MATRIX V1(D, T);
	MATRIX Z0(N, T);	MATRIX Z1(N, T);
	genUVZmultTriplets(U0, U1, V0, V1, Z0, Z1);

	// tests:
	/*cout << "MATRIX U0: \n" << U0 << endl;
	cout << "MATRIX U1: \n" << U1 << endl;
	cout << "MATRIX V0: \n" << V0 << endl;
	cout << "MATRIX V1: \n" << V1 << endl;
	cout << "MATRIX Z0: \n " << Z0 << endl;
	cout << "MATRIX Z1: \n " << Z1 << endl;*/

	
	// * STEP 1: each party runs Ei = Xi - Ui masking
	MATRIX E0(N, D);	MATRIX E1(N, D);	MATRIX E(N, D);
	maskMatrix(X0, U0, E0);
	maskMatrix(X1, U1, E1);
	reconstruct(E0, E1, E);		// |?| how to code how each party does this independently

	// tests:
	/*cout << "MATRIX E0:\n" << E0 << endl;
	cout << "MATRIX E1:\n" << E1 << endl;
	cout << "MATRIX E: \n" << E << endl;*/


	// * STEP 2: start of each iteration
	set<int> usedIndices;
	for (int j = 0; j < T; ++j) {

		//cout << "Iteration " << j+1 << "..." << endl;

		// * STEP 3: select minibatch
		set<int> minibatch = createMiniBatch(usedIndices); // select minibatch indices

		// tests:
		/*cout << "minibatch:" << endl;
		for (int n : minibatch) {
			cout << n << " ";
		}
		cout << endl;*/

		// submatrices size B
		MATRIX EB(B_size, D);
		MATRIX UB0(B_size, D);	MATRIX UB1(B_size, D);
		MATRIX XB0(B_size, D);	MATRIX XB1(B_size, D);
		MATRIX YB0(B_size, 1);	MATRIX YB1(B_size, 1);
		MATRIX ZB0(B_size, T);	MATRIX ZB1(B_size, T);

		int r = 0;	int k = 0;
		for (int n : minibatch) {
			k = n - 1;
			EB.row(r) = E.row(k);
			UB0.row(r) = U0.row(k);
			UB1.row(r) = U1.row(k);
			XB0.row(r) = X0.row(k);
			XB1.row(r) = X1.row(k);
			YB0.row(r) = Y0.row(k);
			YB1.row(r) = Y1.row(k);
			ZB0.row(r) = Z0.row(k);
			ZB1.row(r) = Z1.row(k);
			r++;
		}

		// tests:
		/*cout << "MATRIX EB:\n" << EB << endl;
		cout << "MATRIX UB0:\n" << UB0 << endl;
		cout << "MATRIX UB1:\n" << UB1 << endl;
		cout << "MATRIX XB0:\n" << XB0 << endl;
		cout << "MATRIX XB1:\n" << XB1 << endl;
		cout << "MATRIX YB0:\n" << YB0 << endl;
		cout << "MATRIX YB1:\n" << YB1 << endl;
		cout << "MATRIX ZB0:\n" << ZB0 << endl;
		cout << "MATRIX ZB1:\n" << ZB1 << endl;*/


		// generate V' and Zp multiplication triplets based on minibatch indices
		MATRIX Vp0(B_size, T);	MATRIX Vp1(B_size, T);
		MATRIX Zp0(D, T);		MATRIX Zp1(D, T);
		genVZprimeMultTriplets(UB0, UB1, Vp0, Vp1, Zp0, Zp1);

		// tests:
	/*	cout << "matrix Vp0: \n" << Vp0 << endl;
		cout << "matrix Vp1: \n" << Vp1 << endl;
		cout << "matrix Zp0: \n" << Zp0 << endl;
		cout << "matrix Zp1: \n" << Zp1 << endl;*/


		//// * STEP 4: each party computes Fji = wi - V[j]i	jth iteration
		MATRIX F0(D, 1);	MATRIX F1(D, 1);	MATRIX F(D, 1);
		MATRIX V0j = V0.col(j);	
		MATRIX V1j = V1.col(j);
		maskMatrix(W0, V0j, F0);
		maskMatrix(W1, V1j, F1);
		reconstruct(F0, F1, F);

		// tests:
		/*cout << "MATRIX V0j: \n" << V0j << endl;
		cout << "MATRIX F0:\n" << F0 << endl;
		cout << "MATRIX V1j: \n" << V1j << endl;
		cout << "MATRIX F1:\n" << F1 << endl;
		cout << "MATRIX F:\n" << F << endl;*/


		// * STEP 5: calculate predicted output
		MATRIX Ypred_0 = -0 * (EB * F) + (XB0 * F) + (EB * W0) + ZB0.col(j);	// replace with j
		MATRIX Ypred_1 = -1 * (EB * F) + (XB1 * F) + (EB * W1) + ZB1.col(j);	// replace with j

		// tests:
		/*cout << "EB\n" << EB << endl;
		cout << "XB0\n" << XB0<< endl;
		cout << "XB1\n" << XB1<< endl;
		cout << "ZB0.col(0)\n" << ZB0.col(0) << endl;
		cout << "ZB1.col(0)\n" << ZB1.col(0) << endl;
		cout << "Ypred_0:\n" << Ypred_0 << endl;
		cout << "Ypred_1:\n" << Ypred_1 << endl;*/


		// * STEP 6: 
		MATRIX D0; MATRIX D1;
		computeDifference(D0, Ypred_0, YB0);
		computeDifference(D1, Ypred_1, YB1);

		// tests:
		/*cout << "Ypred_0:\n" << Ypred_0 << endl;
		cout << "YB0 \n" << YB0 << endl;
		cout << "D0:\n" <<D0 << endl;
		cout << "Ypred_1:\n" << Ypred_1 << endl;
		cout << "YB1 \n" << YB1 << endl;
		cout << "D1:\n" <<D1 << endl;*/


		// * STEP 7: mask D0, and D1
		MATRIX Fp0;	MATRIX Fp1; MATRIX Fp;
		MATRIX Vp0j = Vp0.col(j);
		MATRIX Vp1j = Vp1.col(j);
		maskMatrix(D0, Vp0j, Fp0);
		maskMatrix(D1, Vp1j, Fp1);
		reconstruct(Fp0, Fp1, Fp);
		


		// tests:
		/*cout << "MATRIX Fp0: \n" << Fp0 << endl;
		cout << "MATRIX Fp1: \n" << Fp1 << endl;
		cout << "MATRIX Fp: \n" << Fp << endl;*/


		// * STEP 8: compute shares of delta
		MATRIX delta0 = -0 * (EB.transpose() * Fp) + (XB0.transpose() * Fp) + (EB.transpose() * D0) + Zp0.col(j);
		MATRIX delta1 = -1 * (EB.transpose() * Fp) + (XB1.transpose() * Fp) + (EB.transpose() * D1) + Zp1.col(j);

		// tests:
		/*cout << "Before truncation:" << endl;
		cout << "delta0:\n" << delta0 << endl;
		cout << "delta1:\n" << delta1 << endl;*/

		// *STEP 9: truncation
		int l = 64;	int x = 32;	int d = 13;

		for (int i = 0; i < D; ++i) {
			delta0(i) = delta0(i) >> d;
			delta1(i) = delta1(i) >> d;
		}

		// tests:
		/*cout << "After truncation:" << endl;
		cout << "delta0:\n" << delta0 << endl;
		cout << "delta1:\n" << delta1 << endl;*/

		// *STEP 10: 
		/*cout << "W0:\n" << W0 << endl;
		cout << "W1:\n" << W1 << endl;*/
		//cout << (B_size * ALPHA) << endl;

		W0 = W0 - (delta0 / (B_size * ALPHA));
		W1 = W1 - (delta1 / (B_size * ALPHA));

		/*cout << "W0:\n" << W0 << endl;
		cout << "W1:\n" << W1 << endl;*/

	}

	// *STEP 11: Reconstruct W
	MATRIX W(D,1);
    reconstruct(W0, W1, W);

	// tests:
	//cout << "FINAL W:\n" << W << endl;

	clock_t end = clock();
	double elapsed = (double(end) - double(start)) / CLOCKS_PER_SEC;
	//cout << "Time measured : " << elapsed << "seconds.\n";


	ColVectorXd Wd;
	W /= SCALING_FACTOR;
	Wd = W.cast <double>();


	//cout << Yt << endl;

	/*cout << "Xt size: " << Xt.rows() << " " << Xt.cols() << endl;	
	cout << "Wd size: " << Wd.rows() << " " << Wd.cols() << endl;*/
	// Xt will always be 10000 x 784, make sure Wd is also 784
	
	ColVectorXd prediction = Xt * Wd;

	prediction /= SCALING_FACTOR;
	prediction *= 10;
    n_ = Yt.rows();	// 10000
	double temp1;
	long int temp2, temp3;
    prediction = round(prediction.array());

	cout << "SIZE: " << n_ << endl;


	cout << prediction << endl;
	int count = 0; int count_zero = 0;
    for (int i = 0; i < n_ ; i++){
		temp3 = (long long int)prediction(i);
		
		//temp1 = temp3/(double)pow(2,20+8);
		temp2 = (long long int)Yt(i);
		//if (temp2 == 0) {
		//	count_zero++;
		//}



		//if(temp3>0.5 && temp2 == 1){
		//	count++;
		//}
		//else if (temp3 < 0.5 && temp2 == 0) {
		//	count++;
		//}
		
		if(prediction(i) == Yt(i))
            count++;

		//cout << "pred: " << prediction(i) << "	test label: " << Yt(i) << endl;
  //      
    }

	cout << "num zeros: " << count_zero << endl;

	cout << "num correct" << count << endl;
	double accuracy = count/((double) n_);
	cout << "Accuracy on testing the trained model is " << accuracy * 100 << endl;


} // end of main()


void secretShareData(MATRIX& X, MATRIX& Y, MATRIX& X0, MATRIX& X1, MATRIX& Y0, MATRIX& Y1) {
	unsigned long long int n, n0;

	for (int i = 0; i < N; ++i) {
		for (int j = 0; j < D; j++) {
			n = X(i, j);
			n0 = rand();
			X0(i, j) = n0;
			X1(i, j) = (n - n0);
		}
	}

	for (int i = 0; i < N; ++i) {
		n = Y(i);
		n0 = rand();
		Y0(i) = n0;
		Y1(i) = (n - n0)  ;	// add mod 2l ?
	}
}

void genDataShares(MATRIX& X0, MATRIX& X1, MATRIX& Y0, MATRIX& Y1) {
	unsigned long long int n, n0, n1;
	for (int i = 0; i < N; ++i) {
		for (int j = 0; j < D; j++) {
			n = rand();
			n0 = rand();
			n1 = (n - n0); 		// add mod 2l ?
			X0(i, j) = n0;
			X1(i, j) = n1;
		}
	}

	for (int i = 0; i < N; ++i) {
		n = rand();
		n0 = rand();
		n1 = (n - n0) ;		// add mod 2l ?
		Y0(i) = n0;
		Y1(i) = n1;
	}
}

void initializeW(MATRIX& W0, MATRIX& W1) {
	for (int i = 0; i < D; ++i) {		
		W0(i) = rand();
		W1(i) = rand();
	}
}

void genUVZmultTriplets(MATRIX& U0, MATRIX& U1, MATRIX& V0, MATRIX& V1, MATRIX& Z0, MATRIX& Z1) {
	
	unsigned long long int n, n0;
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < D; j++) {
			n = rand();
			n0 = rand();
			U0(i, j) = n0;
			U1(i, j) = (n - n0);
		}
	}

	for (int i = 0; i < D; i++) {
		for (int j = 0; j < T; j++) {
			n = rand();
			n0 = rand();
			V0(i, j) = n0;
			V1(i, j) = (n - n0);
		}
	}

	Z0 = U0 * V0;
	Z1 = U1 * V1;
}

void genVZprimeMultTriplets(MATRIX& UB0, MATRIX& UB1, MATRIX& Vp0, MATRIX& Vp1, MATRIX& Zp0, MATRIX& Zp1) {
	MATRIX UBT0 = UB0.transpose();
	MATRIX UBT1 = UB1.transpose();

	unsigned long long int n, n0;
	for (int i = 0; i < B_size; i++) {
		for (int j = 0; j < T; j++) {
			n = rand();
			n0 = rand();
			Vp0(i, j) = n0;
			Vp1(i, j) = n - n0;
		}
	}

	Zp0 = UBT0 * Vp0;
	Zp1 = UBT1 * Vp1;
	/*cout << "MATRIX UBT0:\n" << UBT0 << endl;
	cout << "MATRIX UBT1:\n" << UBT1 << endl;*/
}

void reconstruct(MATRIX M0, MATRIX M1, MATRIX& M) {	M = M0 + M1;}

void maskMatrix(MATRIX Data, MATRIX Mask, MATRIX& Final) {	Final = Data - Mask;}

set<int> createMiniBatch(set<int>& usedIndices) {
	set<int> minibatch;
	int num = 0;
	//srand(time(NULL));
	while (minibatch.size() != B_size) {
		num = (rand() % (N - 1 + 1)) + 1;
		if (usedIndices.count(num) == 0) {
			usedIndices.insert(num);
			minibatch.insert(num);
		}
	}
	return minibatch;
}

void computeDifference(MATRIX& Dif, MATRIX Y_, MATRIX Y) {    Dif = Y_ - Y;}


int reverse_int(int i){
    unsigned char ch1, ch2, ch3, ch4;
    ch1 = i & 255;
    ch2 = (i >> 8) & 255;
    ch3 = (i >> 16) & 255;
    ch4 = (i >> 24) & 255;
    return ((int) ch1 << 24) + ((int) ch2 << 16) + ((int) ch3 << 8) + ch4;
}

void read_MNIST_data(bool train, vector<vector<unsigned long long int>> &vec, int& number_of_images, int& number_of_features){
    std::ifstream file;
	if (train == true)
		file.open("/Users/ariad/Documents/train-images-idx3-ubyte", std::ios::binary);
	else
		file.open("/Users/ariad/Documents/t10k-images-idx3-ubyte", std::ios::binary);
    
    if(!file){
        std::cout<<"Unable to open file";
        return;
    }
    if (file.is_open()){
        int magic_number = 0;
        int n_rows = 0;
        int n_cols = 0;
        file.read((char*) &magic_number, sizeof(magic_number));
        magic_number = reverse_int(magic_number);
        file.read((char*) &number_of_images, sizeof(number_of_images));
        number_of_images = reverse_int(number_of_images);
        if(train == true)
            number_of_images = N;
        file.read((char*) &n_rows, sizeof(n_rows));
        n_rows = reverse_int(n_rows);
        file.read((char*) &n_cols, sizeof(n_cols));
        n_cols = reverse_int(n_cols);
        number_of_features = n_rows * n_cols;
        std::cout << "Number of Images: " << number_of_images << std::endl;
        std::cout << "Number of Features: " << number_of_features << std::endl;
        for(int i = 0; i < number_of_images; ++i){
            std::vector<unsigned long long int> tp;
            for(int r = 0; r < n_rows; ++r)
                for(int c = 0; c < n_cols; ++c){
                    unsigned char temp = 0;
                    file.read((char*) &temp, sizeof(temp));
                    tp.push_back(((unsigned long long int) temp));
                }
            vec.push_back(tp);
        }
    }
}


void read_MNIST_labels(bool train, vector<unsigned long long int> &vec){
    std::ifstream file;
	if (train == true) {
		file.open("/Users/ariad/Documents/train-labels-idx1-ubyte", std::ios::binary);
	}
	else {
		file.open("/Users/ariad/Documents/t10k-labels-idx1-ubyte", std::ios::binary);
	}
    if(!file){
        std::cout << "Unable to open file";
        return;
    }
    if (file.is_open()){
        int magic_number = 0;
        int number_of_images = 0;
        file.read((char*) &magic_number, sizeof(magic_number));
        magic_number = reverse_int(magic_number);
        file.read((char*) &number_of_images, sizeof(number_of_images));
        number_of_images = reverse_int(number_of_images);
		cout << "number of images inside MNIST LAbels: " << number_of_images << endl;

        if(train == true)
            number_of_images = N;
        for(int i = 0; i < number_of_images; ++i){
            unsigned char temp = 0;
            file.read((char*) &temp, sizeof(temp));
			//vec.push_back((unsigned long int) temp);
			
            if((unsigned long long int) temp == 0)
                vec.push_back((unsigned long long int) 0);
            else
                vec.push_back((unsigned long long int) 1);
        }
    }
}


void read_MNIST_data_test(bool train, vector<vector<double>> &vec, int& number_of_images, int& number_of_features){
    std::ifstream file;
	if (train == true)
		file.open("/Users/ariad/Documents/train-images-idx3-ubyte", std::ios::binary);
	else
		file.open("/Users/ariad/Documents/t10k-images-idx3-ubyte", std::ios::binary);
    
    if(!file){
        std::cout<<"Unable to open file";
        return;
    }
    if (file.is_open()){
        int magic_number = 0;
        int n_rows = 0;
        int n_cols = 0;
        file.read((char*) &magic_number, sizeof(magic_number));
        magic_number = reverse_int(magic_number);
        file.read((char*) &number_of_images, sizeof(number_of_images));
        number_of_images = reverse_int(number_of_images);
        if(train == true)
            number_of_images = N;
        file.read((char*) &n_rows, sizeof(n_rows));
        n_rows = reverse_int(n_rows);
        file.read((char*) &n_cols, sizeof(n_cols));
        n_cols = reverse_int(n_cols);
        number_of_features = n_rows * n_cols;
        std::cout << "Number of Images: " << number_of_images << std::endl;
        std::cout << "Number of Features: " << number_of_features << std::endl;
        for(int i = 0; i < number_of_images; ++i){
            std::vector<double> tp;
            for(int r = 0; r < n_rows; ++r)
                for(int c = 0; c < n_cols; ++c){
                    unsigned char temp = 0;
                    file.read((char*) &temp, sizeof(temp));
                    tp.push_back(((double) temp));
                }
            vec.push_back(tp);
        }
    }
}

void read_MNIST_labels_test(bool train, vector<double> &vec){
    std::ifstream file;
	if (train == true) {
		file.open("/Users/ariad/Documents/train-labels-idx1-ubyte", std::ios::binary);
	}
	else {
		file.open("/Users/ariad/Documents/t10k-labels-idx1-ubyte", std::ios::binary);
	}
    if(!file){
        std::cout << "Unable to open file";
        return;
    }
    if (file.is_open()){
        int magic_number = 0;
        int number_of_images = 0;
        file.read((char*) &magic_number, sizeof(magic_number));
        magic_number = reverse_int(magic_number);
        file.read((char*) &number_of_images, sizeof(number_of_images));
        number_of_images = reverse_int(number_of_images);
		cout << "number of images inside MNIST LAbels: " << number_of_images << endl;

        if(train == true)
            number_of_images = N;
        for(int i = 0; i < number_of_images; ++i){
            unsigned char temp = 0;
            file.read((char*) &temp, sizeof(temp));
			//vec.push_back((double) temp);
			
            if((double) temp == 0)
                vec.push_back((double) 0);
            else
                vec.push_back((double) 1);
        }
    }
}

/*
	Modulus
	Epochs, testing accuracy
*/