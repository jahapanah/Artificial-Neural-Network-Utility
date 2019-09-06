#pragma once
#include "Data.h"
#include "Chart.h"
#include <iostream>

using namespace std;



class NeuralNetworkApp
{
public:
	NeuralNetworkApp();
	~NeuralNetworkApp();

	
	void testBackpropagation();
	

private:

};

NeuralNetworkApp::NeuralNetworkApp()
{
}

NeuralNetworkApp::~NeuralNetworkApp()
{
}



inline void NeuralNetworkApp::testBackpropagation()
{
	Data Input("data", "input_breast.csv");  // select the input file for training data
	Data Output("data", "output_breast.csv"); // select the output file for training data

	Data InputTestRNA("data", "input_test_breast.csv");  // select the input file for test data
	Data OutputTestRNA("data", "output_test_breast.csv"); // select the output file for test data

	Data InputValidate("data", "input_validate_breast.csv"); // select the input file for validation data
	Data OutputValidate("data", "output_validate_breast.csv"); // select the output file for validation data


	NormalizationTypesENUM NORMALIZATION_TYPE = NormalizationTypesENUM::MAX_MIN;

	std::vector<std::vector<double>> matrixInput = Input.rawData2Matrix(Input);
	std::vector<std::vector<double>> matrixOutput = Output.rawData2Matrix(Output);

	std::vector<std::vector<double>> matrixInputValidate = OutputValidate.rawData2Matrix(InputValidate);
	std::vector<std::vector<double>> matrixOutputValidate = OutputValidate.rawData2Matrix(OutputValidate);

	std::vector<std::vector<double>> matrixInputTestRNA = OutputTestRNA.rawData2Matrix(InputTestRNA);
	std::vector<std::vector<double>> matrixOutputTestRNA = OutputTestRNA.rawData2Matrix(OutputTestRNA);

	std::vector<std::vector<double>> matrixInputNorm = Input.normalize(matrixInput, NORMALIZATION_TYPE);
	std::vector<std::vector<double>> matrixOutputNorm = Output.normalize(matrixOutput, NORMALIZATION_TYPE);

	std::vector<std::vector<double>> matrixInputValidateNorm = OutputValidate.normalize(matrixInputValidate, NORMALIZATION_TYPE);
	std::vector<std::vector<double>> matrixOutputValidateNorm = OutputValidate.normalize(matrixOutputValidate, NORMALIZATION_TYPE);

	std::vector<std::vector<double>> matrixInputTestRNANorm = OutputTestRNA.normalize(matrixInputTestRNA, NORMALIZATION_TYPE);
	std::vector<std::vector<double>> matrixOutputTestRNANorm = OutputTestRNA.normalize(matrixOutputTestRNA, NORMALIZATION_TYPE);


	NeuralNet testNet;
	//Network initiation of input layer, hidden layer and output layer
	testNet.initNet(30, 40, 1);  
	std::cout << "-------------BACKPROPAGATION INET NET----------------" << std::endl;
	testNet.printNet(testNet);

	NeuralNet trainedNet;
	testNet.setTrainSet(matrixInputNorm);
	testNet.setRealMatrixOutputSet(matrixOutputNorm);
	testNet.setMaxEpochs(1000);  //set epochs value
	testNet.setTargetError(0.03); //set target error
	testNet.setLearningRate(0.25); // set the learning rate
	testNet.setActivationFnc(ActivationFncENUM::HYPERTAN); //set activation function of hidden layer
	testNet.setActivationFncOutputLayer(ActivationFncENUM::HYPERTAN); // set activation function of output layer

	trainedNet = testNet.trainNet(testNet);
	std::cout << std::endl;

	std::cout << "-------------BACKPROPAGATION TRAINED NET----------------" << std::endl;
	testNet.printNet(trainedNet);
	
	
	Chart::plotXData(testNet.getListOfMSE(), "MSE Error", "MSEValue", "Epochs");


	//TRAINING
	
	std::vector<std::vector<double>> matrixOuputRNA = trainedNet.getNetOutputValues(trainedNet);
	Data* temp = new Data();
	std::vector<std::vector<double>> matrixOutputRNADenorm = temp->denormalize(matrixOutput, matrixOuputRNA, NORMALIZATION_TYPE);

	std::vector<std::vector<std::vector<double>>> listOfArraysToJoin;
	listOfArraysToJoin.push_back(matrixOutput);
	listOfArraysToJoin.push_back(matrixOutputRNADenorm);

	std::vector<std::vector<double>> matrixOutputsJoined = temp->joinArrays(listOfArraysToJoin);
	
	
	Chart::plotXYData(matrixOutputsJoined, "Real x Estimated - Training Data", "Actual", "Predicted");
	


   // VALIDATION
	
	trainedNet.setTrainSet(matrixInputValidateNorm);
	trainedNet.setRealMatrixOutputSet(matrixOutputValidateNorm);

	std::vector<std::vector<double>> matrixOutput_Validate = trainedNet.getNetOutputValues(trainedNet);

	std::vector<std::vector<double>> matrixOutputRNADenormValidate = temp->denormalize(matrixOutputValidate, matrixOutput_Validate, NORMALIZATION_TYPE);

	std::vector<std::vector<std::vector<double>>> listOfArraysToJoinValidate;
	listOfArraysToJoinValidate.push_back(matrixOutputValidate);
	listOfArraysToJoinValidate.push_back(matrixOutputRNADenormValidate);

	std::vector<std::vector<double>> matrixOuputsJoinedValidate = temp->joinArrays(listOfArraysToJoinValidate);

	Chart::plotXYData(matrixOuputsJoinedValidate, "Real x Estimated - Validation Data", "Actual", "Predicted");
	
	
	//TEST:
	
	trainedNet.setTrainSet(matrixInputTestRNANorm);
	trainedNet.setRealMatrixOutputSet(matrixOutputTestRNANorm);
	
	std::vector<std::vector<double>> matrixOutputRNATest = trainedNet.getNetOutputValues(trainedNet);
	
	std::vector<std::vector<double>> matrixOutputRNADenormTest = temp->denormalize(matrixOutputTestRNA, matrixOutputRNATest, NORMALIZATION_TYPE);

	std::vector<std::vector<std::vector<double>>> listOfArraysToJoinTest;
	listOfArraysToJoinTest.push_back(matrixOutputTestRNA);
	listOfArraysToJoinTest.push_back(matrixOutputRNADenormTest);

	std::vector<std::vector<double>> matrixOuputsJoinedTest = temp->joinArrays(listOfArraysToJoinTest);
	delete temp;
	Chart::plotXYData(matrixOuputsJoinedTest, "Real x Estimated - Test Data", "Actual", "Predicted");

	std::vector<std::vector<double>>().swap(matrixInput);
	std::vector<std::vector<double>>().swap(matrixOutput);
	std::vector<std::vector<double>>().swap(matrixOutputsJoined);
	std::vector<std::vector<std::vector<double>>>().swap(listOfArraysToJoin);
	std::vector<std::vector<std::vector<double>>>().swap(listOfArraysToJoinValidate);
	std::vector<std::vector<std::vector<double>>>().swap(listOfArraysToJoinTest);
	
	

	}
