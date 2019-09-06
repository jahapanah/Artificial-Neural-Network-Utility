#pragma once

class NeuralNet;
#include <iterator>
#include <list>
#include "InputLayer.h"
#include "OutputLayer.h"
#include "HiddenLayer.h"
#include "enum.h"
#include "Matrix.h"
#include "IdentityMatrix.h"


class NeuralNet
{
public:
	NeuralNet();
	~NeuralNet();

	NeuralNet initNet(int numberOfInputNeurons,int numberOfNeuronsInHiddenLayer,int numberOfOutputNeurons);
	NeuralNet trainNet(NeuralNet& n);
	void netValidation(NeuralNet& n);

	void printNet( NeuralNet& n) ;
	void printTrainedNetResult( NeuralNet& n) ;
	std::vector<std::vector<double>> getNetOutputValues(NeuralNet traineNet);
	
private:
	InputLayer inputLayer;
	HiddenLayer hiddenLayer;
	OutputLayer outputLayer;

	std::vector<std::vector<double>> trainSet;
	std::vector<std::vector<double>> validationSet;
	std::vector<double> realOutputSet;
	std::vector<std::vector<double>> realMatrixOutputSet;

	int maxEpochs;
	double learningRate;
	double targetError;
	double trainingError;
	double errorMean;

	std::vector<double>listOfMSE;
	ActivationFncENUM activationFnc;
	ActivationFncENUM activationFncOutputLayer;
	


public:
	InputLayer & getInputLayer() { return inputLayer; }
	void setInputLayer(const InputLayer& inputLayer) { this->inputLayer = inputLayer; }

	 HiddenLayer& getHiddenLayer()  { return hiddenLayer; }
	void setHiddenLayer(const HiddenLayer& hiddenLayer) { this->hiddenLayer = hiddenLayer; }


	 OutputLayer& getOutputLayer()  { return outputLayer; }
	void setOutputLayer(const OutputLayer& ouputLayer) { this->outputLayer = ouputLayer; }


	 std::vector<std::vector<double>>& getTrainSet()  { return trainSet; }
	void setTrainSet(const std::vector<std::vector<double>>& trainSet) { this->trainSet = trainSet; }

	std::vector<std::vector<double>>& getValidationSet() { return validationSet; }
	void setValidationSet(const std::vector<std::vector<double>> validationSet) { this->validationSet = validationSet; }

	 std::vector<double> getRealOutputSet()  { return realOutputSet; }
	void setRealOutputSet(const std::vector<double>& realOutputSet) { this->realOutputSet = realOutputSet; }

	 std::vector<std::vector<double>>& getRealMatrixOutputSet() { return realMatrixOutputSet; }
	void setRealMatrixOutputSet(const std::vector<std::vector<double>>& realMatrixOutputSet) { this->realMatrixOutputSet = realMatrixOutputSet; }

	 int getMaxEpochs() { return maxEpochs; }
	void setMaxEpochs(const int maxEpochs) { this->maxEpochs = maxEpochs; }

	 double getTargetError()  { return targetError; }
	void setTargetError(const double targetError) { this->targetError = targetError; }

	 double getLearningRate()  { return learningRate; }
	void setLearningRate(const double learningRate) { this->learningRate = learningRate; }

	 double getTrainingError()  { return trainingError; }
	void setTrainingError(const double trainingError) { this->trainingError = trainingError; }

	 double getErrorMean() { return errorMean; }
	void setErrorMean(const double errorMean) { this->errorMean = errorMean; }

	 ActivationFncENUM getActivationFnc()  { return activationFnc; }
	void setActivationFnc(ActivationFncENUM activationFnc) { this->activationFnc = activationFnc; }

	 ActivationFncENUM getActivationFncOutputLayer() { return activationFncOutputLayer; }
	void setActivationFncOutputLayer(const ActivationFncENUM activationFncOutputLayer) { this->activationFncOutputLayer = activationFncOutputLayer; }

	std::vector<double>& getListOfMSE()  { return listOfMSE; }
	void setListOfMSE(const std::vector<double>& listOfMSE) { this->listOfMSE = listOfMSE; }
};

NeuralNet::NeuralNet()
{
}

NeuralNet::~NeuralNet()
{
}

NeuralNet NeuralNet::initNet(
	int numberOfInputNeurons, 
	int numberOfNeuronsInHiddenLayer, 
	int numberOfOutputNeurons)
{
	
	inputLayer.setNumberOfNeuronsInLayer(numberOfInputNeurons);
	hiddenLayer.setNumberOfNeuronsInLayer(numberOfNeuronsInHiddenLayer);
	outputLayer.setNumberOfNeuronsInLayer(numberOfOutputNeurons);

	inputLayer = inputLayer.initLayer(inputLayer);
	hiddenLayer = hiddenLayer.initLayer(hiddenLayer,inputLayer,outputLayer);
	outputLayer = outputLayer.initLayer(outputLayer);

	NeuralNet newNet;
	newNet.setInputLayer(inputLayer);
	newNet.setHiddenLayer(hiddenLayer);
	newNet.setOutputLayer(outputLayer);

	return newNet;
}
	
inline void NeuralNet::printNet( NeuralNet& n) {
	inputLayer.printLayer(n.getInputLayer());
	std::cout<<std::endl;
	hiddenLayer.printLayer(n.getHiddenLayer());
	std::cout<<std::endl;
	outputLayer.printLayer(outputLayer);
}
#include "Training.h"
#include "Backpropagation.h"


inline void NeuralNet::printTrainedNetResult (NeuralNet & n)
{
	
		Backpropagation b;
		b.printTrainedNetResult(n);
	
}

inline std::vector<std::vector<double>> NeuralNet::getNetOutputValues(NeuralNet trainedNet)
{
	size_t rows = trainedNet.getTrainSet().size();
	size_t cols = trainedNet.getOutputLayer().getNumberOfNeuronsInLayer();

	std::vector<std::vector<double>> matrixOutputValues(rows, std::vector<double>(cols));
	
		Backpropagation b;
		for (size_t rows_i = 0; rows_i < rows; rows_i++)
			for (size_t cols_i = 0; cols_i < cols; cols_i++) 
				matrixOutputValues[rows_i][cols_i] = b.forward(trainedNet, rows_i).getOutputLayer().getListOfNeurons()[cols_i].getOutputValue();
	
		
	return matrixOutputValues;
}

NeuralNet NeuralNet::trainNet(NeuralNet & n)
{
	NeuralNet trainedNet;
	
	
		Backpropagation b;
		trainedNet = b.train(n);
		return trainedNet;
	
	
}


