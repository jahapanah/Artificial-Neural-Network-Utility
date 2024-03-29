#pragma once
#include<iostream>
using namespace std;

class Backpropagation:public Training
{
public:
	Backpropagation();
	~Backpropagation();

	NeuralNet& train(NeuralNet& n);

	NeuralNet& forward(NeuralNet& n, size_t row);
	NeuralNet& backpropagation(NeuralNet& n, size_t row);
private:
};

Backpropagation::Backpropagation(): Training()
{
}

Backpropagation::~Backpropagation()
{
}

inline NeuralNet & Backpropagation::train(NeuralNet & n)
{
	int epoch = 0;
	setMse(1.0);
	n.getListOfMSE().push_back(getMse());
	while (getMse() > n.getTargetError()) {
		if (epoch >= n.getMaxEpochs())break;
		size_t rows = n.getTrainSet().size();
		double sumErrors = 0.0;

		for (size_t rows_i = 0; rows_i < rows; rows_i++) {
			n = forward(n, rows_i);
			n = backpropagation(n, rows_i);
			sumErrors = sumErrors + n.getErrorMean();
		}
		setMse(sumErrors / rows);
		n.getListOfMSE().push_back(getMse());
		epoch++;
	}
	std::cout << getMse()<<std::endl;
	std::cout << "Number of epochs: " << epoch << std::endl;
	return n;
}

inline NeuralNet & Backpropagation::forward(NeuralNet & n, size_t row)
{

	double estimatedOutput = 0.0;
	double realOutput = 0.0;
	double sumError = 0.0;

	
		int hiddenLayer_i = 0;

			size_t numberOfNeuronsInLayer = n.getHiddenLayer().getNumberOfNeuronsInLayer();
			for(std::vector<Neuron>::iterator neuron= n.getHiddenLayer().getListOfNeurons().begin();neuron!= n.getHiddenLayer().getListOfNeurons().end();neuron++){
				double netValueOut = 0.0;
				if (neuron->getListOfWeightIn().size() > 0) {
					double netValue = 0.0;
					for (size_t layer_j = 0; layer_j < neuron->getListOfWeightIn().size(); layer_j++) {
						double hiddenWeightIn = neuron->getListOfWeightIn()[layer_j];
						netValue = netValue + hiddenWeightIn * n.getTrainSet()[row][layer_j];
					}
					// Output hidden layer (1)
					netValueOut = (this->*activationFnc[n.getActivationFnc()])(netValue);
					neuron->setOutputValue(netValueOut);
				}
				else {
					neuron->setOutputValue(1.0);
				}
			}
			n.setHiddenLayer(n.getHiddenLayer());//////// Important to know whether hiddenLayer is a copy of the last element of listOfHiddenLayer
			// Output hidden layer (2)
			for (size_t outLayer_i = 0; outLayer_i < n.getOutputLayer().getNumberOfNeuronsInLayer(); outLayer_i++) {
				double netValue = 0.0;
				double netValueOut = 0.0;
				for (Neuron neuron : n.getHiddenLayer().getListOfNeurons()) {
					double hiddenWeightOut = neuron.getListOfWeightOut()[outLayer_i];
					netValue = netValue + hiddenWeightOut * neuron.getOutputValue();
				}
				netValueOut = (this->*activationFnc[n.getActivationFncOutputLayer()])(netValue);
				n.getOutputLayer().getListOfNeurons()[outLayer_i].setOutputValue(netValueOut);

				// Error
				estimatedOutput = netValueOut;
				realOutput = n.getRealMatrixOutputSet()[row][outLayer_i];
				double error = realOutput - estimatedOutput;
				n.getOutputLayer().getListOfNeurons()[outLayer_i].setError(error);
				sumError += pow(error, 2);
			}
			// Error mean
			double errorMean = sumError / n.getOutputLayer().getNumberOfNeuronsInLayer();
			n.setErrorMean(errorMean);
			n.getHiddenLayer().setListOfNeurons(n.getHiddenLayer().getListOfNeurons());
			hiddenLayer_i++;
		
	
	return n;
}

inline NeuralNet & Backpropagation::backpropagation(NeuralNet & n, size_t row)
{
	std::vector<Neuron>& outputLayer = n.getOutputLayer().getListOfNeurons();
	std::vector<Neuron>& hiddenLayer = n.getHiddenLayer().getListOfNeurons();

	double error = 0.0;
	double netValue = 0.0;
	double delta = 0.0;
	// delta output layer
	for(std::vector<Neuron>::iterator it=outputLayer.begin();it!=outputLayer.end();it++){
		error = it->getError();
		netValue = it->getOutputValue();
		delta = error*(this->*derivateActivationFnc[n.getActivationFncOutputLayer()])(netValue);
		it->setdelta(delta);
	}
	n.getOutputLayer().setListOfNeurons(outputLayer);
	// delta hidden layer
	for (std::vector<Neuron>::iterator it = hiddenLayer.begin(); it != hiddenLayer.end(); it++) {
		delta = 0.0;
		if (it->getListOfWeightIn().size() > 0) {
			std::vector<double>listOfWeightOut = it->getListOfWeightOut();
			double tempdelta = 0.0;
			for (size_t i = 0; i < listOfWeightOut.size(); i++)
				tempdelta += listOfWeightOut[i] * outputLayer[i].getdelta();
			delta = (this->*derivateActivationFnc[n.getActivationFnc()])(it->getOutputValue())*tempdelta;
			it->setdelta(delta);
		}
	}
	n.getHiddenLayer().setListOfNeurons(hiddenLayer);///////////////
	 
	// Fix weights (teach) [output layer to hidden layer]
	for (size_t outLayer_i = 0; outLayer_i < n.getOutputLayer().getNumberOfNeuronsInLayer(); outLayer_i++) {
		for (std::vector<Neuron>::iterator it = hiddenLayer.begin(); it != hiddenLayer.end(); it++) {
			double newWeight = it->getListOfWeightOut()[outLayer_i] + (n.getLearningRate()*outputLayer[outLayer_i].getdelta()*it->getOutputValue());
			it->getListOfWeightOut()[outLayer_i] = newWeight;
		}
	}
	n.getOutputLayer().setListOfNeurons(outputLayer);////////////////
	// Fix weights (teach) [output layer to input layer]
	for (std::vector<Neuron>::iterator it = hiddenLayer.begin(); it != hiddenLayer.end(); it++) {
		std::vector<double> hiddenLayerInputWeights = it->getListOfWeightIn();
		if (hiddenLayerInputWeights.size() > 0) {
			double newWeight = 0.0;
			for (size_t i = 0; i < n.getInputLayer().getNumberOfNeuronsInLayer(); i++) {
				newWeight = hiddenLayerInputWeights[i] + (n.getLearningRate()*it->getdelta()*n.getTrainSet()[row][i]);
				it->getListOfWeightIn()[i] = newWeight;
			}
		}
	}
	n.getHiddenLayer().setListOfNeurons(hiddenLayer);
	return n;
}
