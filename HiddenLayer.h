#pragma once
#include "Layer.h"
#include "InputLayer.h"
#include "OutputLayer.h"


class HiddenLayer : public Layer
{
public:
	HiddenLayer();
	~HiddenLayer();

	HiddenLayer& initLayer( HiddenLayer&,const InputLayer& ,const OutputLayer& );
	void printLayer(HiddenLayer&);
	void setNumberOfNeuronsInLayer(int numberOfNeuronInLayer) { this->numberOfNeuronInLayer = numberOfNeuronInLayer+1; }

private:

};

HiddenLayer::HiddenLayer() : Layer()
{
}

HiddenLayer::~HiddenLayer()
{
}

inline HiddenLayer& HiddenLayer::initLayer(
	HiddenLayer&					    hiddenLayer, 
	const InputLayer &					inputLayer,
	const OutputLayer &					outputLayer)
{
	std::vector<double> listOfWeightIn;
	std::vector<double> listOfWeightOut;
	std::vector<Neuron> listOfNeurons;


		for (size_t j = 0; j <hiddenLayer.getNumberOfNeuronsInLayer(); j++) {

			Neuron neuron;
			size_t limitIn;
			size_t limitOut;

			
				limitIn = inputLayer.getNumberOfNeuronsInLayer()-1;
				limitOut = outputLayer.getNumberOfNeuronsInLayer()-1;
		

			if (j >= 1) {
				for (size_t k = 0; k <= limitIn; k++) {
					listOfWeightIn.push_back(neuron.initNeuron());
				}
			}
			for (size_t k = 0; k <= limitOut; k++) {
				listOfWeightOut.push_back(neuron.initNeuron());
			}

			neuron.setListOfWeightIn(listOfWeightIn);
			neuron.setListOfWeightOut(listOfWeightOut);
			listOfNeurons.push_back(neuron);

			listOfWeightIn.clear();
			listOfWeightOut.clear();
		}
		hiddenLayer.setListOfNeurons(listOfNeurons);
		this->setListOfNeurons(listOfNeurons);
		listOfNeurons.clear();
	
	return hiddenLayer;
}

inline void HiddenLayer::printLayer(HiddenLayer& hiddenLayer) 
{std::cout << "### HIDDEN LAYER ###" << std::endl;
	int n = 1;
	std::vector<Neuron> neuron = hiddenLayer.getListOfNeurons();
		for(int i = 1; i < neuron.size(); i++)
       {
		std::cout << "Neuron #" << n << ":" << std::endl;
		std::cout << "Input Weights:" << std::endl;
		std::vector<double> weights = neuron[i].getListOfWeightIn();

		for (double weight : weights) {
			std::cout << weight << std::endl;
		}
    n++;
	}
}
