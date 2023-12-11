/**
 * @file
 * @brief Builds and runs a network that decodes sentence vectors into sentences.
 *
 * Builds the decoder part of the sentence encoder/decoder network, to use for decoding sentence vectors into sentences. Uses the
 * residual-recurrent architecture. Uses pre-trained parameters.
 */

#if PROGRAM_VERSION == 5

#include <nonlinearity/Nonlinearity.h>
#include <input/WordVectors.h>
#include <input/VecIterator.h>
#include <input/VecInputSource.h>
#include <layers/VecInputLayer.h>
#include <layers/RecDTR2nLayer.h>
#include <layers/ActivationLayer2.h>
#include <layers/DenseLayer2.h>
#include <layers/DupOutLayer.h>
#include <loss/LossFunction.h>
#include <config/WeightInit.h>
#include <config/Config.h>
#include <state/State.h>
#include <Network.h>
#include <optimizers/Adam.h>
#include <util/FileUtil.h>
#include "Trainer.h"
#include "cublas_v2.h"
#include <iostream>
#include <helper_cuda.h>
#include <boost/filesystem.hpp>

using namespace std;
using namespace boost::filesystem;
using namespace netlib;

Config config;
const char* configFilename = "config.json";

int main(int argc, char* argv[]) {
	if (argc < 4) {
		cerr << "format: input_file output_file num_batch" << endl;
		exit(-1);
	}

	const char* inputFilename = argv[1];
	const char* outputFilename = argv[2];
	int numBatch = atoi(argv[3]);

	cout << "numBatch: " << numBatch << endl;

	path configFile(configFilename);

	if (exists(configFile)) {
		config.load(configFilename);
	} else {
		config.batchSize = 32;
		config.encoder_rnn_width = 2;
		config.decoder_rnn_width = 2;
		config.sentence_vector_width = 2;
		config.numBatchError = 1;
		config.numBatchGrad = 1; //70;
		config.numBatchG = 1; //4;
		config.pDropMatch = 0.0;
		config.seed = 1234;
		config.ffWeightCoef = 0.14150421321392059;
		config.recWeightCoef = 0.030431009829044342;
		config.l2 = 7e-04;
		config.initDamp = 11.449784278869629;
		config.structDampCoef = 1.8929900988950976e-06;
		config.optimizer = "adam";
	}


	int batchSize = config.batchSize;
	int maxSeqLength = 60;
	int wvLength = 300;
	config.maxSeqLength = maxSeqLength;
	config.wvLength = wvLength;
	config.weightsFilename = "dec_weights.bin";

	cublasHandle_t handle, handle1;
	checkCudaErrors(cublasCreate(&handle));

	cublasHandle_t* wvHandle;
	if (config.dictGpu1) {
		wvHandle = &handle1;
	} else {
		wvHandle = &handle;
	}

	WordVectors wv(*wvHandle, config.dictGpu1, batchSize);
	wv.initDictGpu(*wvHandle);

	int svSize = config.sentence_vector_width;
	int decSize = config.decoder_rnn_width;
	int decMod = config.decoder_mod;

	Linear linear(batchSize);
	Tanh tanh(batchSize);

	VecIterator corpus(inputFilename, svSize, batchSize, false , "", 0);
	VecInputSource inputSource(corpus, batchSize, svSize, maxSeqLength, 1, 0);

	VecInputLayer inputLayer("input", handle, batchSize, svSize, &linear, 0.0f);

	DupOutLayer dupOutLayer("dup_out", handle, batchSize, svSize, &linear, 0.0f);
	RecDTR2nLayer decRecLayer("dec_rec", handle, batchSize, decSize, maxSeqLength, 0.5f, &tanh, &tanh, 0.0f, 0.0f, 0.0f, decMod);
	ActivationLayer2 decActLayer("dec_act", handle, batchSize, decSize, maxSeqLength, &tanh);
	DenseLayer2 outputLayer("output", handle, batchSize, wvLength, maxSeqLength, &linear, 0.0f);
	outputLayer.asOutputLayer();

	dupOutLayer.setPrev(&inputLayer);
	decRecLayer.setPrev(&dupOutLayer);
	decActLayer.setPrev(&decRecLayer);
	outputLayer.setPrev(&decActLayer);

	LossFunction* loss = new SquaredError(batchSize, maxSeqLength, handle);
	SparseWeightInit ffWeightInit(config.seed, config.ffWeightCoef, 0);
	SparseWeightInit recWeightInit(config.seed, config.recWeightCoef, 0);

	Network net(handle, batchSize, maxSeqLength, config, *loss, ffWeightInit, recWeightInit, false);
	net.setTrainSource(&inputSource);

	net.addInput(&inputLayer);
	net.addHidden(&dupOutLayer);
	net.addHidden(&decRecLayer);
	net.addHidden(&decActLayer);
	net.setOutput(&outputLayer);

	net.setParamOffset(303040000);
	net.init();

	std::ofstream outputFile(outputFilename, ios::trunc);

	int arraySize = batchSize * svSize * sizeof(dtype2);
	dtype2* hAct;
	checkCudaErrors(cudaMallocHost((void **)&hAct, arraySize));

	dtype2 *vecOut_d;
	checkCudaErrors(cudaMalloc((void **)&vecOut_d, wv.wvLength * sizeof(dtype2)));

	net.iterInit();

	vector<string> batchesMatches[batchSize];

	int b = 0;
	while (b < numBatch) {
		inputSource.toFirstBatch();
		net.forwardNext();
		dtype2** outputs = outputLayer.activation;

		for (int s = 0; s < net.maxSeqLength; s++) {
			dtype2* output = outputs[s];
			string* matches = wv.nearestBatch(output, batchSize);

			for (int i = 0; i < batchSize; i++) {
				string match = matches[i];
				batchesMatches[i].push_back(match);
			}
		}

		for (int i = 0; i < batchSize; i++) {
			vector<string>* matches = &batchesMatches[i];
			string sent("");

			bool first = true;
			vector<string>::iterator it;
			for (it = matches->begin(); it != matches->end(); it++) {
				if (!first) {
					sent = sent + " ";
				}
				sent = sent + *it;

				if (*it == "EOS") {
					break;
				}
				first = false;
			}
			sent = sent + "\n";
			FileUtil::writeString(sent, outputFile);
			matches->clear();
		}

		if (b % 1000 == 0) {
			cout << b << endl;
		}

		b++;
		inputSource.toNextBatchSet();
	}

	outputFile.close();

	cublasDestroy(handle);
	if (config.dictGpu1) {
		cublasDestroy(handle1);
	}

	net.clearLayers(); //so won't delete, layers not dynamically allocated
	net.setTrainSource(NULL);
}

#endif
