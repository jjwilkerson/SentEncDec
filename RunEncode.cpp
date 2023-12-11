/**
 * @file
 * @brief Builds and runs a network that encodes sentences as vectors.
 *
 * Builds the encoder part of the sentence encoder/decoder network, to use for encoding sentences as vectors. Uses the
 * residual-recurrent architecture. Uses pre-trained parameters.
 */

#if PROGRAM_VERSION == 4

#include <nonlinearity/Nonlinearity.h>
#include <input/WordVectors.h>
#include <input/WvCorpusIterator.h>
#include <input/SedInputSource.h>
#include <layers/SeqInputLayer.h>
#include <layers/RecDTR2nLayer.h>
#include <layers/ActivationLayer2.h>
#include <layers/DupInLayer.h>
#include <layers/ConcatLayer.h>
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
	config.weightsFilename = "enc_weights.bin";

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

	cout << endl << "Loading training corpus" << endl;
	WvCorpusIterator trainCorpus(config, wv, inputFilename, batchSize, maxSeqLength, false, false);

	int encSize = config.encoder_rnn_width;
	int svSize = config.sentence_vector_width;
	int encMod = config.encoder_mod;

	Linear linear(batchSize);
	Tanh tanh(batchSize);

	bool forward = false;
	SedInputSource trainSource(trainCorpus, wv, batchSize, maxSeqLength, 1, forward, 0, config.dictGpu1);

	SeqInputLayer revInputLayer("rev_input", handle, batchSize, wvLength, maxSeqLength, &linear, 0.0f, false);

	RecDTR2nLayer revRecLayer("enc_rec", handle, batchSize, encSize, maxSeqLength, 0.0f, &tanh, &tanh, 0.0f, 0.0f, 0.0f, encMod);
	DupInLayer revDupInLayer("dup_in", handle, batchSize, encSize, &linear, 0.5f);
	ActivationLayer2 encActLayer("enc_act", handle, batchSize, encSize, 1, &tanh);
	DenseLayer2 svLayer("sv", handle, batchSize, svSize, 1, &tanh, 0.0f);

	svLayer.asOutputLayer();

	revRecLayer.setPrev(&revInputLayer);
	revDupInLayer.setPrev(&revRecLayer);
	encActLayer.setPrev(&revDupInLayer);
	svLayer.setPrev(&encActLayer);

	LossFunction* loss = new SquaredError(batchSize, maxSeqLength, handle);
	SparseWeightInit ffWeightInit(config.seed, config.ffWeightCoef, 0);
	SparseWeightInit recWeightInit(config.seed, config.recWeightCoef, 0);

	Network net(handle, batchSize, maxSeqLength, config, *loss, ffWeightInit, recWeightInit, false);
	net.setTrainSource(&trainSource);

	net.addInput(&revInputLayer);
	net.addHidden(&revRecLayer);
	net.addHidden(&revDupInLayer);
	net.addHidden(&encActLayer);
	net.addHidden(&svLayer);
	net.setOutput(&svLayer);

	net.init();

	std::ofstream outputFile(outputFilename, ios::binary);

	int arraySize = batchSize * svSize * sizeof(dtype2);
	dtype2* hAct;
	checkCudaErrors(cudaMallocHost((void **)&hAct, arraySize));

	net.iterInit();

	int b = 0;
	while (trainSource.hasNext() && b < numBatch) {
		trainSource.toFirstBatch();
		net.forwardNext();
		dtype2* svAct = svLayer.activation[0];

//		for (int b = 0; b < batchSize; b++) {
//			Network::printStatsGpu(svAct+)
//		}

		checkCudaErrors(cudaMemcpy(hAct, svAct, arraySize, cudaMemcpyDeviceToHost));

		if (b % 1000 == 0) {
			cout << b << endl;
		}

#ifdef DEVICE_HALF
		for (int b = 0; b < batchSize; b++) {
			for (int i = 0; i < svSize-1; i+=2) {
				int val;
				dtype2* d2p = (dtype2*) &val;
				d2p[0] = hAct[IDX2(b, i, batchSize)];
				d2p[1] = hAct[IDX2(b, i+1, batchSize)];
				FileUtil::writeInt(val, outputFile);
			}
		}
#else
		for (int b = 0; b < batchSize; b++) {
			for (int i = 0; i < svSize; i++) {
				dtype2 val = hAct[IDX2(b, i, batchSize)];
				dtypeh singleVal = d2float(val);
				FileUtil::writeFloat(singleVal, outputFile);
			}
		}
#endif

		b++;
		trainSource.toNextBatchSet();
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
