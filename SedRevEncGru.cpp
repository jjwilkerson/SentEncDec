/**
 * @file
 * @brief Executes a single training run with the GRU architecture.
 *
 * Builds a GRU network, trainer, and optimizer and executes a single training run. Takes configuration, state, and
 * other objects as input. Called from RunTrainSingle.cpp, RunTrainRandomSearch.cpp, or RunTrainExtend.cpp.
 */

#if NETWORK_VERSION == 2

#include <nonlinearity/Nonlinearity.h>
#include <input/WordVectors.h>
#include <input/WvCorpusIterator.h>
#include <input/SedInputSource.h>
#include <layers/SeqInputLayer.h>
#include <layers/GruLayer.h>
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
#include "Trainer.h"
#include "cublas_v2.h"
#include <iostream>
#include <helper_cuda.h>
#include <boost/filesystem.hpp>

using namespace netlib;

/**
 * @brief Executes a single training run with the GRU architecture.
 *
 * Builds a GRU network, trainer, and optimizer and executes a single training run. Takes configuration, state, and
 * other objects as input. Called from RunTrainSingle.cpp, RunTrainRandomSearch.cpp, or RunTrainExtend.cpp.
 *
 * @param dirName
 * @param config
 * @param state
 * @param wv
 * @param trainCorpus
 * @param testCorpus
 * @param testMatchingCorpus
 * @param handle
 * @param maxIter
 * @param maxTimeS
 * @param scoreFinal
 * @param matchesFinal
 * @return final test score
 */
dtype2 run(string dirName, Config& config, State* state, WordVectors& wv, WvCorpusIterator& trainCorpus, WvCorpusIterator& testCorpus,
		WvCorpusIterator& testMatchingCorpus, cublasHandle_t& handle, int maxIter = -1, int maxTimeS = 0, dtype2* scoreFinal = NULL,
		int* matchesFinal = NULL) {
	int batchSize = config.batchSize;
	int maxSeqLength = config.maxSeqLength;
	int wvLength = config.wvLength;
	int encSize = config.encoder_rnn_width;
	int svSize = config.sentence_vector_width;
	int decSize = config.decoder_rnn_width;
	int encMod = config.encoder_mod;
	int decMod = config.decoder_mod;

	int startEpoch = state->epoch;
	int startIter = state->iter + 1;
	ix_type startIx = state->sentenceIx;
	unsigned int clockOffset = state->clock;
	string initDeltaFilename = state->initDeltaFilename;

	int maxNumBatch = max(config.numBatchError, max(config.numBatchGrad, config.numBatchG));

	Linear linear(batchSize);
	Tanh tanh(batchSize);

	bool forward = false;
	SedInputSource trainSource(trainCorpus, wv, batchSize, maxSeqLength, maxNumBatch, forward, startIx, config.dictGpu1);
	SedInputSource testSource(testCorpus, wv, batchSize, maxSeqLength, 1, forward, 0, config.dictGpu1);
	SedInputSource testMatchingSource(testMatchingCorpus, wv, batchSize, maxSeqLength, 1, forward, 0, config.dictGpu1);

	SeqInputLayer revInputLayer("rev_input", handle, batchSize, wvLength, maxSeqLength, &linear, 0.0f, false);

	GruLayer revRecLayer("enc_rec_gru", handle, batchSize, encSize, maxSeqLength, &linear, 0.0f);
	DupInLayer revDupInLayer("dup_in", handle, batchSize, encSize, &linear, 0.5f);
	ActivationLayer2 encActLayer("enc_act", handle, batchSize, encSize, 1, &tanh);
	DenseLayer2 svLayer("sv", handle, batchSize, svSize, 1, &tanh, 0.0f);
	DupOutLayer dupOutLayer("dup_out", handle, batchSize, svSize, &linear, 0.0f);
	GruLayer decRecLayer("dec_rec_gru", handle, batchSize, decSize, maxSeqLength, &linear, 0.5f);
	ActivationLayer2 decActLayer("dec_act", handle, batchSize, decSize, maxSeqLength, &tanh);
	DenseLayer2 outputLayer("output", handle, batchSize, wvLength, maxSeqLength, &linear, 0.0f);
	outputLayer.asOutputLayer();

	revRecLayer.setPrev(&revInputLayer);
	revDupInLayer.setPrev(&revRecLayer);
	encActLayer.setPrev(&revDupInLayer);
	svLayer.setPrev(&encActLayer);
	dupOutLayer.setPrev(&svLayer);
	decRecLayer.setPrev(&dupOutLayer);
	decActLayer.setPrev(&decRecLayer);
	outputLayer.setPrev(&decActLayer);

//	SquaredError squaredError(batchSize, maxSeqLength, handle);
//	StructuralDamping structuralDamping(batchSize, maxSeqLength, config.structDampCoef);
//	structuralDamping.damping = config.initDamp;
//	LossFunction* lossFuncs[2] = {&squaredError, &structuralDamping};
//	LossSet loss(batchSize, maxSeqLength, 2, lossFuncs);

//	SquaredError loss(batchSize, maxSeqLength, handle);

	LossFunction* loss = NULL;
	StructuralDamping* structuralDamping = NULL;
	if (config.optimizer == Config::HF) {
		SquaredError squaredError(batchSize, maxSeqLength, handle);
		structuralDamping = new StructuralDamping(batchSize, maxSeqLength, config.structDampCoef);
		structuralDamping->damping = config.initDamp;
		LossFunction* lossFuncs[2] = {&squaredError, structuralDamping};
		loss = new LossSet(batchSize, maxSeqLength, 2, lossFuncs);
	} else {
		loss = new SquaredError(batchSize, maxSeqLength, handle);
//		loss = new CosineSim(batchSize, maxSeqLength, handle);
	}

	SparseWeightInit ffWeightInit(config.seed, config.ffWeightCoef, 0);
	SparseWeightInit recWeightInit(config.seed, config.recWeightCoef, 0);

	Network net(handle, batchSize, maxSeqLength, config, *loss, ffWeightInit, recWeightInit, true);
	net.setTrainSource(&trainSource);

	net.addInput(&revInputLayer);
	net.addHidden(&revRecLayer);
	net.addHidden(&revDupInLayer);
	net.addHidden(&encActLayer);
	net.addHidden(&svLayer);
	net.addHidden(&dupOutLayer);
	net.addHidden(&decRecLayer);
	net.addHidden(&decActLayer);
	net.setOutput(&outputLayer);

	net.init();

	cout << endl << "sentence_ix: " << startIx << endl;

//	net.iterInit();
//	dtype2 error = net.error();
//	cout << "error: " << error << endl;
////	exit(0);
//	dtype2 *grad = net.calcGrad();
//	dtype2 norm;
//	CublasFunc::nrm2(handle, net.nParams, grad, 1, &norm);
//	cout << "grad norm: " << norm << endl;
//
//	Network::printStatsGpu("grad", grad, net.nParams);
////	exit(0);
////#ifdef DEBUG
//	net.checkGrad(grad);
////#endif
////
////	dtype2* Gv = net.calcG(grad, config.initDamp, NULL);
////	CublasFunc::nrm2(handle, net.nParams, Gv, 1, &norm);
////	cout << "Gv norm: " << norm << endl;
////
////	Network::printStatsGpu("Gv", Gv, net.nParams);
////
////#ifdef DEBUG
////	net.checkG(Gv, grad, config.initDamp);
////#endif
//    checkCudaErrors(cudaFree(grad));
////    checkCudaErrors(cudaFree(Gv));
//    exit(0);

	Optimizer* optimizer = NULL;
//	if (config.optimizer == Config::ADAM) {
		optimizer = new Adam(handle, net, config, startIter, false, initDeltaFilename);
//	} else {
//		optimizer = new HessianFree(handle, net, config, structuralDamping, false, initDeltaFilename);
//	}

	Trainer trainer(handle, config, net, *optimizer, wv, trainSource, testSource, testMatchingSource);

	dtype2 score = trainer.train(dirName, startEpoch, startIter, maxIter, maxTimeS, clockOffset, config.testPeriod,
			config.testMatchingPeriod, config.demoPeriod, config.savePeriod, config.tempSavePeriod,
			config.printPeriod, 100, false, scoreFinal, matchesFinal);

    delete optimizer;

    if (structuralDamping != NULL) {
    	delete structuralDamping;
    }
   	delete loss;

	return score;
}

#endif
