/**
 * @file
 * @brief Entry point for single training run.
 *
 * Used to build an executable for a single training run, using a local config.json for configuration and loading
 * state from a local state.json file, if present. Enabled by building with -DPROGRAM_VERSION=1
 */

#if PROGRAM_VERSION == 1

#include <nonlinearity/Nonlinearity.h>
#include <input/WordVectors.h>
#include <input/WvCorpusIterator.h>
#include <input/SedInputSource.h>
#include <layers/SeqInputLayer.h>
#include <layers/RecDTR2nLayer.h>
#include <layers/ActivationLayer.h>
#include <layers/DupInLayer.h>
#include <layers/ConcatLayer.h>
#include <layers/DenseLayer.h>
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
#include <cstdlib>
#include <helper_cuda.h>
#include <boost/filesystem.hpp>

using namespace std;
using namespace boost::filesystem;
using namespace netlib;

Config config;
const char* configFilename = "config.json";
const char* stateFilename = "state.json";

extern dtype2 run(string outputDirName, Config& config, State* state, WordVectors& wv, WvCorpusIterator& trainCorpus, WvCorpusIterator& testCorpus,
		WvCorpusIterator& testMatchingCorpus, cublasHandle_t& handle, int maxIter = -1, int maxTimeS = 0, dtype2* scoreFinal = NULL,
		int* matchesFinal = NULL);

int main(int argc, char* argv[]) {
    if (getenv("SED_DATASETS_DIR") == nullptr) {
    	cerr << "Environment variable SED_DATASETS_DIR must be set." << endl;
    	exit(1);
    }

	path curr(argc > 1 ? argv[1] : ".");
	current_path(curr);
	string sDir = current_path().string();
	const char* dataDirName = sDir.c_str();

	cout << "data dir: " << dataDirName << endl;

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

	State* state = NULL;

	path stateFile(stateFilename);
	if (exists(stateFile)) {
		state = new State();
		state->load(stateFilename);
	}

	if (state == NULL) {
		state = new State();
		state->ixFilename = datasetsDir + "/sentence_ixs_epoch0";
		state->curandStatesFilename = "";
		state->replacerStateFilename = "";
		state->initDeltaFilename = "";
		state->epoch = 0;
		state->iter = -1;
		state->sentenceIx = 0;
		state->clock = 0;
	} else {
		config.weightsFilename = state->weightsFilename;
		config.curandStatesFilename = state->curandStatesFilename;
		config.replacerStateFilename = state->replacerStateFilename;

		config.initDamp = state->damping;
		config.deltaDecayInitial = state->deltaDecay;

		if (state->l2 >= 0) {
			config.l2 = state->l2;
		}

		config.numBatchGrad = state->numBatchGrad;
		config.numBatchG = state->numBatchG;
		config.numBatchError = state->numBatchError;

		if (state->maxIterCG > 0) {
			config.maxIterCG = state->maxIterCG;
		}

		if (state->learningRate > 0) {
			config.learningRate = state->learningRate;
		}

		config.lossScaleFac = state->lossScaleFac;
		config.iterNoOverflow = state->iterNoOverflow;
	}

	string ixFilename = state->ixFilename;
	ix_type startIx = state->sentenceIx;

	int batchSize = config.batchSize;
	int maxSeqLength = 60;
	int wvLength = 300;
	config.maxSeqLength = maxSeqLength;
	config.wvLength = wvLength;

	cublasHandle_t handle, handle1;
	checkCudaErrors(cublasCreate(&handle));

	if (config.dictGpu1) {
		int canAccess;
		checkCudaErrors(cudaDeviceCanAccessPeer(&canAccess, 0, 1));
		cout << "can access 0-1: " << canAccess << endl;
		checkCudaErrors(cudaDeviceCanAccessPeer(&canAccess, 1, 0));
		cout << "can access 1-0: " << canAccess << endl;
		checkCudaErrors(cudaSetDevice(1));
		checkCudaErrors(cublasCreate(&handle1));
		checkCudaErrors(cudaSetDevice(0));
	}

	cublasHandle_t* wvHandle;
	if (config.dictGpu1) {
		wvHandle = &handle1;
	} else {
		wvHandle = &handle;
	}

	WordVectors wv(*wvHandle, config.dictGpu1, batchSize);
	wv.initDictGpu(*wvHandle);

	cout << endl << "Loading training corpus" << endl;
	string trainFilename = datasetsDir + "/news.2007-2017.en.proc3_sym9_max59_train";
	WvCorpusIterator trainCorpus(config, wv, trainFilename, batchSize, maxSeqLength, true, true, 200, ixFilename, startIx);

	cout << endl << "Loading test corpus" << endl;
	string testFilename = datasetsDir + "/news.2007-2017.en.proc3_sym9_max59_tune";
	WvCorpusIterator testCorpus(config, wv, testFilename, batchSize, maxSeqLength, false, false);
	WvCorpusIterator testMatchingCorpus(config, wv, testFilename, batchSize, maxSeqLength, false, false);

	int maxIter = -1;
	int maxTimeS = 0;

	run(sDir, config, state, wv, trainCorpus, testCorpus, testMatchingCorpus, handle, maxIter, maxTimeS);

    if (state != NULL) {
    	delete state;
    }

	cublasDestroy(handle);
	if (config.dictGpu1) {
		cublasDestroy(handle1);
	}
}

#endif
