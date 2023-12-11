/**
 * @file
 * @brief Entry point for a training run that extends a previous run.
 *
 * Used to build an executable for a training run that extends a previous run, to a max number of iterations or max
 * training time. Typically used to extend selected runs produced by random search. Enabled by building with
 * -DPROGRAM_VERSION=3
 */

#if PROGRAM_VERSION == 3

#include <input/WordVectors.h>
#include <input/WvCorpusIterator.h>
#include <config/Config.h>
#include <state/State.h>
#include <string>
#include <list>
#include <boost/filesystem.hpp>

using namespace std;
using namespace boost::filesystem;
using namespace netlib;

const char* resultsInFilename = "results_3h_sorted";
const char* resultsOutFilename = "results_3h-24h";
const char* configFilename = "config.json";
const char* stateFilename = "encdeccu_state_save.bak.json";
string weightsFilename = "encdeccu_weights_save.bak.bin";
string initDeltaFilename = "encdeccu_initDelta_save.bak.bin";
string curandStatesFilename = "encdeccu_curandStates_save.bak.bin";

int maxIter = -1;
int maxTimeS = 21*60*60;

extern dtype2 run(string outputDirName, Config& config, State* state, WordVectors& wv, WvCorpusIterator& trainCorpus,
		WvCorpusIterator& testCorpus, WvCorpusIterator& testMatchingCorpus, cublasHandle_t& handle, int maxIter = -1,
		int maxTimeS = 0, dtype2* scoreFinal = NULL, int* matchesFinal = NULL);

list<string> loadModelNames(const char* modelsFilename);

int main(int argc, char* argv[]) {
    if (getenv("SED_DATASETS_DIR") == nullptr) {
    	cerr << "Environment variable SED_DATASETS_DIR must be set." << endl;
    	exit(1);
    }

    path curr(argc > 1 ? argv[1] : ".");
	current_path(curr);
	string sDir = current_path().string();
	const char* baseDirName = sDir.c_str();

	cout << "base dir: " << baseDirName << endl;

	path baseDir(baseDirName);
	if (!exists(baseDir)) {
		create_directories(baseDir);
	}
	current_path(baseDir);

	int batchSize = 32;
	int maxSeqLength = 80;
	int wvLength = 300;

	cublasHandle_t handle;
	checkCudaErrors(cublasCreate(&handle));

	string ixFilename = datasetsDir + "/sentence_ixs_epoch0";
	WordVectors wv(handle, false, batchSize);
	wv.initDictGpu(handle);

	Config config;
	config.seed = 1234;
	config.replacerStateFilename = "";

	cout << endl << "Loading training corpus" << endl;
	string trainFilename = datasetsDir + "/news.2007-2017.en.proc3_sym9_max79_train";
	WvCorpusIterator trainCorpus(config, wv, trainFilename, batchSize, maxSeqLength, true, false, 0, ixFilename, 0);

	cout << endl << "Loading test corpus" << endl;
	string testFilename = datasetsDir + "/news.2007-2017.en.proc3_sym9_max79_tune";
	WvCorpusIterator testCorpus(config, wv, testFilename, batchSize, maxSeqLength, false, false);
	WvCorpusIterator testMatchingCorpus(config, wv, testFilename, batchSize, maxSeqLength, false, false);

	list<string> modelNames = loadModelNames(resultsInFilename);

	for (list<string>::iterator it=modelNames.begin(); it != modelNames.end(); ++it) {
		path netDir = absolute(*it, baseDir);
		if (!exists(netDir)) {
			cout << "dir " << *it << " not found" << endl;
			exit(1);
		}
		current_path(netDir);
		cout << endl << "net dir " << *it << endl;

		Config config;
		config.load(configFilename);

		State state;
		state.load(stateFilename);
		state.weightsFilename = weightsFilename;
		state.initDeltaFilename = initDeltaFilename;
		state.curandStatesFilename = curandStatesFilename;
		config.updateFromState(&state);

		dtype2 score;
		int matches;
		run(netDir.c_str(), config, &state, wv, trainCorpus, testCorpus, testMatchingCorpus, handle, maxIter, maxTimeS, &score,
				&matches);

		cout << endl << "score : " << score << endl;
		cout << endl << "matches : " << matches << endl;

		current_path(baseDir);
		std::ofstream resultsOutFile(resultsOutFilename, ios::app);
		resultsOutFile << *it << '\t' << score << '\t' << matches << endl;
		resultsOutFile.close();
	}

	return 0;
}

list<string> loadModelNames(const char* modelsFilename) {
	list<string> modelNames;

	std::ifstream inFile(resultsInFilename);
	string line;

	if (inFile) {
		while (getline(inFile, line)) {
			modelNames.push_back(line);
		}
	}

	inFile.close();
	return modelNames;
}

#endif
