/**
 * @file
 * @brief Entry point for a random search of the hyperparameter space.
 *
 * Used to build an executable for a random search for hyperparameters. Enabled by building with -DPROGRAM_VERSION=2
 */

#if PROGRAM_VERSION == 2

#include <input/WordVectors.h>
#include <input/WvCorpusIterator.h>
#include <config/Config.h>
#include <state/State.h>
#include <boost/filesystem.hpp>

using namespace std;
using namespace boost::filesystem;
using namespace netlib;

const char* configFilename = "config.json";
const char* resultsFilename = "results";
int maxIter = 11700;
int maxTimeS = 0;

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
	const char* baseDirName = sDir.c_str();

	cout << "base dir: " << baseDirName << endl;

	path baseDir(baseDirName);
	if (!exists(baseDir)) {
		create_directories(baseDir);
	}
	current_path(baseDir);

	static char dirName[100];
	time_t startTime;
	time(&startTime);
	int start = (int) startTime;

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
	string trainFilename = datasetsDir + "/news.2007-2017.en.proc3_sym9_max59_train";
	WvCorpusIterator trainCorpus(config, wv, trainFilename, batchSize, maxSeqLength, true, true, 200, ixFilename, 0);

	cout << endl << "Loading test corpus" << endl;
	string testFilename = datasetsDir + "/news.2007-2017.en.proc3_sym9_max59_tune";
	WvCorpusIterator testCorpus(config, wv, testFilename, batchSize, maxSeqLength, false, false);
	WvCorpusIterator testMatchingCorpus(config, wv, testFilename, batchSize, maxSeqLength, false, false);

	int netNum = 1;
	while (true) {
		sprintf(dirName, "sentencdec_%d_%d", start, netNum);
		path netDir = absolute(dirName, baseDir);
		if (exists(netDir)) {
			cout << "dir " << dirName << " already exists" << endl;
			exit(1);
		}
		create_directories(netDir);
		current_path(netDir);

		cout << endl << "net dir " << dirName << endl;

		Config* config = Config::random();
		config->batchSize = batchSize;
		config->maxSeqLength = maxSeqLength;
		config->wvLength = wvLength;
		config->save(configFilename);

		State* state = new State();
		state->ixFilename = ixFilename;
		state->curandStatesFilename = "";
		state->initDeltaFilename = "";
		state->epoch = 0;
		state->iter = -1;
		state->sentenceIx = 0;
		state->clock = 0;

		trainCorpus.reset();

		dtype2 score;
		int matches;
		run(netDir.c_str(), *config, state, wv, trainCorpus, testCorpus, testMatchingCorpus, handle, maxIter, maxTimeS, &score,
				&matches);
		if (score == 0.0) {
			netNum++;
			continue;
		}

		cout << endl << "score : " << score << endl;
		cout << endl << "matches : " << matches << endl;


		current_path(baseDir);
		std::ofstream resultsFile(resultsFilename, ios::app);
		resultsFile << dirName << '\t' << score << '\t' << matches << endl;
		resultsFile.close();

		delete config;
		delete state;

		netNum++;
	}

    cublasDestroy(handle);

	return 0;
}

#endif
