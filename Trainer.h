/**
 * @file
 * @brief Declares Trainer class
 *
 */

#ifndef TRAINER_H_
#define TRAINER_H_

#include <NetLib.h>
#include "cublas_v2.h"
#include <string>
#include <map>
#include <vector>

using namespace std;

namespace netlib {

class Config;
class Network;
class Optimizer;
class SedInputSource;
class WordVectors;
struct TestInfoByLength;

/**
 * @brief Trainer class
 *
 * Trains the given network with the given optimizer and other objects.
 */
class Trainer {
public:
	Trainer(cublasHandle_t& handle, Config& config, Network& net, Optimizer& optimizer, WordVectors &wv, SedInputSource& trainSource,
			SedInputSource& testSource, SedInputSource& testMatchingSource);
	virtual ~Trainer();
	dtype2 train(string outputDirName, int startEpoch = 0, int startIter = 0, int maxIter = -1, int maxTimeS = 0,
			unsigned int clockOffset = 0, int testPeriod = 2000, int testMatchingPeriod = 10000, int demoPeriod = 1000,
			int savePeriod = 10000, int tempSavePeriod = 200, int printPeriod = 20, int maxEpochs = 100, bool debug = false,
			dtype2* scoreFinal = NULL, int* matchesFinal = NULL);
private:
	string ixFilenameBase;
	cublasHandle_t& handle;
	Config& config;
	Network& net;
	Optimizer& optimizer;
	WordVectors &wv;
	SedInputSource& trainSource;
	SedInputSource& testSource;
	SedInputSource& testMatchingSource;
	void saveScore(int iterNum, dtype2 score, dtype2 loss, long clock);
	void saveMatching(int iterNum, int numWords, int tightMatches, int looseMatches,
			dtypeh allMatch, long clock);
	void save(int epochNum, int iterNum, int sentenceIx, long clock);
	void tempSave(int epochNum, int iterNum, int sentenceIx, long clock);
	void saveState(const char* stateFilename, string weightsFilename, string initDeltaFilename,
			string curandStatesFilename, string replacerStateFilename, int epochNum, int iterNum,
			int sentenceIx, long clock);
	string buildIxFilename(int epochNum);
	void epochCopyTempSaves();
	dtypeh test(SedInputSource& source, dtypeh* loss = NULL);
	dtypeh cosineSimBatch(int batchSize, map<int,TestInfoByLength*>* testInfoMap = NULL);
	void testMatching(SedInputSource& source, int* numWordsOut = NULL, int* tightMatchesOut = NULL,
			int* looseMatchesOut = NULL, dtypeh* allMatchOut = NULL);
	void matchBatch(int batchSize, int* numWords, int* tightMatches, int* looseMatches, int* numAllMatch,
			vector<string>* tokensRet, map<int,TestInfoByLength*>* testInfoMap = NULL);
	bool looseMatch(string s1, string s2);
	void demo(SedInputSource& source);
	void formatDemo(vector<string> tokens, vector<string> outTokens, vector<dtypeh> similarities,
			int lineLength);
	string replaceAllDigits(string in);
	string replaceDigit(string in);
	string pad(string token, int padLength);
	int utf8Length(string s);
	void printDemoLines(string tokens, string out, string sim, int lineLength);
};

struct TestInfoByLength {
	int sentenceLength;
	int numSentences;
	dtypeh simSum;
	dtypeh simSqSum;
	int numWords;
	int numMatches;
	int numAllMatch;
};

} /* namespace netlib */

#endif /* TRAINER_H_ */
