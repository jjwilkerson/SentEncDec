/**
 * @file
 * @brief Defines Trainer class.
 */

#include "Trainer.h"
#include <Network.h>
#include <layers/Layer.h>
#include <optimizers/Optimizer.h>
#include <input/WvCorpusIterator.h>
#include <gpu/CublasFunc.h>
#include <state/IterInfo.h>
#include <input/WordVectors.h>
#include <iostream>
#include <iomanip>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <ctime>
#include <boost/filesystem.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_int.hpp>
#include <boost/random/variate_generator.hpp>
#include <config/Config.h>
#include <state/State.h>
#include <input/SedInputSource.h>
#include <loss/LossFunction.h>
#include <utf8.h>

//#define SAVE_TEST_TOKENS 1
//#define SAVE_TEST_OUTPUT 1

typedef boost::mt19937 base_generator_type;

using namespace std;
using namespace boost::filesystem;

const char* scoresFilename = "iter_scores";
const char* weightsTemp1Filename = "encdeccu_weights_save.bin";
const char* weightsTemp2Filename = "encdeccu_weights_save.bak.bin";
const char* initDeltaTemp1Filename = "encdeccu_initDelta_save.bin";
const char* initDeltaTemp2Filename = "encdeccu_initDelta_save.bak.bin";
const char* curandStatesTemp1Filename = "encdeccu_curandStates_save.bin";
const char* curandStatesTemp2Filename = "encdeccu_curandStates_save.bak.bin";
const char* replacerStateTemp1Filename = "encdeccu_replacerState_save.bin";
const char* replacerStateTemp2Filename = "encdeccu_replacerState_save.bak.bin";
const char* stateTemp1Filename = "encdeccu_state_save.json";
const char* stateTemp2Filename = "encdeccu_state_save.bak.json";
const char* latestStateFilename = "state.json";
const char* iterInfoFilename = "iter_info";
const char* matchingFilename = "matching_counts";

const char* weightsEpochTemp1Filename = "encdeccu_weights_epoch_save.bin";
const char* weightsEpochTemp2Filename = "encdeccu_weights_epoch_save.bak.bin";
const char* initDeltaEpochTemp1Filename = "encdeccu_initDelta_epoch_save.bin";
const char* initDeltaEpochTemp2Filename = "encdeccu_initDelta_epoch_save.bak.bin";
const char* curandStatesEpochTemp1Filename = "encdeccu_curandStates_epoch_save.bin";
const char* curandStatesEpochTemp2Filename = "encdeccu_curandStates_epoch_save.bak.bin";
const char* replacerStateEpochTemp1Filename = "encdeccu_replacerState_epoch_save.bin";
const char* replacerStateEpochTemp2Filename = "encdeccu_replacerState_epoch_save.bak.bin";
const char* stateEpochTemp1Filename = "encdeccu_state_epoch_save.json";
const char* stateEpochTemp2Filename = "encdeccu_state_epoch_save.bak.json";

const char* color_red = "\033[31m";
const char* color_green = "\033[32m";
const char* color_reset = "\033[0m";

#ifdef SAVE_TEST_OUTPUT
std::ofstream matchesFile("test_matches.out", ios::trunc);
#endif

namespace netlib {

Trainer::Trainer(cublasHandle_t& handle, Config& config, Network& net, Optimizer& optimizer, WordVectors &wv, SedInputSource& trainSource,
		SedInputSource& testSource, SedInputSource& testMatchingSource)
	: handle(handle), config(config), net(net), optimizer(optimizer), wv(wv), trainSource(trainSource), testSource(testSource),
	  testMatchingSource(testMatchingSource) {
	ixFilenameBase = datasetsDir + "/sentence_ixs_epoch";
}

Trainer::~Trainer() {
}

dtype2 Trainer::train(string outputDirName, int startEpoch, int startIter, int maxIter, int maxTimeS, unsigned int clockOffset,
		int testPeriod, int testMatchingPeriod, int demoPeriod, int savePeriod, int tempSavePeriod, int printPeriod,
		int maxEpochs, bool debug, dtype2* scoreFinal, int* matchesFinal) {

	path outputDir(outputDirName);
	if (!exists(outputDir)) {
		create_directories(outputDir);
	}
	current_path(outputDir);

	long startTime = clock();

	cout << "maxIter: " << maxIter << endl;
	cout << "maxTimeS: " << maxTimeS << endl;

	dtype2 score = 0.0;
	int iterNum = startIter;
	bool lastIter = false;
	bool firstEpoch = true;
	long offTime = 0;

	bool testFirst = false;
	bool demoFirst = false;

	for (int ep = startEpoch; ep < maxEpochs && !lastIter; ep++) {
		cout << "========================================" << endl;
		cout << "Epoch " << ep << endl;

		time_t curtime;

//		cudaProfilerStart();

		bool epochEnd = false;

		while (trainSource.hasNext() && !lastIter && !epochEnd) {
			IterInfo iterInfo(iterNum);

			bool printing = ((printPeriod > 0 && (iterNum % printPeriod == 0)) || debug);

#ifndef DEBUG
			if (testFirst && iterNum == 0) {
				long before = clock();

//				demo(trainSource);

//				int numWords, tightMatches, looseMatches;
//				testMatching(testMatchingSource, &numWords, &tightMatches, &looseMatches);

				score = test(testSource);

				testFirst = false;
				long after = clock();
				offTime += (after - before) / CLOCKS_PER_SEC;
			}

			if (iterNum % demoPeriod == 0 || demoFirst) {
				long before = clock();
				demo(trainSource);
				long after = clock();
				demoFirst = false;
				offTime += (after - before) / CLOCKS_PER_SEC;
			}
#endif

			long now = clock();
			long elapsed = (now - startTime) / CLOCKS_PER_SEC;
			elapsed -= offTime;

			iterInfo.clock = elapsed + clockOffset;

			if ((maxIter >= 0 && iterNum >= maxIter) ||
				(maxTimeS > 0 && elapsed >= maxTimeS)) {
				lastIter = true;
			}

			if (printing) {
				time(&curtime);
				cout << endl << ctime(&curtime) << "Iteration " << iterNum << endl;

				cout << "sentence ix " << trainSource.firstBatchIx << endl;
			}
			iterInfo.sentenceIx = trainSource.firstBatchIx;

			trainSource.toFirstBatch();

//			net.resetSavedMasks();

			optimizer.computeUpdate(iterInfo, printing);

			long clockNow = ((clock() - startTime) / CLOCKS_PER_SEC) + clockOffset;
			clockNow -= offTime;

			int saveEpoch = ep;

			trainSource.toNextBatchSet();
			if (!trainSource.hasNextBatchSet()) {
				epochEnd = true;

				long before = clock();
				string ixFilename = buildIxFilename(ep + 1);
				path ixPath(ixFilename);
				if (exists(ixPath)) {
					cout << endl << "Loading corpus indices" << endl;
					trainSource.loadIxs(ixFilename);
				} else {
					cout << endl << "Shuffling training corpus" << endl;
					trainSource.shuffle();
					trainSource.saveIxs(ixFilename);
				}
				trainSource.reset();
				long after = clock();
				offTime += (after - before) / CLOCKS_PER_SEC;

				saveEpoch = ep + 1;
				save(saveEpoch, iterNum, trainSource.firstBatchIx, clockNow);
			}

			if ((iterNum % tempSavePeriod == 0) && (iterNum != 0)) {
				cout << endl << "offTime: " << offTime << endl;
				tempSave(saveEpoch, iterNum, trainSource.firstBatchIx, clockNow);
			}

			if (((iterNum % savePeriod == 0) && (iterNum != 0)) || lastIter) {
				save(saveEpoch, iterNum, trainSource.firstBatchIx, clockNow);
			}

			if (((iterNum % testPeriod == 0) && (iterNum != 0)) || lastIter) {
				dtypeh loss;
				long before = clock();
				score = test(testSource, &loss);
				saveScore(iterNum, score, loss, clockNow);
				long after = clock();
				offTime += (after - before) / CLOCKS_PER_SEC;
				//if best score, save score and W
				if (scoreFinal != NULL) {
					*scoreFinal = score;
				}
			}

#ifndef DEBUG
			if (((iterNum % testMatchingPeriod == 0) && (iterNum != 0)) || lastIter) {
				long before = clock();
				int numWords, tightMatches, looseMatches;
				dtypeh allMatch;
				testMatching(testMatchingSource, &numWords, &tightMatches, &looseMatches,
						&allMatch);
				saveMatching(iterNum, numWords, tightMatches, looseMatches, allMatch, clockNow);
				long after = clock();
				offTime += (after - before) / CLOCKS_PER_SEC;
				if (matchesFinal != NULL) {
					*matchesFinal = tightMatches;
				}
			}
#endif
			if (iterNum == 0) {
				IterInfo::saveHeader(iterInfoFilename);
			}
			if (printing) {
				iterInfo.save(iterInfoFilename);
			}

			iterNum++;
		}

		firstEpoch = false;
	}

	return score;
}

void Trainer::saveScore(int iterNum, dtype2 score, dtype2 loss, long clock) {
	std::ofstream file(scoresFilename, ios::app);
	file << fixed << setprecision(4);
	file << iterNum << '\t' << score << '\t' << loss << '\t' << clock << endl;
	file.close();
}

void Trainer::saveMatching(int iterNum, int numWords, int tightMatches, int looseMatches,
		dtypeh allMatch, long clock) {
	std::ofstream file(matchingFilename, ios::app);
	dtypeh matchRate = (dtypeh) tightMatches / (dtypeh) numWords;
	file << iterNum << '\t' << numWords << '\t' << tightMatches << '\t' << looseMatches
			<< '\t' << clock << '\t' << matchRate << '\t' << allMatch << endl;
	file.close();
}

void Trainer::save(int epochNum, int iterNum, int sentenceIx, long clock) {
	static char buf[100];
	sprintf(buf, "encdeccu_weights_iter%d.bin", iterNum);
	net.saveWeights(buf);
	string weightsFilename(buf);

	sprintf(buf, "encdeccu_initDelta_iter%d.bin", iterNum);
	optimizer.saveInitDelta(buf);
	string initDeltaFilename(buf);

	sprintf(buf, "encdeccu_curandStates_iter%d.bin", iterNum);
	net.saveCurandStates(buf);
	string curandStatesFilename(buf);

	sprintf(buf, "encdeccu_replacerState_iter%d.bin", iterNum);
	trainSource.saveReplacerState(buf);
	string replacerStateFilename(buf);

	sprintf(buf, "encdeccu_state_iter%d.json", iterNum);
	saveState(buf, weightsFilename, initDeltaFilename, curandStatesFilename, replacerStateFilename,
			epochNum, iterNum, sentenceIx, clock);

	path toPath(buf);
	path fromPath(latestStateFilename);
	if (exists(fromPath)) {
		remove(fromPath);
	}
	create_symlink(toPath, fromPath);
}

void Trainer::tempSave(int epochNum, int iterNum, int sentenceIx, long clock) {
	path weightsPath1(weightsTemp1Filename);
	path weightsPath2(weightsTemp2Filename);
	path idPath1(initDeltaTemp1Filename);
	path idPath2(initDeltaTemp2Filename);
	path csPath1(curandStatesTemp1Filename);
	path csPath2(curandStatesTemp2Filename);
	path rsPath1(replacerStateTemp1Filename);
	path rsPath2(replacerStateTemp2Filename);
	path sPath1(stateTemp1Filename);
	path sPath2(stateTemp2Filename);

	if (exists(weightsPath2)) {
		remove(weightsPath2);
	}
	if (exists(idPath2)) {
		remove(idPath2);
	}
	if (exists(csPath2)) {
		remove(csPath2);
	}
	if (exists(rsPath2)) {
		remove(rsPath2);
	}
	if (exists(sPath2)) {
		remove(sPath2);
	}

	if (exists(weightsPath1)) {
		rename(weightsPath1, weightsPath2);
	}
	net.saveWeights(weightsTemp1Filename);

	if (exists(idPath1)) {
		rename(idPath1, idPath2);
	}
	optimizer.saveInitDelta(initDeltaTemp1Filename);

	if (exists(csPath1)) {
		rename(csPath1, csPath2);
	}
	net.saveCurandStates(curandStatesTemp1Filename);

	if (exists(rsPath1)) {
		rename(rsPath1, rsPath2);
	}
	trainSource.saveReplacerState(replacerStateTemp1Filename);

	if (exists(sPath1)) {
		rename(sPath1, sPath2);
	}

	saveState(stateTemp1Filename, weightsTemp1Filename, initDeltaTemp1Filename, curandStatesTemp1Filename,
			replacerStateTemp1Filename, epochNum, iterNum, sentenceIx, clock);

	path toPath(stateTemp1Filename);
	path fromPath(latestStateFilename);
	if (exists(fromPath)) {
		remove(fromPath);
	}
	create_symlink(toPath, fromPath);
}

void Trainer::saveState(const char* stateFilename, string weightsFilename, string initDeltaFilename,
		string curandStatesFilename, string replacerStateFilename,int epochNum, int iterNum,
		int sentenceIx, long clock) {
	State state;
	state.weightsFilename = weightsFilename;
	state.initDeltaFilename = initDeltaFilename;
	state.curandStatesFilename = curandStatesFilename;
	state.replacerStateFilename = replacerStateFilename;
	state.ixFilename = trainSource.getIxFilename();

	state.epoch = epochNum;
	state.iter = iterNum;
	state.sentenceIx = sentenceIx;
	state.damping = optimizer.getDamping();
	state.deltaDecay = optimizer.getDeltaDecay();
	state.l2 = net.l2;
	state.numBatchGrad = net.numBatchGrad;
	state.numBatchG = net.numBatchG;
	state.numBatchError = net.numBatchError;
	state.maxIterCG = optimizer.getMaxIterCG();
	state.clock = clock;
	state.learningRate = optimizer.getLearningRate();
	state.lossScaleFac = net.getLossScaleFac();
	state.iterNoOverflow = net.getIterNoOverflow();

	state.save(stateFilename);
}

string Trainer::buildIxFilename(int epochNum) {
	static char buf[100];
	sprintf(buf, "%s%d", ixFilenameBase.c_str(), epochNum);
	return string(buf);
}

void Trainer::epochCopyTempSaves() {
	path weightsPath1(weightsTemp1Filename);
	path weightsPath2(weightsTemp2Filename);
	path weightsEpochPath1(weightsEpochTemp1Filename);
	path weightsEpochPath2(weightsEpochTemp2Filename);
	if (exists(weightsPath1)) {
		copy(weightsPath1, weightsEpochPath1);
	}
	if (exists(weightsPath2)) {
		copy(weightsPath2, weightsEpochPath2);
	}

	path idPath1(initDeltaTemp1Filename);
	path idPath2(initDeltaTemp2Filename);
	path idEpochPath1(initDeltaEpochTemp1Filename);
	path idEpochPath2(initDeltaEpochTemp2Filename);
	if (exists(idPath1)) {
		copy(idPath1, idEpochPath1);
	}
	if (exists(idPath2)) {
		copy(idPath2, idEpochPath2);
	}

	path csPath1(curandStatesTemp1Filename);
	path csPath2(curandStatesTemp2Filename);
	path csEpochPath1(curandStatesEpochTemp1Filename);
	path csEpochPath2(curandStatesEpochTemp2Filename);
	if (exists(csPath1)) {
		copy(csPath1, csEpochPath1);
	}
	if (exists(csPath2)) {
		copy(csPath2, csEpochPath2);
	}

	path rsPath1(replacerStateTemp1Filename);
	path rsPath2(replacerStateTemp2Filename);
	path rsEpochPath1(replacerStateEpochTemp1Filename);
	path rsEpochPath2(replacerStateEpochTemp2Filename);
	if (exists(rsPath1)) {
		copy(rsPath1, rsEpochPath1);
	}
	if (exists(rsPath2)) {
		copy(rsPath2, rsEpochPath2);
	}

	path sPath1(stateTemp1Filename);
	path sPath2(stateTemp2Filename);
	path sEpochPath1(stateEpochTemp1Filename);
	path sEpochPath2(stateEpochTemp2Filename);
	if (exists(sPath1)) {
		copy(sPath1, sEpochPath1);
	}
	if (exists(sPath2)) {
		copy(sPath2, sEpochPath2);
	}
}

dtypeh Trainer::test(SedInputSource& source, dtypeh* loss) {
	cout << endl << "====================   TEST   ====================" << endl;

	int numBatches = 0;
	dtypeh totalError = 0.0;
	dtypeh totalLoss = 0.0;
	source.reset();
	int batchSize = source.getBatchSize();

	map<int,TestInfoByLength*> testInfoMap;

	net.copyParams();

	source.reset();
	while (source.hasNext() && numBatches < config.testMaxBatches) {
		source.next();
		net.forward(0, NULL, false, false, &source, false);

		dtypeh err = -cosineSimBatch(batchSize, &testInfoMap);
		totalError += err;

		totalLoss += net.lossFunction.batchLoss(net.getOutputLayer()->activation, net.getTargets(), NULL,
				false, NULL, net.getDInputLengths()[0]);

		numBatches++;
	}

	dtypeh avgErr = totalError / numBatches;
	dtypeh avgLoss = totalLoss / (batchSize * numBatches);
	cout << endl << "test error: " << avgErr << endl;
	cout << "test loss: " << avgLoss << endl;

	cout << endl << "error by length" << endl;
	map<int,TestInfoByLength*>::iterator it;
	for (it = testInfoMap.begin(); it != testInfoMap.end(); it++) {
		int length = it->first;
		TestInfoByLength* testInfo = it->second;
		assert(length == testInfo->sentenceLength);

		int numWords = length * testInfo->numSentences;
		dtypeh avgErr = testInfo->simSum / numWords;

		dtypeh stdErr = sqrt(testInfo->simSqSum/numWords - avgErr * avgErr);

//		cout << length << ": " << err << endl;
		cout << length << " (" << testInfo->numSentences << "): " << avgErr << ", " << stdErr << endl;

		delete testInfo;
	}

	cout << endl << "==================== TEST END ====================" << endl;

	if (loss != NULL) {
		*loss = avgLoss;
	}
	return avgErr;
}

dtypeh Trainer::cosineSimBatch(int batchSize, map<int, TestInfoByLength*>* testInfoMap) {
	dtypeh totalSim = 0.0;
	dtypeh eps = 1e-8;

	dtype2** outputs = net.getOutputLayer()->activation;
	unsigned int* inputLengths = net.getHInputLengths();
	dtype2** targets = net.getTargets();

	for (int i = 0; i < batchSize; i++) {
		dtypeh sentSim = 0.0;
		dtypeh sentSimSq = 0.0;
		unsigned int length = inputLengths[i];
		for (int s = 0; s < length; s++) {
			dtype2* output = outputs[s];
			dtype2* vecOut = output + IDX2(i,0,net.batchSize);
			dtype2* vecTarget = targets[s] + IDX2(i,0,net.batchSize);

			dtype2 normTarget_d;
			CublasFunc::nrm2(handle, wv.wvLength, vecTarget, net.batchSize, &normTarget_d);
			dtypeh normTarget = d2h(normTarget_d);
			if (vecUnitNorm) assert(abs(normTarget - 1.0) < 1e-6);

			dtype2 normOut_d;
			CublasFunc::nrm2(handle, wv.wvLength, vecOut, net.batchSize, &normOut_d);
			dtypeh normOut = d2h(normOut_d);
			normOut = max(normOut, eps);

			dtypeh sim;
			CublasFunc::dot(handle, wv.wvLength, vecOut, net.batchSize, vecTarget, net.batchSize, &sim);
			sim /= normOut;
			if (!vecUnitNorm) sim /= normTarget;
			totalSim += sim;
			sentSim += sim;
			sentSimSq += sim * sim;
		}

		if (testInfoMap != NULL) {
			TestInfoByLength* testInfo;
			map<int,TestInfoByLength*>::iterator it = testInfoMap->find(length);
			if (it == testInfoMap->end()) {
				testInfo = new TestInfoByLength();
				testInfo->sentenceLength = length;
				testInfo->numSentences = 1;
				testInfo->simSum = sentSim;
				testInfo->simSqSum = sentSimSq;
				testInfoMap->insert(pair<int,TestInfoByLength*>(length, testInfo));
			} else {
				testInfo = it->second;
				assert(length == testInfo->sentenceLength);
				testInfo->numSentences++;
				testInfo->simSum += sentSim;
				testInfo->simSqSum += sentSimSq;
			}
		}
	}

	return totalSim / batchSize;
}

#ifndef DEBUG
void Trainer::testMatching(SedInputSource& source, int* numWordsOut, int* tightMatchesOut, int* looseMatchesOut,
		dtypeh* allMatchOut) {
	cout << endl << "=================   MATCH TEST   =================" << endl;

	long startTime = clock();

	int numBatches = 0;

	net.copyParams();

	source.reset();

	unsigned testBatchSize = source.getCorpus().getBatchSize();

	vector<string> tokensRet[testBatchSize];
	map<int,TestInfoByLength*> testInfoMap;

#ifdef SAVE_TEST_TOKENS
	std::ofstream tokensFile("test_tokens.out", ios::trunc);
#endif

	int totalNumWords = 0, totalTightMatches = 0, totalLooseMatches = 0, totalNumAllMatch = 0;
	while (source.hasNext() && numBatches < config.testMatchingMaxBatches) {
		source.next(tokensRet);

#ifdef SAVE_TEST_TOKENS
		vector<string>::iterator it;
		for (int i = 0; i < testBatchSize; i++) {
			bool first = true;
			for (it = tokensRet[i].begin(); it != tokensRet[i].end(); it++) {
				if (!first) {
					tokensFile << " ";
				}
				tokensFile << *it;
				first = false;
			}
			tokensFile << "\n";
		}
#endif

		net.forward(0, NULL, false, false, &source, false);

		int numWords, tightMatches, looseMatches, numAllMatch;
		matchBatch(testBatchSize, &numWords, &tightMatches, &looseMatches, &numAllMatch, tokensRet, &testInfoMap);
		totalNumWords += numWords;
		totalTightMatches += tightMatches;
		totalLooseMatches += looseMatches;
		totalNumAllMatch  += numAllMatch;

		numBatches++;
	}

#ifdef SAVE_TEST_TOKENS
	tokensFile.close();
#endif
#ifdef SAVE_TEST_OUTPUT
	matchesFile.close();
#endif

	dtypeh matchRate = (dtypeh) totalTightMatches / (dtypeh) totalNumWords;
	int numSentences = numBatches * testBatchSize;
	dtypeh allMatchRate = (dtypeh) totalNumAllMatch / (dtypeh) numSentences;

	cout << endl << "num words: " << totalNumWords << endl;
	cout << "tight matches: " << totalTightMatches << endl;
	cout << "loose matches: " << totalLooseMatches << endl;
	cout << "match rate: " << matchRate << endl;
	cout << "num sentences all match: " << totalNumAllMatch << endl;
	cout << "all match rate: " << allMatchRate << endl;

	cout << endl << "matches by length" << endl;
	map<int,TestInfoByLength*>::iterator it;
	for (it = testInfoMap.begin(); it != testInfoMap.end(); it++) {
		int length = it->first;
		TestInfoByLength* testInfo = it->second;
		assert(length == testInfo->sentenceLength);

		dtypeh avgMatch = (dtypeh) testInfo->numMatches / (dtypeh) testInfo->numWords;
		dtypeh allMatch = (dtypeh) testInfo->numAllMatch / (dtypeh) testInfo->numSentences;
		cout << length << " (" << testInfo->numSentences << "): " << avgMatch << ", "
				<< allMatch << endl;

		delete testInfo;
	}

	long stopTime = clock();
	long elapsed = (stopTime - startTime) / CLOCKS_PER_SEC;
	cout << endl << "test time: " << elapsed << " s" << endl;

	if (numWordsOut != NULL) {
		*numWordsOut = totalNumWords;
	}
	if (tightMatchesOut != NULL) {
		*tightMatchesOut = totalTightMatches;
	}
	if (looseMatchesOut != NULL) {
		*looseMatchesOut = totalLooseMatches;
	}
	if (allMatchOut != NULL) {
		*allMatchOut = allMatchRate;
	}

	cout << endl << "================= MATCH TEST END =================" << endl;

}
#endif

#ifndef DEBUG
void Trainer::matchBatch(int batchSize, int* numWords, int* tightMatches, int* looseMatches, int* numAllMatch, vector<string>* tokensRet,
		map<int,TestInfoByLength*>* testInfoMap) {
	dtype2** outputs = net.getOutputLayer()->activation;
	unsigned int* h_inputLengths = net.getHInputLengths();

	*numWords = 0;
	*tightMatches = 0;
	*looseMatches = 0;
	*numAllMatch = 0;

	bool* allMatch = new bool[batchSize];
	for (int i = 0; i < batchSize; i++) {
		allMatch[i] = true;
	}

#ifdef SAVE_TEST_MATCHES
	vector<string> batchesMatches[batchSize];
#endif

	for (int s = 0; s < net.maxSeqLength; s++) {
		dtype2* output = outputs[s];
		string* matches = wv.nearestBatch(output, batchSize);
//		string* matches = wv.nearestBatchLS(output, batchSize, h_inputLengths, s);

		for (int i = 0; i < batchSize; i++) {
			int sentLen = h_inputLengths[i];
			if (s >= sentLen) {
				continue;
			}

			(*numWords)++;

			vector<string> tokens = tokensRet[i];
			string token = tokens[s];

			string match = matches[i];

#ifdef SAVE_TEST_MATCHES
			batchesMatches[i].push_back(match);
#endif

			bool tightMatch = match == token;
			if (tightMatch) {
				(*tightMatches)++;
				(*looseMatches)++;
			} else {
				allMatch[i] = false;
				if (looseMatch(match, token)) {
					(*looseMatches)++;
				}
			}

			if (testInfoMap != NULL) {
				TestInfoByLength* testInfo;
				map<int,TestInfoByLength*>::iterator it = testInfoMap->find(sentLen);
				if (it == testInfoMap->end()) {
					testInfo = new TestInfoByLength();
					testInfo->sentenceLength = sentLen;
					testInfo->numSentences = 1;
					testInfo->numWords = 1;
					testInfo->numAllMatch = 0;
					if (tightMatch) {
						testInfo->numMatches = 1;
					} else {
						testInfo->numMatches = 0;
					}
					testInfoMap->insert(pair<int,TestInfoByLength*>(sentLen, testInfo));
				} else {
					testInfo = it->second;
					assert(sentLen == testInfo->sentenceLength);
					if (s == 0) {
						testInfo->numSentences++;
					}
					testInfo->numWords++;
					if (tightMatch) {
						testInfo->numMatches += 1;
					}
				}
			}
		}

		delete [] matches;
	}

#ifdef SAVE_TEST_OUTPUT
	for (int i = 0; i < batchSize; i++) {
		vector<string>* batchMatches = &batchesMatches[i];

		bool first = true;
		vector<string>::iterator it;
		for (it = batchMatches->begin(); it != batchMatches->end(); it++) {
			if (!first) {
				matchesFile << " ";
			}
			matchesFile << *it;
			first = false;
		}
		matchesFile << "\n";
	}
#endif

	for (int i = 0; i < batchSize; i++) {
		if (!allMatch[i]) {
			continue;
		}
		(*numAllMatch)++;

		int sentLen = h_inputLengths[i];
		map<int,TestInfoByLength*>::iterator it = testInfoMap->find(sentLen);
		assert (it != testInfoMap->end());
		TestInfoByLength* testInfo = it->second;
		testInfo->numAllMatch++;
	}

	wv.freeTemp();
}
#endif

bool Trainer::looseMatch(string s1, string s2) {
	string s1b;
	for (int i = 0; i < s1.length(); i++) {
		char c = tolower(s1[i]);
		if (c >= 'a' && c <= 'z') {
			s1b.push_back(c);
		}
	}

	string s2b;
	for (int i = 0; i < s2.length(); i++) {
		char c = tolower(s2[i]);
		if (c >= 'a' && c <= 'z') {
			s2b.push_back(c);
		}
	}

	if (s1b.length() == 0 && s2b.length() == 0) { //don't match all punct
		return false;
	}

	return s1b == s2b;
}

#ifndef DEBUG
void Trainer::demo(SedInputSource& source) {
	static unsigned int seed = static_cast<unsigned int>(time(0));
	static base_generator_type generator(seed);
	static boost::uniform_int<> uni_dist(0, source.size()-1);
	static boost::variate_generator<base_generator_type&, boost::uniform_int<> > uniRnd(generator, uni_dist);

	cout << endl << "====================   DEMO   ====================" << endl;

	net.copyParams();

	int sIx = uniRnd();
	string sentence;
	source.get(sIx, sentence);
	cout << "i: " << sentence << endl;
	cout << "---------------------------------------------------------------------------" << endl;

//	memset(h_inputLengths, 0, net.batchSize * sizeof(unsigned int));

	vector<string> tokens;

	source.inputFor(&sentence, 1, false, &tokens);
	net.forward(0, NULL, false, false, &source, false);

	dtype2** outputs = net.getOutputLayer()->activation;
	dtype2** targets = source.getTargets();
	unsigned int* h_inputLengths = source.getHInputLengths();

	dtype2 *vecOut_d, *targetVec;
	checkCudaErrors(cudaMalloc((void **)&vecOut_d, wv.wvLength * sizeof(dtype2)));
	checkCudaErrors(cudaMalloc((void **)&targetVec, wv.wvLength * sizeof(dtype2)));

	vector<string> outTokens;
	vector<dtypeh> similarities;
	for (int s = 0; s < net.maxSeqLength; s++) {
		if (s >= h_inputLengths[0]) {
			break;
		}

		CublasFunc::copy(handle, wv.wvLength, outputs[s], net.batchSize, vecOut_d, 1);
		CublasFunc::copy(handle, wv.wvLength, targets[s], net.batchSize, targetVec, 1);

		wv.deviceNormalize(vecOut_d);
		pair<string,dtypeh> match = wv.nearest(vecOut_d);
		outTokens.push_back(match.first);

//		dtype2 *targetVec = wv.lookup(tokens[s]);
		dtypeh sim = wv.cosineSim(vecOut_d, targetVec);
		similarities.push_back(sim);
//		wv.freeVector(targetVec);
	}

	int lineLength = 70;
	formatDemo(tokens, outTokens, similarities, lineLength);

	cout << "---------------------------------------------------------------------------" << endl;
	dtypeh sim = cosineSimBatch(1);
	cout << "e: " << -sim << endl;
	cout << "s: " << (sim / outTokens.size()) << endl;

	cout << endl << "==================== DEMO END ====================" << endl;

	checkCudaErrors(cudaFree(vecOut_d));
	checkCudaErrors(cudaFree(targetVec));
}
#endif

void Trainer::formatDemo(vector<string> tokens, vector<string> outTokens, vector<dtypeh> similarities,
		int lineLength) {
	string strTokens(""), strOutTokens(""), strSim("");

	int seriesLength = tokens.size();
    for (int i = 0; i < seriesLength; i++) {
    	string token = tokens[i];
    	string outToken = outTokens[i];

    	stringstream ss;
    	ss << fixed << setprecision(2) << similarities[i];
    	string sim = ss.str();
    	if (sim[0] == '0') {
    		sim.erase(0, 1);
    	} else if (sim[0] == '-' && sim[1] == '0') {
    		sim.erase(1, 1);
    	} else if (sim == "1.00") {
    		sim = "1";
    	}

    	size_t maxLength = max(utf8Length(token), utf8Length(outToken));
    	maxLength = max(maxLength, sim.length());
    	int padLength = maxLength + 1;

    	strTokens += pad(token, padLength);
    	const char* color;
    	if (outToken == token) {
    		color = color_green;
    	} else {
    		color = color_red;
    	}
    	strOutTokens += color + pad(outToken, padLength);
    	strSim += pad(sim, padLength);
    }

    printDemoLines(strTokens, strOutTokens, strSim, lineLength);
}

int Trainer::utf8Length(string s) {
	return utf8::distance(s.begin(), s.end());
}

string Trainer::replaceAllDigits(string in) {
	vector<string> words;
	boost::split(words, in, boost::is_any_of(" "));
	for (int i = 0; i < words.size(); i++) {
		string word = words[i];
		if (word == "zero") {
			words[i] = "0";
		} else if (word == "one") {
			words[i] = "1";
		} else if (word == "two") {
			words[i] = "2";
		} else if (word == "three") {
			words[i] = "3";
		} else if (word == "four") {
			words[i] = "4";
		} else if (word == "five") {
			words[i] = "5";
		} else if (word == "six") {
			words[i] = "6";
		} else if (word == "seven") {
			words[i] = "7";
		} else if (word == "eight") {
			words[i] = "8";
		} else if (word == "nine") {
			words[i] = "9";
		}
	}

	return boost::join(words, " ");
}

string Trainer::replaceDigit(string in) {
	if (in == "zero") {
		return "0";
	} else if (in == "one") {
		return "1";
	} else if (in == "two") {
		return "2";
	} else if (in == "three") {
		return "3";
	} else if (in == "four") {
		return "4";
	} else if (in == "five") {
		return "5";
	} else if (in == "six") {
		return "6";
	} else if (in == "seven") {
		return "7";
	} else if (in == "eight") {
		return "8";
	} else if (in == "nine") {
		return "9";
	} else {
		return in;
	}
}

string Trainer::pad(string token, int padLength) {
	int inLength = utf8Length(token);
	assert(inLength <= padLength);

	int diff = padLength - inLength;
	string out(token);
	for (int i = 0; i < diff; i++) {
		out += " ";
	}
	return out;
}

void Trainer::printDemoLines(string tokens, string out, string sim, int lineLength) {
	vector<string> tLines;
	vector<string> oLines;
	vector<string> sLines;

	do {
		int tLength = utf8Length(tokens);
		int oLength = utf8Length(out);
		int sLength = utf8Length(sim);
		int maxLength = max(max(tLength, oLength), sLength);

		if (maxLength <= lineLength) {
			tLines.push_back(tokens);
			oLines.push_back(out);
			sLines.push_back(sim);
			tokens = out = sim = "";
		} else {
			string::iterator itTokens = tokens.end();
			string::iterator itOut = out.end();
			string::iterator itSim = sim.end();

			char t, o, s;
			do {
				t = utf8::prior(itTokens, tokens.begin());
				o = utf8::prior(itOut, out.begin());
				if (o == 'm') {
					string esc = string(itOut-4, itOut-3);
					if (esc == "\033") {
						itOut -= 4;
						o = utf8::prior(itOut, out.begin());
					}
				}
				s = utf8::prior(itSim, sim.begin());
			} while (utf8::distance(tokens.begin(), itTokens) >= lineLength ||
					t != ' ' || o != ' ' || s != ' ');

			tLines.push_back(string(tokens.begin(), itTokens));
			oLines.push_back(string(out.begin(), itOut));
			sLines.push_back(string(sim.begin(), itSim));

			tokens = string(itTokens+1, tokens.end());
			out = string(itOut+1, out.end());
			sim = string(itSim+1, sim.end());
		}
	} while (tokens.length() > 0);

	for (int l = 0; l < tLines.size(); l++) {
		cout << endl << "t: " << tLines[l];
		cout << endl << "o: " << oLines[l] << color_reset;
		cout << endl << "s: " << sLines[l] << endl;
	}
}

} /* namespace netlib */
