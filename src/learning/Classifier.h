#ifndef CLASSIFIER_4OFDPALS
#define CLASSIFIER_4OFDPALS

/*
File: Classifier.h
Author: Samuel Barrett
Description: abstract classifier
Created:  2011-11-22
Modified: 2011-12-27
*/

#include <boost/unordered_map.hpp>
#include "Common.h"

typedef boost::unordered_map<Instance,Classification> ClassificationCache;
std::size_t hash_value(const Instance &i);

class Classifier {
public:
  Classifier(const std::vector<Feature> &features, bool caching);

  virtual ~Classifier() {}
  void train(bool incremental=true);
  void classify(const InstancePtr &instance, Classification &classification);
  virtual void addData(const InstancePtr &instance) = 0;

  std::ostream& outputHeader(std::ostream &out) const;

protected:
  virtual void trainInternal(bool incremental) = 0;
  virtual void classifyInternal(const InstancePtr &instance, Classification &classification) = 0;


protected:
  std::vector<Feature> features;
  const std::string classFeature;
  unsigned int numClasses;
  bool caching;
  boost::shared_ptr<ClassificationCache> cache;
};

#endif /* end of include guard: CLASSIFIER_4OFDPALS */
