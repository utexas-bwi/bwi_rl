#ifndef BWI_RL_VI_TABULAR_ESTIMATOR_H
#define BWI_RL_VI_TABULAR_ESTIMATOR_H

#include <fstream>
#include <map>

#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/serialization/map.hpp>

#include <bwi_rl/planning/VIEstimator.h>

template<class State, class Action>
class VITabularEstimator : public VIEstimator<State, Action> {

  public:
    VITabularEstimator () {}
    virtual ~VITabularEstimator () {}

    virtual float getValue(const State &state) {
      return value_cache_[state];
    }

    virtual void updateValue(const State &state, float value) {
      value_cache_[state] = value;
    }

    virtual Action getBestAction(const State &state) {
      return best_action_cache_[state];

    }
    virtual void setBestAction(const State &state, const Action& action) {
      best_action_cache_[state] = action;
    }

    virtual void loadEstimatedValues(const std::string& file) {
      std::ifstream ifs(file.c_str());
      boost::archive::binary_iarchive ia(ifs);
      ia >> *this;
    }

    virtual void saveEstimatedValues(const std::string& file) {
      std::ofstream ofs(file.c_str());
      boost::archive::binary_oarchive oa(ofs);
      oa << *this;
    }

    virtual std::string generateDescription(unsigned int indentation = 0) {
      return std::string("");
    }

  private:

    std::map<State, float> value_cache_;
    std::map<State, Action> best_action_cache_;

    friend class boost::serialization::access;
    template<class Archive>
    void serialize(Archive & ar, const unsigned int version) {
      ar & BOOST_SERIALIZATION_NVP(value_cache_);
      ar & BOOST_SERIALIZATION_NVP(best_action_cache_);
    }

};


#endif /* end of include guard: BWI_RL_VI_TABULAR_ESTIMATOR_H */
