#ifndef MODEL_RSATLNKK
#define MODEL_RSATLNKK

/*
File: Model.h
Author: Samuel Barrett
Description: An abstract generative model for planning. Any implementation of
this model should be thread-safe, where a single model should not have any state
changes during a single monte-carlo rollout. 
Created:  2011-08-23
Modified: 2013-08-08
*/

#include <bwi_rl/common/RNG.h>
#include <boost/shared_ptr.hpp>
#include <string>
#include <vector>

template<class State, class Action>
class Model {
public:
  typedef boost::shared_ptr<Model<State,Action> > Ptr;

  Model () {}
  virtual ~Model () {}

  virtual void takeAction(const State &state, const Action &action, float &reward, State &next_state, bool &terminal, int &depth_count, boost::shared_ptr<RNG> rng) = 0;
  virtual void getFirstAction(const State &state, Action &action) = 0;
  virtual bool getNextAction(const State &state, Action &action) = 0; // returns true if there is a next action, else false

  virtual void getAllActions(const State &state, std::vector<Action>& stateActions) = 0;

  virtual std::string generateDescription(unsigned int indentation = 0) = 0;
};

#endif /* end of include guard: MODEL_RSATLNKK */
