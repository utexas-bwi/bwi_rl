#ifndef MULTI_THREADED_MCTS_H
#define MULTI_THREADED_MCTS_H

#include <boost/thread/thread.hpp>
#include <boost/foreach.hpp>

#include <bwi_rl/common/RNG.h>
#include <bwi_rl/common/Params.h>
#include <bwi_rl/common/Util.h>

#include <bwi_rl/planning/Model.h>
/* #include <bwi_rl/planning/MultiThreadedMCTSEstimator.h> */
#include <bwi_rl/planning/ModelUpdater.h>
#include <bwi_rl/planning/StateMapping.h>

#include <tbb/concurrent_hash_map.h>

#ifdef MCTS_DEBUG
#define MCTS_OUTPUT(x) std::cout << x << std::endl
#else
#define MCTS_OUTPUT(x) ((void) 0)
#endif

#ifdef MCTS_TIMINGS
#include <bwi_rl/common/Enum.h>
#define MCTS_TIC(_) tic(MCTS_Timer::_)
#define MCTS_TOC(_) toc(MCTS_Timer::_)
#define MCTS_PRINT_TIMINGS() for(int i = 0; i < MCTS_Timer::NUM; i++) { \
  std::cout << getName((MCTS_Timer_t)i) << ": " << getTimer(i) << std::endl; \
}
#define MCTS_RESET_TIMINGS() for(int i = 0; i < MCTS_Timer::NUM; i++) { \
  resetTimer(i); \
}
ENUM(MCTS_Timer,
  TOTAL,
  SELECT_MODEL,
  SELECT_PLANNING_ACTION,
  TAKE_ACTION,
  VISIT,
  FINISH_ROLLOUT
)
#else
#define MCTS_TIC(_) (void)(0);
#define MCTS_TOC(_) (void)(0);
#define MCTS_PRINT_TIMINGS() (void)(0);
#define MCTS_RESET_TIMINGS() (void)(0);
#endif

class StateActionInfo {
  public:
    StateActionInfo(unsigned int visits, float val):
      visits(visits),
      val(val) {}
    unsigned int visits;
    float val;
};

class StateInfo {
  public:
    StateInfo() {}
    StateInfo(unsigned int numActions, unsigned int initialVisits) : 
        stateVisits(initialVisits) {
      actionInfos.resize(numActions);
    }

    std::vector<boost::shared_ptr<StateActionInfo> > actionInfos;
    unsigned int stateVisits;
};

template<class State, class StateHash, class Action>
class MultiThreadedMCTS {
  public:

    class HistoryStep {
      public:
        HistoryStep(const State &state, unsigned int action_idx, unsigned int num_actions, float reward):
          state(state), action_idx(action_idx), num_actions(num_actions), reward(reward) {}
        State state;
        unsigned int action_idx;
        unsigned int num_actions;
        float reward;
    };

    typedef boost::shared_ptr<MultiThreadedMCTS<State, StateHash, Action> > Ptr;
    /* typedef typename MultiThreadedMCTSEstimator<State,Action>::Ptr ValuePtr; */
    typedef typename ModelUpdater<State,Action>::Ptr ModelUpdaterPtr;
    typedef typename Model<State,Action>::Ptr ModelPtr;
    typedef typename StateMapping<State>::Ptr StateMappingPtr;
    typedef typename tbb::concurrent_hash_map<State, StateInfo, StateHash> StateInfoTable;

#define PARAMS(_) \
    _(unsigned int,maxDepth,maxDepth,0) \
    _(int,numThreads,numThreads,1) \
    _(float,lambda,lambda,0.0) \
    _(float,gamma,gamma,1.0) \
    _(float,rewardBound,rewardBound,10000) \
    _(float,maxNewNodesPerRollout,maxNewNodesPerRollout,5) \
    _(float,unknownActionValue,unknownActionValue,-1e10) \
    _(float,unknownActionPlanningValue,unknownActionPlanningValue,1e10) \
    _(float,unknownBootstrapValue,unknownBootstrapValue,0.0) \
    _(bool,theoreticallyCorrectLambda,theoreticallyCorrectLambda,false) \

    Params_STRUCT(PARAMS)
#undef PARAMS

    MultiThreadedMCTS (/*ValuePtr valueEstimator, */ModelUpdaterPtr modelUpdater,
        StateMappingPtr stateMapping, boost::shared_ptr<RNG> rng, const Params &p);
    virtual ~MultiThreadedMCTS () {}

    unsigned int search(const State &startState, 
        unsigned int& termination_count,
        float maxPlanningtime = 1.0,
        int maxPlayouts = 0);
    void singleThreadedSearch();
    Action selectWorldAction(const State &state);
    void restart();
    std::string generateDescription(unsigned int indentation = 0);

  private:
    float calcActionValue(const boost::shared_ptr<StateActionInfo> &actionInfo,
        const StateInfo &state, bool usePlanningBounds);
    float maxValueForState(const State &state, const StateInfo& stateInfo);
    Action selectAction(const State &state, bool usePlanningBounds, 
        unsigned int& action_idx, unsigned int& numActions);

  private:
    ModelPtr model;
    ModelUpdaterPtr modelUpdater;
    StateMappingPtr stateMapping;
    bool valid;

    State startState;
    unsigned int maxPlayouts;
    unsigned int currentPlayouts;
    unsigned int terminatedPlayouts;

    float maxPlanningTime;
    float endPlanningTime;

    Params p;
    boost::shared_ptr<RNG> rng;

    StateInfoTable stateInfoTable;
};

template<class State, class StateHash, class Action>
MultiThreadedMCTS<State, StateHash, Action>::MultiThreadedMCTS(
    /* ValuePtr valueEstimator, */
    ModelUpdaterPtr modelUpdater, StateMappingPtr stateMapping, 
    boost::shared_ptr<RNG> rng, const Params &p) : // valueEstimator(valueEstimator),
    modelUpdater(modelUpdater), stateMapping(stateMapping), rng(rng), p(p) {}

template<class State, class StateHash, class Action>
unsigned int MultiThreadedMCTS<State, StateHash, Action>::search(
    const State &startState, 
    unsigned int &termination_count,
    float maxPlanningTime, int maxPlayouts) {

  if (maxPlanningTime < 0) {
    std::cerr << "Invalid maxPlanningTime, must be >= 0" << std::endl;
    maxPlayouts = 1000;
  } else if ((maxPlayouts == 0) && (maxPlanningTime <= 0)) {
    std::cerr << "Must stop planning at some point, either specify "
              << "maxPlayouts or maxPlanningTime" << std::endl;
    maxPlayouts = 1000;
  }

  this->startState = startState;
  this->maxPlayouts = maxPlayouts; 
  this->maxPlanningTime = maxPlanningTime;
  currentPlayouts = 0;
  terminatedPlayouts = 0;
  endPlanningTime = getTime() + maxPlanningTime;

  MCTS_RESET_TIMINGS();
  MCTS_TIC(TOTAL);

  std::vector<boost::shared_ptr<boost::thread> > threads; 
  // Launch n - 1 threads;
  for (int n = 1; n < p.numThreads; ++n) {
    boost::shared_ptr<boost::thread> thread(new
        boost::thread(&MultiThreadedMCTS<State, StateHash, Action>::singleThreadedSearch,
          this));
  }

  // Launch search in the main thread
  singleThreadedSearch();

  // Join all threads;
  for (int n = 1; n < p.numThreads; ++n) {
    threads[n - 1]->join();
  }

  MCTS_TOC(TOTAL);
  MCTS_PRINT_TIMINGS();

  termination_count = terminatedPlayouts;
  return currentPlayouts;
}

template<class State, class StateHash, class Action>
void MultiThreadedMCTS<State, StateHash, Action>::singleThreadedSearch() {

  while ((maxPlanningTime <= 0.0) || (getTime() < endPlanningTime) && 
         (maxPlayouts <= 0) || (currentPlayouts < maxPlayouts)) {

    //TODO not locked
    ++currentPlayouts;

    MCTS_OUTPUT("------------START ROLLOUT--------------");
    MCTS_TIC(SELECT_MODEL);
    ModelPtr model = modelUpdater->selectModel(startState);
    this->model = model;
    MCTS_TOC(SELECT_MODEL);
    State state(startState);
    State newState;
    Action action;
    unsigned int action_idx, num_actions;
    float reward;
    bool terminal = false;
    int depth_count;

    stateMapping->map(state); // discretize state
    
    std::vector<HistoryStep> history;
    std::map<State, unsigned int> stateCount;

    for (unsigned int depth = 0; (depth < p.maxDepth) || (p.maxDepth == 0); depth+=depth_count) {
      MCTS_OUTPUT("MCTS State: " << state << " " << "DEPTH: " << depth);
      if (terminal || ((maxPlanningTime > 0) && (getTime() > endPlanningTime)))
        break;
      MCTS_TIC(SELECT_PLANNING_ACTION);
      action = selectAction(state, true, action_idx, num_actions);
      MCTS_OUTPUT("Action: " << action);
      MCTS_TOC(SELECT_PLANNING_ACTION);
      MCTS_TIC(TAKE_ACTION);
      model->takeAction(action, reward, newState, terminal, depth_count);
      MCTS_TOC(TAKE_ACTION);
      modelUpdater->updateSimulationAction(action, newState);
      MCTS_TIC(VISIT);
      stateCount[state] += 1; // Should construct with 0 if unavailable
      history.push_back(HistoryStep(state, action_idx, num_actions, reward));
      MCTS_TOC(VISIT);
      state = newState;
      stateMapping->map(state); // discretize state
    }

    if (terminal) {
      ++terminatedPlayouts;
    }

    MCTS_OUTPUT("------------ BACKPROP --------------");
    float value = 0;
    if (!terminal) {
      // Get bootstrap value of final state
      // http://stackoverflow.com/questions/11275444/c-template-typename-iterator
      typename StateInfoTable::const_accessor a;
      if (!stateInfoTable.find(a, state)) { // The state does not exist, choose an action randomly
        value = p.unknownBootstrapValue;
      } else {
        StateInfo stateInfo = a->second; // Create copy and release lock.
        a.release(); // TODO improper locking
        value = maxValueForState(state, stateInfo);
      }
    }

    typedef std::pair<State, StateInfo> StateToInfoPair;
    std::vector<StateToInfoPair> statesToAdd;

    for (int step = history.size() - 1; step >= 0; --step) {

      State &state = history[step].state;
      unsigned int &action_idx = history[step].action_idx;
      unsigned int &num_actions = history[step].num_actions;
      float &reward = history[step].reward;

      value = reward + p.gamma * value;
      
      // Get information about this state
      typename StateInfoTable::const_accessor a;
      StateInfo stateInfo(num_actions, 0); 
      bool is_new_state = true;
      if (stateInfoTable.find(a, state)) { // The state does not exist, choose an action randomly and act
        stateInfo = a->second; // Create copy and release lock.
        a.release(); //TODO improper locking
        is_new_state = false;
      }

      if (p.theoreticallyCorrectLambda) {
        if (stateInfo.stateVisits != 0) {
          value = p.lambda * value + (1.0 - p.lambda) * maxValueForState(state, stateInfo);
        } // else don't change the value being backed up
      }

      // Modify the action appropriately
      stateCount[state]--;
      if (stateCount[state] == 0) { // First Visit Monte Carlo
        stateInfo.stateVisits++;
        if (!stateInfo.actionInfos[action_idx]) {
          stateInfo.actionInfos[action_idx].reset(new StateActionInfo(0, 0));
        }
        stateInfo.actionInfos[action_idx]->visits++;
        stateInfo.actionInfos[action_idx]->val += 
          (1.0 / stateInfo.actionInfos[action_idx]->visits) * 
          (value - stateInfo.actionInfos[action_idx]->val);

        if (is_new_state) {
          statesToAdd.push_back(std::make_pair(state, stateInfo));
        }
      }
      
      if (!p.theoreticallyCorrectLambda) {
        if (stateInfo.stateVisits != 0) {
          value = p.lambda * value + (1.0 - p.lambda) * maxValueForState(state, stateInfo);
        } // else don't change the value being backed up
      }

    }

    // Finally add the states
    for (int s = statesToAdd.size() - 1; 
         s >= 0 && s >= statesToAdd.size() - p.maxNewNodesPerRollout;
         --s) {
      stateInfoTable.insert(statesToAdd[s]);
    }
  }
}

template<class State, class StateHash, class Action>
float MultiThreadedMCTS<State, StateHash, Action>::calcActionValue(
    const boost::shared_ptr<StateActionInfo> &actionInfo, 
    const StateInfo &stateInfo,
    bool usePlanningBounds) {
  if (!actionInfo) {
    if (usePlanningBounds) {
      return p.unknownActionPlanningValue;
    } else {
      return p.unknownActionValue;
    }
  }
  if (usePlanningBounds) {
    return actionInfo->val + p.rewardBound * sqrt(log(stateInfo.stateVisits) / actionInfo->visits);
  } else {
    return actionInfo->val;
  }
}

template<class State, class StateHash, class Action>
Action MultiThreadedMCTS<State, StateHash, Action>::selectWorldAction(const State &state) {
  State mappedState(state);
  stateMapping->map(mappedState); // discretize state
  unsigned int unused_action_idx, unused_num_actions;
  return selectAction(mappedState, false, unused_action_idx, unused_num_actions);
}

template<class State, class StateHash, class Action>
Action MultiThreadedMCTS<State, StateHash, Action>::selectAction(const State &state, 
    bool usePlanningBounds, unsigned int& action_idx, unsigned int& num_actions) {

  std::vector<Action> stateActions;
  this->model->getAllActions(state, stateActions);

  typename StateInfoTable::const_accessor a;
  if (!stateInfoTable.find(a, state)) { // The state does not exist, choose an action randomly
    action_idx = rng->randomInt(stateActions.size()); 
    return stateActions[action_idx];
  }
  StateInfo stateInfo = a->second; // Create copy and release lock.
  a.release(); // TODO improper locking

  int idx;
  float maxVal = -std::numeric_limits<float>::max();
  std::vector<unsigned int> maxActionIdx;
  unsigned int currentActionIdx = 0;
  BOOST_FOREACH(const boost::shared_ptr<StateActionInfo>& actionInfo, stateInfo.actionInfos) {
    float val = calcActionValue(actionInfo, stateInfo, usePlanningBounds);
    if (fabs(val - maxVal) < 1e-10) {
      maxActionIdx.push_back(currentActionIdx);
    } else if (val > maxVal) {
      maxVal = val;
      maxActionIdx.clear();
      maxActionIdx.push_back(currentActionIdx);
    }
    ++currentActionIdx;
  }
  num_actions = currentActionIdx;

  action_idx = maxActionIdx[rng->randomInt(maxActionIdx.size())];
  return stateActions[action_idx];
}

template<class State, class StateHash, class Action>
float MultiThreadedMCTS<State, StateHash, Action>::maxValueForState(const State &state,
    const StateInfo& stateInfo) {

  int idx;
  float maxVal = -std::numeric_limits<float>::max();
  unsigned int totalVisits = 0;
  BOOST_FOREACH(const boost::shared_ptr<StateActionInfo>& stateActionInfo, stateInfo.actionInfos) {
    float val = calcActionValue(stateActionInfo, stateInfo, false);
    totalVisits += stateActionInfo->visits;
    if (val > maxVal) {
      maxVal = val;
    }
  }

  return maxVal;
}

template<class State, class StateHash, class Action>
void MultiThreadedMCTS<State, StateHash, Action>::restart() {
  stateInfoTable.clear();
}

template<class State, class StateHash, class Action>
std::string MultiThreadedMCTS<State, StateHash, Action>::generateDescription(unsigned int indentation) {
  std::stringstream ss;
  std::string prefix = indent(indentation);
  std::string prefix2 = indent(indentation + 1);
  ss << prefix  << "MCTS" << std::endl;
  // TODO use the auto stringstream param function here
  return ss.str();
}

#endif /* end of include guard: MULTI_THREADED_MCTS_H */
