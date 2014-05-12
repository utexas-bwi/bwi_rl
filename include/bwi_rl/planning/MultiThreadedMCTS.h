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
        double maxPlanningtime = 1.0,
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

    std::string getStateValuesDescription(const State& state);
    std::string getStateTableDescription();

  private:
    ModelPtr model;
    ModelUpdaterPtr modelUpdater;
    StateMappingPtr stateMapping;
    bool valid;

    State startState;
    unsigned int maxPlayouts;
    unsigned int currentPlayouts;
    unsigned int terminatedPlayouts;

    double maxPlanningTime;
    double endPlanningTime;

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
    double maxPlanningTime, int maxPlayouts) {

  if (maxPlanningTime < 0) {
    std::cerr << "Invalid maxPlanningTime, must be >= 0" << std::endl;
    maxPlayouts = 1000;
  } else if ((maxPlayouts == 0) && (maxPlanningTime <= 0)) {
    std::cerr << "Must stop planning at some point, either specify "
              << "maxPlayouts or maxPlanningTime" << std::endl;
    maxPlayouts = 1000;
  }

  this->model = modelUpdater->selectModel(startState);
  this->startState = startState;
  this->maxPlayouts = maxPlayouts; 
  this->maxPlanningTime = maxPlanningTime;
  currentPlayouts = 0;
  terminatedPlayouts = 0;
  endPlanningTime = getTime() + maxPlanningTime;

  // std::cout << "Starting search: " << maxPlayouts << " " << maxPlanningTime << std::endl;

  // std::cin.ignore().get();

  MCTS_RESET_TIMINGS();
  MCTS_TIC(TOTAL);

  std::vector<boost::shared_ptr<boost::thread> > threads; 
  // Launch n - 1 threads;
  /* std::cout << "MCTS: Spawning " << p.numThreads << " threads." << std::endl;  */
  for (int n = 1; n < p.numThreads; ++n) {
    boost::shared_ptr<boost::thread> thread(new
        boost::thread(&MultiThreadedMCTS<State, StateHash, Action>::singleThreadedSearch,
          this));
    threads.push_back(thread);
  }

  // std::cout << "launching thread" << std::endl;
  // std::cin.ignore().get();

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

  // TODO if this line is not here, bad stuff happens
  // std::cout << "Rolling out!" << ((maxPlanningTime <= 0.0) || (getTime() < endPlanningTime)) << 
  //   " " << getTime() << " " << endPlanningTime << std::endl;

#ifdef MCTS_DEBUG
  int count = 0;
#endif

  while (((maxPlanningTime <= 0.0) || (getTime() < endPlanningTime)) && 
         ((maxPlayouts <= 0) || (currentPlayouts < maxPlayouts))) {

    //TODO not locked
    ++currentPlayouts;

    MCTS_OUTPUT("------------START ROLLOUT--------------");
    State state(startState), discretizedState(startState);
    State newState;
    Action action;
    unsigned int action_idx, num_actions;
    float reward;
    bool terminal = false;
    int depth_count;

    stateMapping->map(discretizedState); // discretize state
    
    std::vector<HistoryStep> history;
    std::map<State, unsigned int> stateCount;

    for (unsigned int depth = 0; (depth < p.maxDepth) || (p.maxDepth == 0); depth += depth_count) {
      MCTS_OUTPUT("MCTS State: " << state << " " << "DEPTH: " << depth);
      if (terminal || ((maxPlanningTime > 0) && (getTime() > endPlanningTime)))
        break;
      MCTS_TIC(SELECT_PLANNING_ACTION);
      action = selectAction(discretizedState, true, action_idx, num_actions);
      MCTS_OUTPUT(" Action Selected: " << action);
      MCTS_TOC(SELECT_PLANNING_ACTION);
      MCTS_TIC(TAKE_ACTION);
      model->takeAction(state, action, reward, newState, terminal, depth_count);
      MCTS_OUTPUT("  Reward: " << reward);
      MCTS_TOC(TAKE_ACTION);
      modelUpdater->updateSimulationAction(action, newState);
      MCTS_TIC(VISIT);
      stateCount[state] += 1; // Should construct with 0 if unavailable
      history.push_back(HistoryStep(discretizedState, action_idx, num_actions, reward));
      MCTS_TOC(VISIT);
      state = newState;
      discretizedState = newState;
      stateMapping->map(discretizedState); // discretize state
    }

    if (terminal) {
      ++terminatedPlayouts;
    }

    MCTS_OUTPUT("------------ BACKPROP --------------");
    float backpropValue = 0;
    if (!terminal) {
      // Get bootstrap value of final state
      // http://stackoverflow.com/questions/11275444/c-template-typename-iterator
      typename StateInfoTable::const_accessor a;
      if (!stateInfoTable.find(a, state)) { // The state does not exist, choose an action randomly
        backpropValue = p.unknownBootstrapValue;
      } else {
        StateInfo stateInfo = a->second; // Create copy and release lock.
        a.release(); // TODO improper locking
        backpropValue = maxValueForState(state, stateInfo);
      }
    }
    MCTS_OUTPUT("At state: " << state << " the backprop value is " << backpropValue);

    typedef std::pair<State, StateInfo> StateToInfoPair;
    std::vector<StateToInfoPair> statesToAdd;

    for (int step = history.size() - 1; step >= 0; --step) {

      State &state = history[step].state;
      MCTS_OUTPUT("Reviewing state: " << state);
      unsigned int &action_idx = history[step].action_idx;
      unsigned int &num_actions = history[step].num_actions;
      float &reward = history[step].reward;

      backpropValue = reward + p.gamma * backpropValue;
      
      // Get information about this state
      typename StateInfoTable::accessor a;
      StateInfo stateInfo(num_actions, 0); 
      boost::shared_ptr<StateActionInfo> actionInfo;
      bool is_new_state = true;
      if (stateInfoTable.find(a, state)) { // The state exists, choose an action randomly and act
        MCTS_OUTPUT("  State: " << state << " found in table!");
        a->second.stateVisits++;
        if (!a->second.actionInfos[action_idx]) {
          a->second.actionInfos[action_idx].reset(new StateActionInfo(0, 0));
        } else {
          MCTS_OUTPUT("    Action " << action_idx << " found in table with value " << 
            a->second.actionInfos[action_idx]->val);
        }
        actionInfo = a->second.actionInfos[action_idx]; // Create copy and release lock
        stateInfo = a->second; // Create copy and release lock.
        is_new_state = false;
      } else { // Use the new state, and create a new action
        stateInfo.actionInfos[action_idx].reset(new StateActionInfo(0, 0));
        stateInfo.stateVisits++;
        actionInfo = stateInfo.actionInfos[action_idx];
      }
      a.release(); //TODO improper locking

      float thisStateValue = backpropValue;
      if (p.theoreticallyCorrectLambda) {
        if (stateInfo.stateVisits != 0) {
          backpropValue = p.lambda * backpropValue + (1.0 - p.lambda) * maxValueForState(state, stateInfo);
        } // else don't change the value being backed up
      }

      // Modify the action appropriately
      stateCount[state]--;
      if (stateCount[state] == 0) { // First Visit Monte Carlo
        actionInfo->visits++;
        actionInfo->val += 
          (1.0 / actionInfo->visits) * 
          (thisStateValue - actionInfo->val);
        if (is_new_state) {
          statesToAdd.push_back(typename StateInfoTable::value_type(state, stateInfo));
        }
        MCTS_OUTPUT("  Set value of action " << action_idx << " to " << actionInfo->val);
      }
      
      if (!p.theoreticallyCorrectLambda) {
        if (stateInfo.stateVisits != 0) {
          backpropValue = p.lambda * backpropValue + (1.0 - p.lambda) * maxValueForState(state, stateInfo);
        } // else don't change the value being backed up
      }

      MCTS_OUTPUT("  At state: " << state << " the backprop value is " << backpropValue);
    }

    MCTS_OUTPUT("------------ ADDING NEW STATES --------------");
    MCTS_OUTPUT("Num new states found in this rollout: " << statesToAdd.size()); 
    MCTS_OUTPUT("Current Table Size: " << stateInfoTable.size());
    /* Finally add the states */
    int numStatesAdded = 0;
    for (int s = statesToAdd.size() - 1; 
         s >= 0 && numStatesAdded < p.maxNewNodesPerRollout;
         --s, ++numStatesAdded) {
      stateInfoTable.insert(statesToAdd[s]);
    }
    MCTS_OUTPUT("Post Addition Table Size: " << stateInfoTable.size());

    MCTS_OUTPUT("State Table: " << std::endl << getStateTableDescription());
#ifdef MCTS_DEBUG
    if (++count == 10) throw std::runtime_error("argh!");
#endif
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
#ifdef MCTS_VALUE_DEBUG
  std::cout << getStateValuesDescription(state) << std::endl;
#endif
  return selectAction(mappedState, false, unused_action_idx, unused_num_actions);
}

template<class State, class StateHash, class Action>
Action MultiThreadedMCTS<State, StateHash, Action>::selectAction(const State &state, 
    bool usePlanningBounds, unsigned int& action_idx, unsigned int& num_actions) {

  std::vector<Action> stateActions;
  model->getAllActions(state, stateActions);

  typename StateInfoTable::const_accessor a;
  if (!stateInfoTable.find(a, state)) { // The state does not exist, choose an action randomly
    action_idx = rng->randomInt(stateActions.size()); 
    num_actions = stateActions.size();
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
  BOOST_FOREACH(const boost::shared_ptr<StateActionInfo>& stateActionInfo, stateInfo.actionInfos) {
    float val = calcActionValue(stateActionInfo, stateInfo, false);
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
  ss << prefix  << "MultiThreadedMCTS: " << std::endl;
  ss << prefix2 << p << std::endl;
  return ss.str();
}

template<class State, class StateHash, class Action>
std::string MultiThreadedMCTS<State, StateHash, Action>::getStateValuesDescription(const State& state) {
  std::stringstream ss;
  typename StateInfoTable::const_accessor a;
  if (!stateInfoTable.find(a, state)) { // The state does not exist, choose an action randomly
    ss << state << ": Not in table!";
    return ss.str();
  }
  StateInfo stateInfo = a->second; // Create copy and release lock. //TODO do we need to copy?
  a.release(); // TODO improper locking

  float maxVal = maxValueForState(state, stateInfo);
  ss << state << " " << maxVal << "(" << stateInfo.stateVisits << "): ";
  unsigned int count = 0;
  BOOST_FOREACH(const boost::shared_ptr<StateActionInfo>& actionInfo, stateInfo.actionInfos) {
    float val = calcActionValue(actionInfo, stateInfo, false);
    unsigned int na = 0;
    if (actionInfo) {
      na = actionInfo->visits;
    }
    ss << "  #" << count << " " << val << "(" << na << ")";
    if (count != stateInfo.actionInfos.size() - 1)
      ss << " "; 
    ++count;
  }

  return ss.str();
}

template<class State, class StateHash, class Action>
std::string MultiThreadedMCTS<State, StateHash, Action>::getStateTableDescription() {
  std::stringstream ss;
  int count = 0;
  // TODO sometimes this gives an extra state. not sure why.
  for (typename StateInfoTable::const_iterator v = stateInfoTable.begin();
      v != stateInfoTable.end(); ++v) {
    ss << "State #" << count << ": " << getStateValuesDescription(v->first) << std::endl; 
    ++count;
  }
  return ss.str();
}


#endif /* end of include guard: MULTI_THREADED_MCTS_H */
