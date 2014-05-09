#ifndef MULTI_THREADED_MCTS_H
#define MULTI_THREADED_MCTS_H

#include <bwi_rl/common/Params.h>
#include <bwi_rl/planning/Model.h>
/* #include <bwi_rl/planning/MultiThreadedMCTSEstimator.h> */
#include <bwi_rl/planning/ModelUpdater.h>
#include <bwi_rl/planning/StateMapping.h>

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

class HistoryStep {
  public:
    HistoryStep(const State &state, const Action &action, float reward):
      state(state), action(action), reward(reward) {}
}

class StateActionInfo {
  public:
    StateActionInfo(unsigned int visits, float val):
      visits(visits),
      val(val) {}
    volatile unsigned int visits;
    volatile float val;
};

class StateInfo {
  public:
    StateInfo(unsigned int numActions, unsigned int initialVisits) : 
        stateVisits(initialVisits) {
      actionInfos.resize(numActions);
    }

    std::vector<boost::shared_ptr<StateActionInfo> > actionInfos;
    unsigned int stateVisits;
};

template<class State, class Action>
class MultiThreadedMCTS {
  public:
    typedef boost::shared_ptr<MultiThreadedMCTS<State,Action> > Ptr;
    /* typedef typename MultiThreadedMCTSEstimator<State,Action>::Ptr ValuePtr; */
    typedef typename ModelUpdater<State,Action>::Ptr ModelUpdaterPtr;
    typedef typename Model<State,Action>::Ptr ModelPtr;
    typedef typename StateMapping<State>::Ptr StateMappingPtr;
    typedef typename tbb::concurrent_hash_map<State, StateInfo, State::Hash> StateInfoTable;

#define PARAMS(_) \
    _(unsigned int,maxDepth,maxDepth,0) \
    _(int,numThreads,numThreads,1)

    Params_STRUCT(PARAMS)
#undef PARAMS

    MCTS (/*ValuePtr valueEstimator, */ModelUpdaterPtr modelUpdater,
        StateMappingPtr stateMapping, const Params &p);
    virtual ~MCTS () {}

    unsigned int search(const State &startState, unsigned int& termination_count);
    Action selectWorldAction(const State &state);
    void restart();
    std::string generateDescription(unsigned int indentation = 0);

  private:
    bool rollout(const State &startState);
    bool rollout(const State &startState, const Action &startAction);

  private:
    ModelUpdaterPtr modelUpdater;
    StateMappingPtr stateMapping;
    bool valid;

    unsigned int maxPlayouts;
    unsigned int currentPlayouts;
    unsigned int terminatedPlayouts;

    float endPlanningTime;

    Params p;

    StateInfoTable stateInfo;
};

template<class State, class Action>
MultiThreadedMCTS<State, Action>::MultiThreadedMCTS(
    /* ValuePtr valueEstimator, */
    ModelUpdaterPtr modelUpdater, StateMappingPtr stateMapping, 
    const Params &p) : // valueEstimator(valueEstimator),
    modelUpdater(modelUpdater), stateMapping(stateMapping), p(p) {}

template<class State, class Action>
unsigned int MultiThreadedMCTS<State, Action>::search(
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
        boost::thread(&MultiThreadedMCTS<State, Action>::singleThreadedSearch,
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

template<class State, class Action>
void MultiThreadedMCTS<State, Action>::singleThreadedSearch() {

  while ((maxPlanningTime <= 0.0) || (getTime() < endPlanningTime) && 
         (maxPlayouts <= 0) || (currentPlayouts < maxPlayouts)) {

    MCTS_OUTPUT("------------START ROLLOUT--------------");
    MCTS_TIC(SELECT_MODEL);
    ModelPtr model = modelUpdater->selectModel(startState);
    MCTS_TOC(SELECT_MODEL);
    State state(startState);
    State newState;
    Action action;
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
      action = selectAction(state, true);
      MCTS_OUTPUT("ACTION: " << action);
      MCTS_TOC(SELECT_PLANNING_ACTION);
      MCTS_TIC(TAKE_ACTION);
      model->takeAction(action, reward, newState, terminal, depth_count);
      MCTS_TOC(TAKE_ACTION);
      modelUpdater->updateSimulationAction(action, newState);
      MCTS_TIC(VISIT);
      stateCount[state] += 1; // Should construct with 0 if unavailable
      history.push_back(HistoryStep(state, action, reward);
      MCTS_TOC(VISIT);
      state = newState;
      stateMapping->map(state); // discretize state
    }

    MCTS_OUTPUT("------------ BACKPROP --------------");
    for (int step = history.size() - 1; step >= 0; --step) {
      

    }
    // TODO compute value backups, updating elements and decide which nodes to add
    // TODO add those nodes to the concurrent hash map
    // TODO ADD parameter to select how many nodes to add
  }
}

template<class State, class Action>
float MultiThreadedMCTS<State, Action>::calcActionValue(
    const boost::shared_ptr<StateActionInfo> &actionInfo, const StateInfo &state,
    bool usePlanningBounds) {
  if (!stateActionInfo) {
    if (usePlanningBounds) {
      return p.unknownActionPlanningValue;
    } else {
      return p.unknownActionValue;
    }
  }
  if (useBounds) {
    unsigned int na = stateActionInfo->visits;
    unsigned int n = stateInfo.stateVisits;
    if (na == 0) {
      // Shouldn't be here, but probably an artifact of thread overwriting
      return p.unknownActionPlanningValue;
    } else {
      return stateActionInfo->val + p.rewardBound * sqrt(log(n) / na);
    }
  } else {
    return stateActionInfo->val;
  }
}

template<class State, class Action>
Action MultiThreadedMCTS<State, Action>::selectWorldAction(const State &state) {
  State mappedState(state);
  stateMapping->map(mappedState); // discretize state
  return selectAction(mappedState, false);
}

template<class State, class Action>
Action MCTS<State,Action>::selectAction(const State &state, bool usePlanningBounds) {

  std::vector<Action> stateActions;
  this->model->getAllActions(stateActions);

  StateInfoTable::const_accessor a;
  if (!stateInfo.find(a, state)) { // The state does not exist, choose an action randomly and act
    return stateActions[rng->randomInt(stateActions.size())];
  }
  StateInfo stateInfo = a->second; // Create copy and release lock.
  a.release();

  int idx;
  float maxVal = -std::numeric_limits<float>::max();
  std::vector<unsigned int> maxActionIdx;
  unsigned int currrentActionIdx = 0;
  BOOST_FOREACH(const boost::shared_ptr<StateActionInfo>& stateActionInfo, stateInfo->actionInfos) {
    float val = calcActionValue(actionInfo, stateInfo, usePlanningBounds);
    if (fabs(val - maxVal) < 1e-10) {
      maxActionIdx.push_back(currentActionIdx);
    } else if (val > maxVal) {
      maxVal = val;
      maxActions.clear();
      maxActionIdx.push_back(currentActionIdx);
    }
    ++currentActionIdx;
  }

  return stateActions[maxActionIdx[rng->randomInt(maxActionIdx.size())]];
}

template<class State, class Action>
Action MultiThreadedMCTS<State, Action>::maxValueForState(const State &state) {

  StateInfoTable::const_accessor a;
  if (!stateInfo.find(a, state)) { // The state does not exist, choose an action randomly and act
    return p.unknownBootstrapValue;
  }
  StateInfo stateInfo = a->second; // Create copy and release lock.
  a.release();
  if (stateInfo.stateVisits == 0) {
    // Shouldn't be here, probably an artifact of threading
    return p.unknownBootstrapValue;
  }

  int idx;
  float maxVal = -std::numeric_limits<float>::max();
  BOOST_FOREACH(const boost::shared_ptr<StateActionInfo>& stateActionInfo, stateInfo->actionInfos) {
    float val = calcActionValue(actionInfo, stateInfo, usePlanningBounds);
    if (val > maxVal) {
      maxVal = val;
    }
  }

  return maxVal;
}

template<class State, class Action>
void MCTS<State,Action>::restart() {
  stateInfo->clear();
}

template<class State, class Action>
std::string MCTS<State,Action>::generateDescription(unsigned int indentation) {
  std::stringstream ss;
  std::string prefix = indent(indentation);
  std::string prefix2 = indent(indentation + 1);
  ss << prefix  << "MCTS" << std::endl;
  // TODO use the auto stringstream param function here
  ss << prefix2 << "Num playouts: " << p.maxPlayouts << "\n";
  ss << prefix2 << "Max planning time: " << p.maxPlanningTime << "\n";
  ss << prefix2 << "Max depth: " << p.maxDepth << "\n";
  ss << prefix2 << "Number of Threads: " << p.numThreads << "\n";
  return ss.str();
}

#endif /* end of include guard: MULTI_THREADED_MCTS_H */#
