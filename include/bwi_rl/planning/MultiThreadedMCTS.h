#ifndef MULTI_THREADED_MCTS_H
#define MULTI_THREADED_MCTS_H

#include <boost/foreach.hpp>
#include <boost/thread/thread.hpp>
#include <boost/tuple/tuple.hpp>

#include <bwi_tools/common/RNG.h>
#include <bwi_tools/common/Params.h>
#include <bwi_tools/common/Util.h>

#include <bwi_rl/planning/Model.h>
/* #include <bwi_rl/planning/MultiThreadedMCTSEstimator.h> */
#include <bwi_rl/planning/ModelUpdater.h>
#include <bwi_rl/planning/StateMapping.h>

/* #include <tbb/concurrent_hash_map.h> */
#include <tbb/concurrent_unordered_map.h>
/* #include <tbb/parallel_for.h> */

#ifdef MCTS_DEBUG
#define MCTS_OUTPUT(x) std::cout << x << std::endl
#else
#define MCTS_OUTPUT(x) ((void) 0)
#endif

class StateActionInfo {
  public:
    StateActionInfo(unsigned int visits = 0, float val = 0.0f) :
      visits(visits),
      val(val) {}
    unsigned int visits;
    float val;
};

class StateInfo {
  public:
    StateInfo(unsigned int num_actions = 0, unsigned int initial_visits = 0) : 
        action_infos(num_actions),
        state_visits(initial_visits) {}

    std::vector<StateActionInfo> action_infos;
    unsigned int state_visits;
};

template<class State, class StateHash, class Action>
class MultiThreadedMCTS {
  public:

    typedef boost::shared_ptr<MultiThreadedMCTS<State, StateHash, Action> > Ptr;
    /* typedef typename MultiThreadedMCTSEstimator<State,Action>::Ptr ValuePtr; */
    typedef typename ModelUpdater<State,Action>::Ptr ModelUpdaterPtr;
    typedef typename Model<State,Action>::Ptr ModelPtr;
    typedef typename StateMapping<State>::Ptr StateMappingPtr;
    typedef typename tbb::concurrent_unordered_map<State, StateInfo, StateHash> StateInfoTable;
    /* typedef typename std::map<State, StateInfo> StateInfoTable; */

    class HistoryStep {
      public:
        typename StateInfoTable::iterator state_info;
        unsigned int action_id;
        float reward;
        bool update_this_state;
    };

#define PARAMS(_) \
    _(unsigned int,maxDepth,maxDepth,0) \
    _(int,numThreads,numThreads,1) \
    _(float,lambda,lambda,0.0) \
    _(float,gamma,gamma,1.0) \
    _(float,rewardBound,rewardBound,10000) \
    _(float,maxNewStatesPerRollout,maxNewStatesPerRollout,0) \
    _(float,unknownActionValue,unknownActionValue,-1e10) \
    _(float,unknownActionPlanningValue,unknownActionPlanningValue,1e10) \
    _(float,unknownBootstrapValue,unknownBootstrapValue,0.0) \
    _(bool,theoreticallyCorrectLambda,theoreticallyCorrectLambda,false) \

    Params_STRUCT(PARAMS)
#undef PARAMS

    MultiThreadedMCTS (/*ValuePtr valueEstimator, */ModelUpdaterPtr modelUpdater,
        StateMappingPtr stateMapping, boost::shared_ptr<RNG> masterRng, const Params &p);
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

    float calcActionValue(const StateActionInfo &action_info,
                          const StateInfo &state_info, 
                          bool use_planning_bound) {
      if (action_info.visits == 0) {
        return (use_planning_bound) ? p.unknownActionPlanningValue : p.unknownActionValue;
      }
      float planning_bound = p.rewardBound * sqrtf(logf(state_info.state_visits) / action_info.visits);
      return (use_planning_bound) ? action_info.val + planning_bound : action_info.val;
    }

    float maxValueForState(const State &state, const StateInfo& state_info);
    Action selectAction(const State &state, bool use_planning_bound, 
        HistoryStep &step, unsigned int& new_states_added_in_rollout, boost::shared_ptr<RNG>& rng);

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
    boost::shared_ptr<RNG> masterRng;

    StateInfoTable stateInfoTable;
};

template<class State, class StateHash, class Action>
MultiThreadedMCTS<State, StateHash, Action>::MultiThreadedMCTS(
    /* ValuePtr valueEstimator, */
    ModelUpdaterPtr modelUpdater, StateMappingPtr stateMapping, 
    boost::shared_ptr<RNG> masterRng, const Params &p) : // valueEstimator(valueEstimator),
    modelUpdater(modelUpdater), stateMapping(stateMapping), masterRng(masterRng), p(p) {}

// TODO also allowing restriction of first action.
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

  std::vector<boost::shared_ptr<boost::thread> > threads; 
  // Launch n - 1 threads;
  for (int n = 1; n < p.numThreads; ++n) {
    boost::shared_ptr<boost::thread> thread(new
        boost::thread(&MultiThreadedMCTS<State, StateHash, Action>::singleThreadedSearch,
          this));
    threads.push_back(thread);
  }

  // Launch search in the main thread
  singleThreadedSearch();

  // Join all threads;
  for (int n = 1; n < p.numThreads; ++n) {
    threads[n - 1]->join();
  }

  termination_count = terminatedPlayouts;
  return currentPlayouts;
}

template<class State, class StateHash, class Action>
void MultiThreadedMCTS<State, StateHash, Action>::singleThreadedSearch() {

  boost::shared_ptr<RNG> rng(new RNG(masterRng->randomUInt()));

#ifdef MCTS_DEBUG
  int count = 0;
#endif

  std::vector<HistoryStep> history;
  if (p.maxDepth > 0)
  {
    history.resize(p.maxDepth);
  }

  while (((maxPlanningTime <= 0.0) || (getTime() < endPlanningTime)) && 
         ((maxPlayouts <= 0) || (currentPlayouts < maxPlayouts))) {

    // TODO not locked
    ++currentPlayouts;

    MCTS_OUTPUT("------------START ROLLOUT--------------");
    State state(startState), discretizedState(startState);
    State newState;
    Action action;
    unsigned int action_id, num_actions;
    bool terminal = false;
    int depth_count;

    stateMapping->map(discretizedState); // discretize state
    
    unsigned int history_size = 0;
    unsigned int new_states_added_in_rollout = 0;

    // Track how many times the same state is seen this rollout to do first visit monte carlo.
    std::map<State, unsigned int> stateCount;

    if (p.maxDepth <= 0) {
      // Version of the code with no max depth (and hence no pre-cached history memory).
      history.clear();
      for (unsigned int depth = 0; 
          (!terminal && (maxPlanningTime <= 0.0 || getTime() < endPlanningTime)); 
          depth += depth_count) {

        // Select action, take it and update the model with the action taken in simulation.
        MCTS_OUTPUT("MCTS State: " << state << " " << "DEPTH: " << depth);
        HistoryStep step;
        action = selectAction(discretizedState, true, step, new_states_added_in_rollout, rng);
        model->takeAction(state, action, step.reward, newState, terminal, depth_count, rng);
        MCTS_OUTPUT(" Action Selected: " << action);
        MCTS_OUTPUT("  Reward: " << step.reward);
        modelUpdater->updateSimulationAction(action, newState);

        // Record this step in history.
        history.push_back(step);

        // Update counters and states for next iteration.
        if (step.update_this_state) {
          ++(stateCount[discretizedState]); // Should construct with 0 if unavailable
        }
        state = newState;
        discretizedState = newState;
        stateMapping->map(discretizedState);
      }
      history_size = history.size();
    } else {
      // Version of the code with max depth (and precached memory for history).
      for (unsigned int depth = 0; 
          (!terminal && (depth < p.maxDepth) && (maxPlanningTime <= 0.0 || getTime() < endPlanningTime)); 
          depth += depth_count) {

        // Select action, take it and update the model with the action taken in simulation.
        MCTS_OUTPUT("MCTS State: " << state << " " << "DEPTH: " << depth);
        HistoryStep &step = history[history_size];
        action = selectAction(discretizedState, true, step, new_states_added_in_rollout, rng);
        model->takeAction(state, action, step.reward, newState, terminal, depth_count, rng);
        MCTS_OUTPUT(" Action Selected: " << action);
        MCTS_OUTPUT("  Reward: " << step.reward);
        modelUpdater->updateSimulationAction(action, newState);

        // Record this step in history.
        ++history_size;

        // Update counters and states for next iteration.
        if (step.update_this_state) {
          ++(stateCount[discretizedState]); // Should construct with 0 if unavailable
        }
        state = newState;
        discretizedState = newState;
        stateMapping->map(discretizedState);
      }

    }

    if (terminal) {
      ++terminatedPlayouts; // TODO not locked
    }

    MCTS_OUTPUT("------------ BACKPROP --------------");
    float backpropValue = 0;
    if (!terminal) {
      // Use the state info iterator in the history step to access the value.
      // Get bootstrap value of final state
      // Use of typename: http://stackoverflow.com/questions/11275444/c-template-typename-iterator
      typename StateInfoTable::const_iterator final_state_info = stateInfoTable.find(discretizedState);
      if ((final_state_info == stateInfoTable.end()) || (final_state_info->second.state_visits == 0)) { 
        backpropValue = p.unknownBootstrapValue;
      } else {
        backpropValue = maxValueForState(state, final_state_info->second);
      }
    }
    MCTS_OUTPUT("At final discretized state: " << discretizedState << " the backprop value is " << backpropValue);

    for (int step = history_size - 1; step >= 0; --step) {

      // Get information about this state
      typename StateInfoTable::iterator &state_info = history[step].state_info;
      unsigned int &action_id = history[step].action_id;
      float &reward = history[step].reward;

      MCTS_OUTPUT("Reviewing state: " << state_info->first << " with reward: " << reward);

      backpropValue = reward + p.gamma * backpropValue;

      MCTS_OUTPUT("Total backprop value: " << backpropValue);

      if (history[step].update_this_state) {

        stateCount[state_info->first]--;
        // Modify the action appropriately
        if (stateCount[state_info->first] == 0) { // First Visit Monte Carlo
          ++(state_info->second.state_visits);
          StateActionInfo &action_info = state_info->second.action_infos[action_id];
          (action_info.visits)++;
          action_info.val += (1.0 / action_info.visits) * (backpropValue - action_info.val);
          MCTS_OUTPUT("  Set value of action " << action_id << " to " << action_info.val);
        }
        
        if (state_info->second.state_visits > 1) {
          float maxValue = maxValueForState(state_info->first, state_info->second);
          MCTS_OUTPUT("  Interpolating backpropagation value between current " << backpropValue << " and max " << maxValue);
          backpropValue = p.lambda * backpropValue + (1.0 - p.lambda) * maxValue;
        } // else don't change the value being backed up
      }

      MCTS_OUTPUT("  At state: " << state_info->first << " the backprop value is " << backpropValue);
    }

    MCTS_OUTPUT("State Table: " << std::endl << getStateTableDescription());
#ifdef MCTS_DEBUG
    if (++count == 10) throw std::runtime_error("argh!");
#endif
  }
}

template<class State, class StateHash, class Action>
Action MultiThreadedMCTS<State, StateHash, Action>::selectWorldAction(const State &state) {
  State mappedState(state);
  stateMapping->map(mappedState); // discretize state
#ifdef MCTS_VALUE_DEBUG
  std::cout << "    " << getStateValuesDescription(state) << std::endl;
#endif
  boost::shared_ptr<RNG> rng(new RNG(masterRng->randomUInt()));
  HistoryStep unused_step;
  unsigned int unused_new_states_counter = 0;
  return selectAction(mappedState, false, unused_step, unused_new_states_counter, rng);
}

template<class State, class StateHash, class Action>
Action MultiThreadedMCTS<State, StateHash, Action>::selectAction(const State &state, 
    bool use_planning_bound, HistoryStep& history_step, unsigned int& new_states_added_in_rollout,
    boost::shared_ptr<RNG>& rng) {

  // Get all the actions available at this state.
  std::vector<Action> stateActions;
  model->getAllActions(state, stateActions);

  // There are three cases here
  //  1. The state information exists, and we should use UCT action selection to choose.
  //  2. The state does not exist in the state info table and needs to be added.
  //  3. The state does not exist in the state info table and does not need to be added.

  history_step.state_info = stateInfoTable.find(state);
  if (history_step.state_info != stateInfoTable.end()) {
    // It may be possible that we hit this state for the first time in this rollout itself.
    // Use the default policy if state visits are zero (i.e. we have not back-propogated yet).
    if (history_step.state_info->second.state_visits != 0) { 
      float maxVal = -std::numeric_limits<float>::max();
      std::vector<unsigned int> maxActionIdx;
      unsigned int currentActionIdx = 0;
      BOOST_FOREACH(const StateActionInfo &action_info, history_step.state_info->second.action_infos) {
        float val = calcActionValue(action_info, history_step.state_info->second, use_planning_bound);
        if (fabs(val - maxVal) < 1e-10) {
          maxActionIdx.push_back(currentActionIdx);
        } else if (val > maxVal) {
          maxVal = val;
          maxActionIdx.clear();
          maxActionIdx.push_back(currentActionIdx);
        }
        ++currentActionIdx;
      }
      history_step.action_id = maxActionIdx[rng->randomInt(maxActionIdx.size())];
    } else {
      // TODO switch this to default policy class.
      history_step.action_id = rng->randomInt(stateActions.size()); 
    }
    history_step.update_this_state = true;
  } else if ((new_states_added_in_rollout < p.maxNewStatesPerRollout) || (p.maxNewStatesPerRollout == 0)) {
    // Add the state + state-action, and choose an action using the default policy.
    StateInfo new_state_info(stateActions.size(), 0); 
    bool unused_bool;
    boost::tie(history_step.state_info, unused_bool) = 
      stateInfoTable.insert(std::pair<State, StateInfo>(state, new_state_info));
    // TODO switch this to default policy class.
    history_step.action_id = rng->randomInt(stateActions.size()); 
    history_step.update_this_state = true;
    ++new_states_added_in_rollout;
  } else {
    // Just store the history step since it's not necessary to add the state-action pair.
    // TODO switch this to default policy class.
    history_step.action_id = rng->randomInt(stateActions.size()); 
    history_step.update_this_state = false;
  }

  return stateActions[history_step.action_id];
}

template<class State, class StateHash, class Action>
float MultiThreadedMCTS<State, StateHash, Action>::maxValueForState(const State &state,
    const StateInfo& state_info) {

  int idx;
  float maxVal = -std::numeric_limits<float>::max();
  BOOST_FOREACH(const StateActionInfo &stateActionInfo, state_info.action_infos) {
    float val = calcActionValue(stateActionInfo, state_info, false);
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
  typename StateInfoTable::const_iterator a = stateInfoTable.find(state);
  if (a == stateInfoTable.end()) { // The state does not exist, choose an action randomly
    ss << state << ": Not in table!";
    return ss.str();
  }

  float maxVal = maxValueForState(state, a->second);
  ss << state << " " << maxVal << "(" << a->second.state_visits << "): ";
  unsigned int count = 0;
  BOOST_FOREACH(const StateActionInfo &action_info, a->second.action_infos) {
    float val = calcActionValue(action_info, a->second, false);
    unsigned int na = action_info.visits;
    ss << " #" << count << " " << val << "(" << na << ")";
    if (count != a->second.action_infos.size() - 1)
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
