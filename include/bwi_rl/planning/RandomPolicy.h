#ifndef BWI_RL_PLANNING_RANDOM_POLICY_H
#define BWI_RL_PLANNING_RANDOM_POLICY_H

#include <bwi_rl/planning/DefaultPolicy.h>

template<class State, class Action>
class RandomPolicy : public DefaultPolicy<State, Action> {

  public:
    typedef boost::shared_ptr<RandomPolicy<State, Action> > Ptr;

    virtual ~RandomPolicy () {}

    virtual int getBestAction(const State& state, 
                              const std::vector<Action> &actions, 
                              const boost::shared_ptr<RNG> &rng) {
      return rng->randomInt(actions.size() - 1);
    }

};

#endif /* end of include guard: BWI_RL_PLANNING_RANDOM_POLICY_H */
