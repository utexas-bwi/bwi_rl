#ifndef BWI_RL_PLANNING_DEFAULT_POLICY_H
#define BWI_RL_PLANNING_DEFAULT_POLICY_H

#include <boost/shared_ptr.hpp>
#include <bwi_tools/common/RNG.h>

template<class State, class Action>
class DefaultPolicy {

  public:
    typedef boost::shared_ptr<DefaultPolicy<State, Action> > Ptr;

    virtual ~DefaultPolicy () {}

    virtual int getBestAction(const State& state, 
                              const std::vector<Action> &actions, 
                              const boost::shared_ptr<RNG> &rng) = 0;

};

#endif /* end of include guard: BWI_RL_PLANNING_DEFAULT_POLICY_H */
