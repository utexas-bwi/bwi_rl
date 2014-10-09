#ifndef BWI_RL_PLANNING_DOMAIN_H
#define BWI_RL_PLANNING_DOMAIN_H

#include <bwi_tools/json/json.h> 

namespace bwi_rl {

  class Domain {

    public:

      virtual bool initialize(Json::Value &experiment, const std::string &base_directory) = 0;
      virtual void precomputeAndSavePolicy(int problem_identifier) = 0;
      virtual void testInstance(int seed) = 0;
      virtual ~Domain() {}

    protected:

      Domain() {}

  };

} /* bwi_rl */

#endif /* end of include guard: BWI_RL_PLANNING_DOMAIN_H */
