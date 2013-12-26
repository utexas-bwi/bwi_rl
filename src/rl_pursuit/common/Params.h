#ifndef PARAMS_2128H8YJ
#define PARAMS_2128H8YJ

#include <rl_pursuit/json/json.h>

#define SET_FROM_JSON(type,jsonCmd) \
  inline void setFromJson(const Json::Value &opts, const char *key, type &val) { \
    const Json::Value &opt = opts[key]; \
    if (!opt.isNull()) \
      val = opt.jsonCmd(); \
  }

#define SET_FROM_JSON_ENUM(ENUM) \
  inline void setFromJson(const Json::Value &opts, const char *key, ENUM::type &val) { \
    const Json::Value &opt = opts[key]; \
    if (!opt.isNull()) \
      val = ENUM::fromName(opt.asCString()); \
  }

SET_FROM_JSON(bool,asBool)
SET_FROM_JSON(int,asInt)
SET_FROM_JSON(unsigned int,asUInt)
SET_FROM_JSON(double,asDouble)
SET_FROM_JSON(float,asDouble)
SET_FROM_JSON(std::string,asString)

#define PARAM_DECL(type,var,key,val) type var;
#define PARAM_INIT(type,var,key,val) var = val;
#define PARAM_SET(type,var,key,val) setFromJson(opts,#key,var);

#define Params_STRUCT(params) \
  struct Params {\
    params(PARAM_DECL) \
    \
    Params() { \
      params(PARAM_INIT) \
    } \
    \
    void fromJson(const Json::Value &opts) { \
      (void)opts; /* to remove any compiler warnings if params is empty */ \
      params(PARAM_SET) \
    } \
  };

#endif /* end of include guard: PARAMS_2128H8YJ */