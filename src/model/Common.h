#ifndef COMMON_B97S1VDF
#define COMMON_B97S1VDF

/*
File: Common.h
Author: Samuel Barrett
Description: some common information
Created:  2011-08-22
Modified: 2011-08-22
*/

#include <vector>
#include <ostream>
#include <boost/shared_ptr.hpp>
#include <common/Point2D.h>
#include <common/RNG.h>

namespace Action {
  enum Type { // order must match moves below
    LEFT,
    RIGHT,
    UP,
    DOWN,
    NOOP,
    NUM_MOVES,
    RANDOM,
    NUM_NEIGHBORS = NOOP,
    NUM_ACTIONS = NUM_MOVES
  };

  static Point2D MOVES[NUM_MOVES] = {Point2D(-1,0),Point2D(1,0),Point2D(0,1),Point2D(0,-1),Point2D(0,0)};
}

class ActionProbs {
public:
  ActionProbs();
  explicit ActionProbs(Action::Type ind);
  float& operator[](Action::Type ind);
  Action::Type selectAction(boost::shared_ptr<RNG> rng);

private:
  float probs[Action::NUM_MOVES];
};

Point2D wrapPoint(const Point2D &dims, Point2D pos);
Point2D movePosition(const Point2D &dims, Point2D pos, Action::Type action);
Point2D movePosition(const Point2D &dims, Point2D pos, const Point2D &move);
unsigned int getDistanceToPoint(const Point2D &dims, const Point2D &pos1, const Point2D &pos2);
Point2D getDifferenceToPoint(const Point2D &dims, const Point2D &start, const Point2D &end);

struct Observation {
  std::vector<Point2D> positions;
  int preyInd;
  unsigned int myInd;

  const Point2D& preyPos() const;
  const Point2D& myPos() const;
  int getCollision(const Point2D &pos) const;
};

std::ostream& operator<<(std::ostream &out, const Observation &obs) ;

#endif /* end of include guard: COMMON_B97S1VDF */
