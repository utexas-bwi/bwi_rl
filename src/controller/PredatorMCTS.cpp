#include "PredatorMCTS.h"

PredatorMCTS::PredatorMCTS(boost::shared_ptr<RNG> rng, const Point2D &dims, boost::shared_ptr<MCTS<Observation,Action::Type> > planner):
  Agent(rng,dims),
  planner(planner)
{}

Action::Type PredatorMCTS::step(const Observation &obs) {
  planner->search(obs);
  return planner->selectWorldAction(obs);
}

void PredatorMCTS::restart() {
  planner->restart();
}

std::string PredatorMCTS::generateDescription() {
  return "PredatorMCTS: a predator using MCTS to select actions";
}
