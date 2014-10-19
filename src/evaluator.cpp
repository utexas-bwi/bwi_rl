#include<fstream>
#include<cstdlib>

#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>
#include <boost/foreach.hpp>
#include <pluginlib/class_loader.h>

#include <bwi_rl/planning/domain.h>
#include <bwi_tools/common/Util.h>

using namespace bwi_rl;

Json::Value experiment_;
std::string experiment_file_;
std::string base_directory_ = ".";      // runtime directory.
int seed_ = 0;
int num_instances_ = 1;
int precompute_only_ = -1;

int processOptions(int argc, char** argv) {

  std::string mcts_params_file, methods_file;

  /** Define and parse the program options 
  */ 
  namespace po = boost::program_options; 
  po::options_description desc("Options"); 
  desc.add_options() 
    ("experiment-file", po::value<std::string>(&experiment_file_)->required(),
     "JSON file containing all the necessary information about this experiment.") 
    ("data-directory", po::value<std::string>(&base_directory_), "Data directory (defaults to runtime directory).") 
    ("seed", po::value<int>(&seed_), "Random seed (process number on condor)")  
    ("num-instances", po::value<int>(&num_instances_), "Number of Instances") 
    ("precompute-only", po::value<int>(&precompute_only_), "Run offline precomputation for each solver.");

  po::variables_map vm; 

  try { 
    po::store(po::command_line_parser(argc, argv).options(desc) 
              /* .positional(positionalOptions).allow_unregistered().run(),  */
              .allow_unregistered().run(), 
              vm); // throws on error 

    po::notify(vm); // throws on error, so do after help in case 
    // there are any problems 
  } catch(boost::program_options::required_option& e) { 
    std::cerr << "ERROR: " << e.what() << std::endl << std::endl; 
    std::cout << desc << std::endl;
    return -1; 
  } catch(boost::program_options::error& e) { 
    std::cerr << "ERROR: " << e.what() << std::endl << std::endl; 
    std::cout << desc << std::endl;
    return -1; 
  } 

  /* Read in methods */
  std::cout << "Experiment File: " << experiment_file_ << std::endl;
  if (!readJson(experiment_file_, experiment_)) {
    return -1;
  }

  /* Create the output directory */
  base_directory_ = base_directory_ + "/out";
  if (!boost::filesystem::is_directory(base_directory_) && !boost::filesystem::create_directory(base_directory_))
  {
    std::cerr << "Unable to create directory for storing intermediate results and output: " << base_directory_;
    return -1;
  }

  return 0;
}

int main(int argc, char** argv) {

  int ret = processOptions(argc, argv);
  if (ret != 0) {
    return ret;
  }

  // Load the domain using pluginlib.
  pluginlib::ClassLoader<Domain> class_loader("bwi_rl", "bwi_rl::Domain");
  std::vector<boost::shared_ptr<Domain> > domains;

  Json::Value domains_json = experiment_["domains"];
  try {
    for (unsigned domain_idx = 0; domain_idx < domains_json.size(); ++domain_idx) {
      std::string domain_name = domains_json[domain_idx]["domain"].asString();
      boost::shared_ptr<Domain> domain = class_loader.createInstance(domain_name);
      if (!(domain->initialize(domains_json[domain_idx], base_directory_))) {
        ROS_FATAL("Could not initialize domain.");
        return -1;
      }
      domains.push_back(domain);
    }
  } catch(pluginlib::PluginlibException& ex) {
    // Print an error should any solver fail to load.
    ROS_FATAL("Unable to load specified domain. Error: %s", ex.what());
    return -1;
  }

  // See if this is a precomputation request only.
  if (precompute_only_ != -1) {
    BOOST_FOREACH(boost::shared_ptr<Domain> &domain, domains) {
      for (int i = 0; i < num_instances_; ++i) {
        domain->precomputeAndSavePolicy(precompute_only_ + i);
      }
    }
    return 0;
  }

  // Otherwise let's start testing instances!
  BOOST_FOREACH(boost::shared_ptr<Domain> &domain, domains) {
    for (int i = 0; i < num_instances_; ++i) {
      domain->testInstance(seed_ + i);
    }
  }

  return 0;
}
