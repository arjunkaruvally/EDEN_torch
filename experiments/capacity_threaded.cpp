#include <iostream>
#include "HopfieldNetwork.h"
#include "EDEN.h"
#include "cxxopts.hpp"
#include "progressbar.hpp"
#include <matplot/matplot.h>
#include <filesystem>
#include <torch/torch.h>

namespace fs = std::filesystem;


template<typename Scalar, typename Matrix>
inline static std::vector< std::vector<Scalar> > fromEigenMatrix( const Matrix & M ){
    std::vector< std::vector<Scalar> > m;
    m.resize(M.rows(), std::vector<Scalar>(M.cols(), 0));
    for(size_t i = 0; i < m.size(); i++)
        for(size_t j = 0; j < m.front().size(); j++)
            m[i][j] = M(i,j);
    return m;
}


inline static std::vector<std::vector<float>> convertTo2D(std::vector<float> &data){
    //convert to 2D
    std::vector<std::vector<float> > map_floor2D;
    map_floor2D.resize(10);
    for (int i = 0; i < 10; i++)
    {
        map_floor2D[i].resize(10);
    }
    for (int i = 0; i < data.size(); i++)
    {
        int row = i / 10;
        int col = i %10;
        map_floor2D[row][col] = data[i];
    }
    return map_floor2D;
}

inline static void diagnostic(EDEN &obj, int time_index){
    std::cout<<obj.v_trajectory[time_index]<<std::endl;
    auto vstate = obj.v_trajectory[time_index].contiguous();
    std::vector<float> data(vstate.data_ptr<float>(), vstate.data_ptr<float>()+vstate.numel());

    matplot::subplot(1, 4, 1);
    matplot::title("vstate");
    auto data2d = convertTo2D(data);
    matplot::heatmap(data2d);

    matplot::subplot(1, 4, 2);
    matplot::title("xi[0]");
    using namespace torch::indexing;
    vstate = obj.xi.index({Slice(0, None), 0}).contiguous();
    data = std::vector<float>(vstate.data_ptr<float>(), vstate.data_ptr<float>()+vstate.numel());
    data2d = convertTo2D(data);
    matplot::heatmap(data2d);

    matplot::subplot(1, 4, 3);
    matplot::title("xi[1]");
    vstate = obj.xi.index({Slice(0, None), 1}).contiguous();
    data = std::vector<float>(vstate.data_ptr<float>(), vstate.data_ptr<float>()+vstate.numel());
    data2d = convertTo2D(data);
    matplot::heatmap(data2d);

    matplot::subplot(1, 4, 4);
    matplot::title("xi[2]");
    vstate = obj.xi.index({Slice(0, None), 2}).contiguous();
    data = std::vector<float>(vstate.data_ptr<float>(), vstate.data_ptr<float>()+vstate.numel());
    data2d = convertTo2D(data);
    matplot::heatmap(data2d);

    matplot::show();
}


void spawn_workers(EDEN *obj, int threadId, int n_threads, int *success_counts, float epsilon, int capacity_type) {
    int workPerThread = ceil(1.0*obj->n / n_threads);
    std::cout<<"Work Per Thread: "<<workPerThread<<std::endl;

    for(int workerId=threadId*workPerThread; workerId < std::min(obj->n, (threadId+1)*workPerThread); workerId++) {
        switch (capacity_type) {
            case 1:
                EDEN::count_fixed_point_success_worker(workerId, *obj, *success_counts, epsilon);
                break;
            case 2:
                EDEN::count_single_step_success_worker(workerId, *obj, *success_counts, epsilon);
                break;
            default:
                std::cout << "Unknown capacity";
                break;
        }
    }
}


int get_success(int n_neurons, int n_sequences, int p_num_memories, float alpha_s, float alpha_c, float beta, float T_f, float T_d, float c,
                 float epsilon, uint64_t seed, uint64_t time_index, float tolerance, uint64_t n_threads, int capacity_type){

    n_threads = std::min((int)n_threads, n_sequences);
    std::vector<int> success_counts(n_threads, 0);
    EDEN *ptr = new EDEN(n_neurons, p_num_memories, n_sequences, alpha_s, alpha_c, beta, T_f, T_d, seed);
    std::thread threads[n_threads];

    // spawn threads
    for(int threadId=0; threadId < n_threads; threadId++){
        threads[threadId] = std::thread(spawn_workers, ptr, threadId, n_threads, &success_counts[threadId],
                                        epsilon, capacity_type);
    }

    // join threads
    for(int threadId=0; threadId < n_threads; threadId++){
        threads[threadId].join();
    }

    return std::accumulate(success_counts.begin(), success_counts.end(), 0);;
}


int main(int argc, char* argv[]) {
    cxxopts::Options options("EDENApp", "Program to configure and run the EDEN simulation");
    options.add_options()
            ("n_threads", "Number of threads", cxxopts::value<uint64_t>()->default_value("1"))
            ("capacity_type", "Type of capacity to evaluate. (1-fixed point, 2-single transition)", cxxopts::value<int>()->default_value("1"))
            ("n_neurons", "Number of neurons", cxxopts::value<int>()->default_value("100"))
            ("n_sequences", "Number of sequences", cxxopts::value<int>()->default_value("1"))
            ("pNumMemories", "Maximum number of memories to store", cxxopts::value<int>()->default_value("100"))
            ("alpha_s", "Alpha_s value", cxxopts::value<float>()->default_value("0.98"))
            ("tolerance", "error tolerance of the search algorithm", cxxopts::value<float>()->default_value("0.2"))
            ("alpha_c", "Alpha_c value", cxxopts::value<float>()->default_value("1"))
            ("beta", "beta of the softmax", cxxopts::value<float>()->default_value("20"))
            ("T_f", "T_f value", cxxopts::value<float>()->default_value("0.3"))
            ("T_d", "T_d value", cxxopts::value<float>()->default_value("20"))
            ("c", "c", cxxopts::value<float>()->default_value("0.1"))
            ("epsilon", "epsilon", cxxopts::value<float>()->default_value("0.1"))
            ("seed", "seed of the probabilistic event", cxxopts::value<uint64_t>()->default_value("0"))
            ("time_index", "Time index to plot", cxxopts::value<uint64_t>()->default_value("0"))
            ("path", "path to directory", cxxopts::value<std::string>()->default_value("tmp/EDEN"))
            ("experiment_id", "Experiment ID", cxxopts::value<std::string>()->default_value("0"))
            ("help", "Print usage");

    auto result = options.parse(argc, argv);

    if (result.count("help")) {
        std::cout << options.help() << std::endl;
        return 0;
    }

    int n_neurons = result["n_neurons"].as<int>();
    int n_sequences = result["n_sequences"].as<int>();
    int pNumMemories = result["pNumMemories"].as<int>();
    int capacity_type = result["capacity_type"].as<int>();
    float tol = result["tolerance"].as<float>();
    float alpha_s = result["alpha_s"].as<float>();
    float alpha_c = result["alpha_c"].as<float>();
    float beta = result["beta"].as<float>();

    float T_f = result["T_f"].as<float>();
    float T_d = result["T_d"].as<float>();
    float c = result["c"].as<float>();
    float epsilon = result["epsilon"].as<float>();
    uint64_t seed = result["seed"].as<uint64_t>();
    uint64_t n_threads = result["n_threads"].as<uint64_t>();
//    uint64_t n_threads = 1;
    auto path = result["path"].as<std::string>();
    auto EXPERIMENT_ID = result["experiment_id"].as<std::string>();;

    // Preliminary checks
    if(pNumMemories > pow(2, n_neurons)){
        std::cerr<<"The number of memories in sequence exceeds the maximum possible";
        return 0;
    }

    // START create directories for writing out results
    // first check if the environment variable for the experiment save directory is specified
    const std::string EXPERIMENT_OUTPUT_DIR = std::getenv("EXPERIMENT_OUTPUT_DIR");
    if(EXPERIMENT_OUTPUT_DIR.empty()) {
        std::cout<<"ERROR: EXPERIMENT_OUTPUT_DIR environment not set";
        return 0;
    }
    fs::path experiment_dir (EXPERIMENT_OUTPUT_DIR);
    fs::path save_path (path);
    fs::path output_path = experiment_dir / save_path;

    fs::create_directories(output_path);
    // END create directories

    std::cout<<"output_path: "<<output_path.string()<<std::endl;

    std::cout << "Computing Capacity with parameters" << std::endl;
    std::cout << "capacity type: "<< capacity_type << std::endl;
    std::cout << "Neurons: " << n_neurons << ", Number of memories: " << pNumMemories << ", Sequences: " << n_sequences << std::endl;
    std::cout << "Alpha_s: " << alpha_s << ", Alpha_c: " << alpha_c << ", beta: "<< beta;
    std::cout << ", T_f: " << T_f << ", T_d: " << T_d << ", seed: " << seed << std::endl;

    int num_success = get_success(n_neurons, n_sequences, pNumMemories, alpha_s, alpha_c, beta, T_f, T_d, c,
                                  epsilon, seed, result["time_index"].as<uint64_t>(), tol, n_threads, capacity_type);

    std::cout<<"Number of successes: "<<num_success<<std::endl;

    // write to file
    fs::path filename = std::string("result_")+EXPERIMENT_ID+std::string(".json");
    fs::path file_save_path = output_path / filename;

    std::ofstream result_file (file_save_path.string());
    if (result_file.is_open()){
        result_file << "{ \n";
        result_file << "\"capacity_type \": "<<capacity_type<<", \n";
        result_file << "\"seed\": "<<seed<<", \n";
        result_file << "\"c\": "<<c<<", \n";
        result_file << "\"epsilon\": "<<epsilon<<", \n";
        result_file << "\"beta\": "<<beta<<", \n";
        result_file << "\"n_neurons\": "<<n_neurons<<", \n";
        result_file << "\"num_memories\": "<<pNumMemories<<", \n";
        result_file << "\"num_events\": "<<n_neurons*pNumMemories<<", \n";
        result_file << "\"num_success\": "<<num_success<<"\n";
        result_file << "} \n";
        result_file.close();
    } else {
        std::cout<<"ERROR: unable to open file";
    }

    std::cout<<"File written to "<<file_save_path.string();
    // END write to file

    std::cout<<"Simulation Complete"<<std::endl;

    return 0;
}
