////
//// Created by arjun on 6/14/24.
////

#include "EDEN.h"

torch::Tensor EDEN::int_to_bits(const torch::Tensor& x, int bits){
    torch::Tensor mask = torch::pow(2, torch::arange(bits-1, -1, -1));
    return x.unsqueeze(-1).bitwise_and(mask).ne(0).to(torch::kFloat);
}

EDEN::EDEN(int n_neurons, int n_memories, int n_sequences, float alpha_s, float alpha_c, float beta, float T_f, float T_d, uint64_t seed){
    // initialize model parameters
    this->d = n_neurons;
    this->n = n_memories;
    this->r = n_sequences;
    this->alpha_s = alpha_s;
    this->alpha_c = alpha_c;
    this->beta = beta;
    this->T_f = T_f;
    this->T_d = T_d;
    this->VERBOSE = false;
    torch::manual_seed(seed);

    // initialize the state variables
    auto opts = torch::TensorOptions().dtype(torch::kFloat);
    this->v = at::zeros({this->d, 1}, opts);
    this->s = at::zeros({this->d, 1}, opts);
    this->h = at::zeros({this->n, 1}, opts);

    torch::Tensor probs = torch::ones({(int)pow(2, this->d)});
    this->xi = int_to_bits(at::multinomial(probs,
                                           this->n,
                                           false).to(torch::kInt),
                           this->d).transpose(0, 1);
    this->xi = this->xi*2 - 1;
//    std::cout<<"xi: "<<this->xi<<std::endl;

//    // define the interaction matrix xi
//    torch::Tensor probs = at::full(2, 0.5);
//    probs.index_put_({0}, 0.5);
//    probs.index_put_({0}, 0.5);
//    this->xi = (at::multinomial(probs, this->d*this->n,
//                                true)*2-1).to(torch::kFloat).reshape({this->d,
//                                                                      this->n});

    // define the phixi matrix TODO
    this->phixi = this->xi.clone();
    this->phixi = at::roll(this->phixi, 1, r);
}

void EDEN::reset_trajectories(){
    this->v_trajectory.clear();
}

void EDEN::update_v() {
    this->h = this->alpha_s * torch::matmul(this->xi.transpose(0, 1), this->v) +
            this->alpha_c * torch::matmul(this->phixi.transpose(0, 1), this->s);

    if(this->VERBOSE){
        this->v_trajectory.push_back(this->v.clone());
    }

    this->v = this->v + (0.01/this->T_f) * (torch::matmul(this->xi, torch::softmax((this->beta/(1.0*this->d))*this->h, 0)) - this->v);
}

void EDEN::update_s() {
    this->s = this->s + (0.01/this->T_d) * (this->v - this->s);
}

float EDEN::fraction_single_step_success(float epsilon){
    using namespace torch::indexing;
    int count_success = 0;
    auto score = torch::zeros({this->d, 1});
    for(int i=0; i<this->n; i++){
        this->s = this->xi.index({Slice(0, None), i}).clone();
        this->v = this->s.clone();
        this->s = 1.2*this->s;  // initialize to increase by a fraction to induce transition

        for(int time=0; time<100; time++){
            this->update_v();
        }

        // count the number of success
        int target_i = (i + this->r) % this->n;
        score = this->v * this->xi.index({Slice(None, None), target_i});
//        errors = torch::abs(errors);
        count_success += (score >= 1-epsilon).sum().item<int>();
        if(this->VERBOSE){
            std::cout<<"score: "<<score<<std::endl;
        }
    }
    if(this->VERBOSE){
        std::cout<<count_success<<"/"<<this->n*this->d<<std::endl;
    }
    return (1.0*count_success) / (this->n*this->d);
}

float EDEN::fraction_fixed_point_success(float epsilon){
    using namespace torch::indexing;
    int count_success = 0;
    auto score = torch::zeros({this->d, 1});
    for(int i=0; i<this->n; i++){
        this->s = this->phixi.index({Slice(0, None), i}).clone();
        this->v = this->xi.index({Slice(0, None), i}).clone();
        this->s = 1.2*this->s;  // initialize to increase by a fraction to induce transition

        for(int time=0; time<100; time++){
            this->update_v();
        }

        // count the number of success
        score = this->v * this->xi.index({Slice(0, None), i});
//        errors = torch::abs(errors);
        count_success += (score >= 1-epsilon).sum().item<int>();
        if(this->VERBOSE){
            std::cout<<"score: "<<i<<" "<<score.sizes()<<" "<<(score >= 1-epsilon).sum().item<int>()<<" "<<score<<std::endl;
        }
    }
    if(this->VERBOSE){
        std::cout<<count_success<<"/"<<this->n*this->d<<std::endl;
    }
    return (1.0*count_success) / (this->n*this->d);
}

// implementation for multithreading
void EDEN::update_v_worker(EDEN &obj, torch::Tensor &v_state, torch::Tensor &s_state) {
    torch::Tensor h_state = obj.alpha_s * torch::matmul(obj.xi.transpose(0, 1), v_state) +
                            obj.alpha_c * torch::matmul(obj.phixi.transpose(0, 1), s_state);
    v_state = v_state + (0.01/obj.T_f) * (torch::matmul(obj.xi,
                                                         torch::softmax((obj.beta/(1.0*obj.d))*h_state,
                                                                        0)) - v_state);
}

void EDEN::count_single_step_success_worker(int workerId, EDEN &ptr, int &num_success, float epsilon) {
    using namespace torch::indexing;
    auto score = torch::zeros({ptr.d, 1});

    torch::Tensor vstate = ptr.xi.index({Slice(0, None), workerId}).clone();
    torch::Tensor sstate = 1.2*vstate.clone();

    for(int time=0; time<100; time++){
        EDEN::update_v_worker(ptr, vstate, sstate);
    }
    int target_i = (workerId + ptr.r) % ptr.n;

    score = vstate * ptr.xi.index({Slice(None, None), target_i});
    num_success += (score >= 1-epsilon).sum().item<int>();
}

void EDEN::count_fixed_point_success_worker(int workerId, EDEN &obj, int &num_success, float epsilon) {
    using namespace torch::indexing;
    auto score = torch::zeros({obj.d, 1});

    torch::Tensor sstate = 1.2*obj.phixi.index({Slice(0, None), workerId}).clone();
    torch::Tensor vstate = obj.xi.index({Slice(0, None), workerId}).clone();
    sstate = 1.2*sstate;

    // check fixed point near a memory
    torch::Tensor h_state = obj.alpha_s * torch::matmul(obj.xi.transpose(0, 1), vstate) +
                            obj.alpha_c * torch::matmul(obj.phixi.transpose(0, 1), sstate);
    torch::Tensor fpointScore = vstate * torch::matmul(obj.xi,
                                               torch::softmax(obj.beta*h_state, 0));
    num_success += (torch::abs(fpointScore - 1) <= epsilon).sum().item<int>();
}
