////
//// Created by arjun on 6/14/24.
////

#ifndef EDEN_EDEN_H
#define EDEN_EDEN_H

#include <iostream>
#include <vector>
#include <torch/torch.h>

class EDEN {
public:
    bool VERBOSE;
    EDEN(int n_neurons, int n_memories, int n_sequences=1, float alpha_s=1.0, float alpha_c=1.0, float beta=2.0, float T_f=1.0, float T_d=20.0, uint64_t seed=42);
    float fraction_single_step_success(float epsilon);
    void reset_trajectories();
    std::vector<torch::Tensor> v_trajectory;
    int n;
    int d;
    int r;
    float alpha_s;
    float alpha_c;
    float beta;
    float T_f;
    float T_d;
    torch::Tensor v;
    torch::Tensor h;
    torch::Tensor s;

    torch::Tensor xi;
    torch::Tensor phixi;

    void update_s();
    void update_v();

    static void count_single_step_success_worker(int workerId, EDEN &ptr, int &num_success, float epsilon);
    static void count_fixed_point_success_worker(int workerId, EDEN &ptr, int &num_success, float epsilon);
    static void update_v_worker(EDEN &obj, torch::Tensor &v_state, torch::Tensor &s_state);
    static torch::Tensor int_to_bits(const torch::Tensor& x, int bits);

    float fraction_fixed_point_success(float epsilon);
};

#endif //EDEN_EDEN_H
