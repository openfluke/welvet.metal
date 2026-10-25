#pragma once

// Volume mount-once forwards (Metal). Bench → WVB2 (mount_ns + exec_ns + iters) + WVFO.
int run_volume_dense_fwd(int bench_iters);     // WVF1
int run_volume_mha_fwd(int bench_iters);       // WMV1
int run_volume_rmsnorm_fwd(int bench_iters);   // WRF1
int run_volume_layernorm_fwd(int bench_iters); // WLF1
int run_volume_softmax_fwd(int bench_iters);   // WSF1
int run_volume_parallel_fwd(int bench_iters);  // WPF1
int run_volume_kmeans_fwd(int bench_iters);    // WKF1
int run_volume_swiglu_fwd(int bench_iters);    // WSG1
int run_volume_sequential_fwd(int bench_iters);    // SQG1
int run_volume_residual_fwd(int bench_iters);      // REG1
int run_volume_rnn_fwd(int bench_iters);           // RNG1
int run_volume_lstm_fwd(int bench_iters);          // LSG1
int run_volume_cnn1_fwd(int bench_iters);          // C1G1
int run_volume_cnn2_fwd(int bench_iters);          // C2G1
int run_volume_cnn3_fwd(int bench_iters);          // C3G1
int run_volume_convt1_fwd(int bench_iters);        // T1G1
int run_volume_convt2_fwd(int bench_iters);        // T2G1
int run_volume_convt3_fwd(int bench_iters);        // T3G1
int run_volume_embedding_fwd(int bench_iters);     // WEG1
int run_volume_mamba_fwd(int bench_iters);         // MBG1
int run_volume_gdn_fwd(int bench_iters);           // GDG1
int run_volume_metacognition_fwd(int bench_iters); // MCG1
