#pragma once

// Volume mount-once forwards (Metal). Bench → WVB2 (mount_ns + exec_ns + iters) + WVFO.
int run_volume_dense_fwd(int bench_iters);     // WVF1
int run_volume_mha_fwd(int bench_iters);       // WMV1
int run_volume_rmsnorm_fwd(int bench_iters);   // WRF1
int run_volume_layernorm_fwd(int bench_iters); // WLF1
int run_volume_softmax_fwd(int bench_iters);   // WSF1
int run_volume_parallel_fwd(int bench_iters);  // WPF1
int run_volume_kmeans_fwd(int bench_iters);    // WKF1
