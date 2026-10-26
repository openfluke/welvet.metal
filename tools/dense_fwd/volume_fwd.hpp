#pragma once
#include "welvet/runtime/line.hpp"

enum class VolumeTrainMode { NormalBP, Tween, TweenChain, TweenSplit, TweenAlt, ProxyHead, ProxyLinear, ProxyFast, ProxyCache, ProxyAsync, ProxySparse };

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

// Dense NormalBP (training)
int run_dense_normalbp(int bench_iters);   // WDB1 → WDBO (+ optional WVB1)
int run_volume_normalbp(int bench_iters, VolumeTrainMode mode = VolumeTrainMode::NormalBP);  // WVN1 → WVNO (+ optional WVB1)

// RMSNorm NormalBP (training)
int run_rmsnorm_normalbp(int bench_iters);          // WRB1 → WRBO (+ optional WVB1)
int run_volume_rmsnorm_normalbp(int bench_iters, VolumeTrainMode mode = VolumeTrainMode::NormalBP);   // WRV1 → WRVO (+ optional WVB1)

// LayerNorm NormalBP (training)
int run_layernorm_normalbp(int bench_iters);          // WLB1 → WLBO (+ optional WVB1)
int run_volume_layernorm_normalbp(int bench_iters, VolumeTrainMode mode = VolumeTrainMode::NormalBP);   // WLV1 → WLVO (+ optional WVB1)

// SwiGLU NormalBP (training)
int run_swiglu_normalbp(int bench_iters);             // WSB1 → WSBO (+ optional WVB1)
int run_volume_swiglu_normalbp(int bench_iters, VolumeTrainMode mode = VolumeTrainMode::NormalBP);      // WSV1 → WSVO (+ optional WVB1)

// Softmax NormalBP (training)
int run_softmax_normalbp(int bench_iters);            // WXB1 → WXBO (+ optional WVB1)
int run_volume_softmax_normalbp(int bench_iters, VolumeTrainMode mode = VolumeTrainMode::NormalBP);     // WXV1 → WXVO (+ optional WVB1)

// Embedding NormalBP (training)
int run_embedding_normalbp(int bench_iters);          // WEB1 → WEBO (+ optional WVB1)
int run_volume_embedding_normalbp(int bench_iters, VolumeTrainMode mode = VolumeTrainMode::NormalBP);   // WEV1 → WEVO (+ optional WVB1)

// MHA NormalBP (training)
int run_mha_normalbp(int bench_iters);                // WHB1 → WHBO (+ optional WVB1)
int run_volume_mha_normalbp(int bench_iters, VolumeTrainMode mode = VolumeTrainMode::NormalBP);         // WHV1 → WHVO (+ optional WVB1)

// CNN1/2/3 NormalBP (training)
int run_cnn1_normalbp(int bench_iters);               // C1B1 → C1BO (+ optional WVB1)
int run_cnn2_normalbp(int bench_iters);               // C2B1 → C2BO (+ optional WVB1)
int run_cnn3_normalbp(int bench_iters);               // C3B1 → C3BO (+ optional WVB1)
int run_volume_cnn1_normalbp(int bench_iters, VolumeTrainMode mode = VolumeTrainMode::NormalBP);        // C1V1 → C1VO (+ optional WVB1)
int run_volume_cnn2_normalbp(int bench_iters, VolumeTrainMode mode = VolumeTrainMode::NormalBP);        // C2V1 → C2VO (+ optional WVB1)
int run_volume_cnn3_normalbp(int bench_iters, VolumeTrainMode mode = VolumeTrainMode::NormalBP);        // C3V1 → C3VO (+ optional WVB1)

// ConvT1/2/3 NormalBP (training)
int run_convt1_normalbp(int bench_iters);              // T1B1 → T1BO (+ optional WVB1)
int run_convt2_normalbp(int bench_iters);              // T2B1 → T2BO (+ optional WVB1)
int run_convt3_normalbp(int bench_iters);              // T3B1 → T3BO (+ optional WVB1)
int run_volume_convt1_normalbp(int bench_iters, VolumeTrainMode mode = VolumeTrainMode::NormalBP);       // T1V1 → T1VO (+ optional WVB1)
int run_volume_convt2_normalbp(int bench_iters, VolumeTrainMode mode = VolumeTrainMode::NormalBP);       // T2V1 → T2VO (+ optional WVB1)
int run_volume_convt3_normalbp(int bench_iters, VolumeTrainMode mode = VolumeTrainMode::NormalBP);       // T3V1 → T3VO (+ optional WVB1)

// RNN NormalBP (training)
int run_rnn_normalbp(int bench_iters);                 // RNB1 → RNBO (+ optional WVB1)
int run_volume_rnn_normalbp(int bench_iters, VolumeTrainMode mode = VolumeTrainMode::NormalBP);          // RNV1 → RNVO (+ optional WVB1)

// LSTM NormalBP (training)
int run_lstm_normalbp(int bench_iters);                // LSB1 → LSBO (+ optional WVB1)
int run_volume_lstm_normalbp(int bench_iters, VolumeTrainMode mode = VolumeTrainMode::NormalBP);         // LSV1 → LSVO (+ optional WVB1)

// Sequential NormalBP (training)
int run_sequential_normalbp(int bench_iters);          // SQB1 → SQBO (+ optional WVB1)
int run_volume_sequential_normalbp(int bench_iters, VolumeTrainMode mode = VolumeTrainMode::NormalBP);   // SQV1 → SQVO (+ optional WVB1)

// Residual NormalBP (training)
int run_residual_normalbp(int bench_iters);          // REB1 → REBO (+ optional WVB1)
int run_volume_residual_normalbp(int bench_iters, VolumeTrainMode mode = VolumeTrainMode::NormalBP);     // REV1 → REVO (+ optional WVB1)

// Mamba NormalBP (training)
int run_mamba_normalbp(int bench_iters);               // MBB1 → MBBO (+ optional WVB1)
int run_volume_mamba_normalbp(int bench_iters, VolumeTrainMode mode = VolumeTrainMode::NormalBP);        // MBV1 → MBVO (+ optional WVB1)

// Metacognition NormalBP (training)
int run_metacognition_normalbp(int bench_iters);       // MCB1 → MCBO (+ optional WVB1)
int run_volume_metacognition_normalbp(int bench_iters, VolumeTrainMode mode = VolumeTrainMode::NormalBP); // MCV1 → MCVO (+ optional WVB1)

// GDN NormalBP (training, f32)
int run_gdn_normalbp(int bench_iters);                 // GDB1 → GDBO (+ optional WVB1)
int run_volume_gdn_normalbp(int bench_iters, VolumeTrainMode mode = VolumeTrainMode::NormalBP);          // GDV1 → GDVO (+ optional WVB1)

// Stack NestedStack-of-Dense NormalBP (training)
int run_stack_normalbp(int bench_iters);               // STKB → STKO (+ optional WVB1)

// Line StepBP + MeshBP FamilyBP (training)
int run_stepbp(int bench_iters, welvet::runtime::line::Credit credit = welvet::runtime::line::Credit::FamilyBP, const char* response = "STBO");                       // STB1 → STBO (+ optional WVB1)
int run_meshbp(int bench_iters);                       // MBP1 → MBPO (+ optional WVB1)
int run_volume_stack_normalbp(int bench_iters, VolumeTrainMode mode = VolumeTrainMode::NormalBP);        // STKV → STVO (+ optional WVB1)

int run_tween_leaf(int bench_iters, VolumeTrainMode mode = VolumeTrainMode::Tween);

int run_meshtween(int bench_iters);
int run_meshtweenchain(int bench_iters);

int run_kmeans_normalbp(int);
int run_volume_kmeans_normalbp(int, VolumeTrainMode mode = VolumeTrainMode::NormalBP);
int run_parallel_normalbp(int);
int run_parallel_poly_train(int);

int run_volume_parallel_normalbp(int, VolumeTrainMode mode = VolumeTrainMode::NormalBP);

int run_normalbp_child(int);

int run_mesh_generic_bp(int bench_iters, bool tween = false, const char* out_magic = nullptr, int proxy_volume = 0, int mesh_family = 0);
