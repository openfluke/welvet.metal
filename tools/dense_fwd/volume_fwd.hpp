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

// Dense NormalBP (training)
int run_dense_normalbp(int bench_iters);   // WDB1 → WDBO (+ optional WVB1)
int run_volume_normalbp(int bench_iters);  // WVN1 → WVNO (+ optional WVB1)

// RMSNorm NormalBP (training)
int run_rmsnorm_normalbp(int bench_iters);          // WRB1 → WRBO (+ optional WVB1)
int run_volume_rmsnorm_normalbp(int bench_iters);   // WRV1 → WRVO (+ optional WVB1)

// LayerNorm NormalBP (training)
int run_layernorm_normalbp(int bench_iters);          // WLB1 → WLBO (+ optional WVB1)
int run_volume_layernorm_normalbp(int bench_iters);   // WLV1 → WLVO (+ optional WVB1)

// SwiGLU NormalBP (training)
int run_swiglu_normalbp(int bench_iters);             // WSB1 → WSBO (+ optional WVB1)
int run_volume_swiglu_normalbp(int bench_iters);      // WSV1 → WSVO (+ optional WVB1)

// Softmax NormalBP (training)
int run_softmax_normalbp(int bench_iters);            // WXB1 → WXBO (+ optional WVB1)
int run_volume_softmax_normalbp(int bench_iters);     // WXV1 → WXVO (+ optional WVB1)

// Embedding NormalBP (training)
int run_embedding_normalbp(int bench_iters);          // WEB1 → WEBO (+ optional WVB1)
int run_volume_embedding_normalbp(int bench_iters);   // WEV1 → WEVO (+ optional WVB1)

// MHA NormalBP (training)
int run_mha_normalbp(int bench_iters);                // WHB1 → WHBO (+ optional WVB1)
int run_volume_mha_normalbp(int bench_iters);         // WHV1 → WHVO (+ optional WVB1)

// CNN1/2/3 NormalBP (training)
int run_cnn1_normalbp(int bench_iters);               // C1B1 → C1BO (+ optional WVB1)
int run_cnn2_normalbp(int bench_iters);               // C2B1 → C2BO (+ optional WVB1)
int run_cnn3_normalbp(int bench_iters);               // C3B1 → C3BO (+ optional WVB1)
int run_volume_cnn1_normalbp(int bench_iters);        // C1V1 → C1VO (+ optional WVB1)
int run_volume_cnn2_normalbp(int bench_iters);        // C2V1 → C2VO (+ optional WVB1)
int run_volume_cnn3_normalbp(int bench_iters);        // C3V1 → C3VO (+ optional WVB1)

// ConvT1/2/3 NormalBP (training)
int run_convt1_normalbp(int bench_iters);              // T1B1 → T1BO (+ optional WVB1)
int run_convt2_normalbp(int bench_iters);              // T2B1 → T2BO (+ optional WVB1)
int run_convt3_normalbp(int bench_iters);              // T3B1 → T3BO (+ optional WVB1)
int run_volume_convt1_normalbp(int bench_iters);       // T1V1 → T1VO (+ optional WVB1)
int run_volume_convt2_normalbp(int bench_iters);       // T2V1 → T2VO (+ optional WVB1)
int run_volume_convt3_normalbp(int bench_iters);       // T3V1 → T3VO (+ optional WVB1)

// RNN NormalBP (training)
int run_rnn_normalbp(int bench_iters);                 // RNB1 → RNBO (+ optional WVB1)
int run_volume_rnn_normalbp(int bench_iters);          // RNV1 → RNVO (+ optional WVB1)

// LSTM NormalBP (training)
int run_lstm_normalbp(int bench_iters);                // LSB1 → LSBO (+ optional WVB1)
int run_volume_lstm_normalbp(int bench_iters);         // LSV1 → LSVO (+ optional WVB1)

// Sequential NormalBP (training)
int run_sequential_normalbp(int bench_iters);          // SQB1 → SQBO (+ optional WVB1)
int run_volume_sequential_normalbp(int bench_iters);   // SQV1 → SQVO (+ optional WVB1)
