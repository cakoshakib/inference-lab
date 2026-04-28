package com.inferencelab.gpuworker.gpu.config;

import lombok.Builder;

@Builder
public record GpuComputeConfig(
    int baseLatencyMs, int latencyPerTokenMs, double batchEfficiencyFactor, int maxBatchSize) {}
