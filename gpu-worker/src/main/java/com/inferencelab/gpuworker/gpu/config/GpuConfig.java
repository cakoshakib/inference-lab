package com.inferencelab.gpuworker.gpu.config;

import lombok.Builder;

@Builder
public record GpuConfig(
    GpuMemoryConfig memoryConfig, GpuComputeConfig computeConfig, double eosProbability) {
  public double stepLatency(int batchSize) {
    if (batchSize == 0) return 0;
    double parallelSavings = computeConfig.batchEfficiencyFactor() * (1 - 1.0 / batchSize);
    return computeConfig.baseLatencyMs()
        + batchSize * computeConfig.latencyPerTokenMs() * (1 - parallelSavings);
  }
}
