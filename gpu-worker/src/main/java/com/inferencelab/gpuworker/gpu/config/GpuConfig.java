package com.inferencelab.gpuworker.gpu.config;

import lombok.Builder;

@Builder
public record GpuConfig(GpuMemoryConfig memoryConfig, GpuComputeConfig computeConfig) {}
