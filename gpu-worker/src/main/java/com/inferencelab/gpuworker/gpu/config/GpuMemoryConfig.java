package com.inferencelab.gpuworker.gpu.config;

import lombok.Builder;

@Builder
public record GpuMemoryConfig(
    long totalVramBytes, long modelWeightBytes, long bytesPerToken, long maxTokensPerRequest) {}
