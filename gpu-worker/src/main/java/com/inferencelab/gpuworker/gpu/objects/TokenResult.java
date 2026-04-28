package com.inferencelab.gpuworker.gpu;

public record TokenResult(int sequenceId, String token, boolean isEos) {}
