package com.inferencelab.gpuworker.gpu.objects;

public record TokenResult(int sequenceId, String token, boolean isEos) {}
