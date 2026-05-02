package com.inferencelab.gpuworker.gpu.objects;

import java.util.List;

public record BatchResult(List<TokenResult> tokenResults, long latency) {}
