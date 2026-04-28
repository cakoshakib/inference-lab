package com.inferencelab.gpuworker.gpu;

import java.util.List;

public record BatchResult(List<TokenResult> tokenResults, long latency) {}
