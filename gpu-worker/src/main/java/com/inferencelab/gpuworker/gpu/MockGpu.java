package com.inferencelab.gpuworker.gpu;

import com.inferencelab.gpuworker.gpu.config.GpuConfig;
import java.util.ArrayList;
import java.util.List;

public class MockGpu {

  private GpuConfig config;

  public MockGpu(GpuConfig config) {
    this.config = config;
  }

  public BatchResult step(List<ActiveSequence> batch) {
    if (batch.size() > config.computeConfig().maxBatchSize()) {
      throw new IllegalArgumentException(
          String.format(
              "Batch size %d exceeds max batch size of %d",
              batch.size(), config.computeConfig().maxBatchSize()));
    }

    // Simulate processing time based on batch size and compute config
    long stepLatency = (long) config.stepLatency(batch.size());

    try {
      Thread.sleep(stepLatency);
    } catch (InterruptedException e) {
      Thread.currentThread().interrupt();
    }

    List<TokenResult> tokenResults = new ArrayList<>();
    for (ActiveSequence seq : batch) {
      TokenResult tokenResult = processSequence(seq);
      tokenResults.add(tokenResult);
    }

    return new BatchResult(tokenResults, stepLatency);
  }

  private TokenResult processSequence(ActiveSequence seq) {
    String token = WordGenerator.getWord();
    double progress = (double) seq.tokensGenerated() / config.memoryConfig().maxTokensPerRequest();
    double eosProbability = config.eosProbability() + (progress * (1 - config.eosProbability()));
    boolean isEos =
        Math.random() < eosProbability
            || seq.tokensGenerated() >= config.memoryConfig().maxTokensPerRequest();
    return new TokenResult(seq.sequenceId(), token, isEos);
  }
}
