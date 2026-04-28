package com.inferencelab.gpuworker.gpu;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.inferencelab.gpuworker.gpu.config.GpuComputeConfig;
import com.inferencelab.gpuworker.gpu.config.GpuConfig;
import com.inferencelab.gpuworker.gpu.config.GpuMemoryConfig;
import com.inferencelab.gpuworker.gpu.objects.ActiveSequence;
import com.inferencelab.gpuworker.gpu.objects.BatchResult;
import java.util.List;
import org.junit.jupiter.api.Test;

class MockGpuTest {

  private static GpuConfig defaultConfig() {
    return GpuConfig.builder()
        .memoryConfig(
            GpuMemoryConfig.builder()
                .totalVramBytes(1_000_000)
                .modelWeightBytes(100_000)
                .bytesPerToken(100)
                .maxTokensPerRequest(100)
                .build())
        .computeConfig(
            GpuComputeConfig.builder()
                .baseLatencyMs(5)
                .latencyPerTokenMs(1)
                .batchEfficiencyFactor(0.5)
                .maxBatchSize(4)
                .build())
        .eosProbability(0.0)
        .build();
  }

  @Test
  void stepReturnsTokenResultPerSequence() {
    MockGpu gpu = new MockGpu(defaultConfig());

    BatchResult result = gpu.step(List.of(new ActiveSequence(1, 0), new ActiveSequence(2, 0)));
    System.out.println("Token result: " + result.tokenResults().get(0).token());

    assertEquals(2, result.tokenResults().size());
    assertEquals(1, result.tokenResults().get(0).sequenceId());
    assertEquals(2, result.tokenResults().get(1).sequenceId());
  }

  @Test
  void stepThrowsWhenBatchExceedsMax() {
    MockGpu gpu = new MockGpu(defaultConfig());

    assertThrows(
        IllegalArgumentException.class,
        () ->
            gpu.step(
                List.of(
                    new ActiveSequence(1, 0),
                    new ActiveSequence(2, 0),
                    new ActiveSequence(3, 0),
                    new ActiveSequence(4, 0),
                    new ActiveSequence(5, 0))));
  }

  @Test
  void stepMarksEosWhenMaxTokensReached() {
    MockGpu gpu = new MockGpu(defaultConfig());

    BatchResult result = gpu.step(List.of(new ActiveSequence(1, 100)));

    assertTrue(result.tokenResults().get(0).isEos());
  }

  @Test
  void stepReportsLatency() {
    GpuConfig config =
        GpuConfig.builder()
            .memoryConfig(
                GpuMemoryConfig.builder()
                    .totalVramBytes(1_000_000)
                    .modelWeightBytes(100_000)
                    .bytesPerToken(100)
                    .maxTokensPerRequest(100)
                    .build())
            .computeConfig(
                GpuComputeConfig.builder()
                    .baseLatencyMs(5)
                    .latencyPerTokenMs(2)
                    .batchEfficiencyFactor(0.0)
                    .maxBatchSize(4)
                    .build())
            .eosProbability(0.0)
            .build();
    MockGpu gpu = new MockGpu(config);

    BatchResult result = gpu.step(List.of(new ActiveSequence(1, 0)));

    assertEquals(7, result.latency());
  }
}
