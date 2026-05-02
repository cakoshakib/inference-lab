package com.inferencelab.gpuworker.kv;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.inferencelab.gpuworker.gpu.config.GpuMemoryConfig;
import org.junit.jupiter.api.Test;

class ContiguousKvManagerTest {

  private static GpuMemoryConfig memoryConfig(long totalVramBytes) {
    return GpuMemoryConfig.builder()
        .totalVramBytes(totalVramBytes)
        .modelWeightBytes(100)
        .bytesPerToken(10)
        .maxTokensPerRequest(20)
        .build();
  }

  @Test
  void constructorThrowsWhenModelWeightsDoNotFit() {
    GpuMemoryConfig config =
        GpuMemoryConfig.builder()
            .totalVramBytes(50)
            .modelWeightBytes(100)
            .bytesPerToken(10)
            .maxTokensPerRequest(20)
            .build();

    assertThrows(RuntimeException.class, () -> new ContiguousKvManager(config));
  }

  @Test
  void tryAdmitSucceedsWhenSpaceAvailable() {
    ContiguousKvManager manager = new ContiguousKvManager(memoryConfig(10_000));

    assertTrue(manager.tryAdmit(1, 50));
  }

  @Test
  void remainingCapacityReflectsPromptBytes() {
    ContiguousKvManager manager = new ContiguousKvManager(memoryConfig(10_000));
    long spaceNeeded = 10 * 20;
    long promptBytes = 50;

    manager.tryAdmit(1, promptBytes);

    assertEquals(spaceNeeded - promptBytes, manager.remainingCapacity(1));
  }

  @Test
  void onTokenGeneratedReducesRemainingCapacity() {
    ContiguousKvManager manager = new ContiguousKvManager(memoryConfig(10_000));
    manager.tryAdmit(1, 50);
    long before = manager.remainingCapacity(1);

    manager.onTokenGenerated(1);

    assertEquals(before - 10, manager.remainingCapacity(1));
  }

  @Test
  void onTokenGeneratedThrowsForUnadmittedSequence() {
    ContiguousKvManager manager = new ContiguousKvManager(memoryConfig(10_000));

    assertThrows(IllegalArgumentException.class, () -> manager.onTokenGenerated(99));
  }

  @Test
  void remainingCapacityThrowsForUnadmittedSequence() {
    ContiguousKvManager manager = new ContiguousKvManager(memoryConfig(10_000));

    assertThrows(IllegalArgumentException.class, () -> manager.remainingCapacity(99));
  }

  @Test
  void freeRemovesSequence() {
    ContiguousKvManager manager = new ContiguousKvManager(memoryConfig(10_000));
    manager.tryAdmit(1, 50);

    manager.free(1);

    assertThrows(IllegalArgumentException.class, () -> manager.remainingCapacity(1));
  }

  @Test
  void freeThrowsForUnadmittedSequence() {
    ContiguousKvManager manager = new ContiguousKvManager(memoryConfig(10_000));

    assertThrows(IllegalArgumentException.class, () -> manager.free(99));
  }

  @Test
  void onTokenGeneratedThrowsWhenCapacityExceeded() {
    GpuMemoryConfig config =
        GpuMemoryConfig.builder()
            .totalVramBytes(10_000)
            .modelWeightBytes(100)
            .bytesPerToken(10)
            .maxTokensPerRequest(2)
            .build();
    ContiguousKvManager manager = new ContiguousKvManager(config);
    manager.tryAdmit(1, 0);

    manager.onTokenGenerated(1);
    manager.onTokenGenerated(1);

    assertThrows(RuntimeException.class, () -> manager.onTokenGenerated(1));
  }

  @Test
  void tryAdmitReturnsFalseWhenInsufficientSpace() {
    GpuMemoryConfig config =
        GpuMemoryConfig.builder()
            .totalVramBytes(150)
            .modelWeightBytes(100)
            .bytesPerToken(10)
            .maxTokensPerRequest(20)
            .build();
    ContiguousKvManager manager = new ContiguousKvManager(config);

    assertFalse(manager.tryAdmit(1, 0));
  }

  @Test
  void freeingSequenceAllowsNewAdmissionWhenFull() {
    GpuMemoryConfig config =
        GpuMemoryConfig.builder()
            .totalVramBytes(301)
            .modelWeightBytes(100)
            .bytesPerToken(10)
            .maxTokensPerRequest(20)
            .build();
    ContiguousKvManager manager = new ContiguousKvManager(config);

    assertTrue(manager.tryAdmit(1, 0));
    assertFalse(manager.tryAdmit(2, 0));

    manager.free(1);

    assertTrue(manager.tryAdmit(2, 0));
  }
}
