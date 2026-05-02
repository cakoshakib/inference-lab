package com.inferencelab.gpuworker.kv;

import com.inferencelab.gpuworker.gpu.config.GpuMemoryConfig;
import java.util.Map;
import java.util.Optional;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentSkipListMap;

public class ContiguousKvManager implements KvManager {

  private final GpuMemoryConfig memoryConfig;
  private Map<Long, VramRegion> allocatedRegions = new ConcurrentSkipListMap<>();
  private Map<Integer, VramRegion> sequenceToRegion = new ConcurrentHashMap<>();

  public ContiguousKvManager(GpuMemoryConfig memoryConfig) {
    this.memoryConfig = memoryConfig;
    // Allocate model weights region
    Optional<VramRegion> modelRegion = findAvailableRegion(memoryConfig.modelWeightBytes());
    if (modelRegion.isEmpty()) {
      throw new RuntimeException("Not enough memory to allocate to model weights");
    }
    modelRegion.get().useBytes(memoryConfig.modelWeightBytes());
    allocatedRegions.put(modelRegion.get().startByte(), modelRegion.get());
  }

  @Override
  public synchronized boolean tryAdmit(int sequenceId, long promptBytes) {
    long spaceNeeded = memoryConfig.bytesPerToken() * memoryConfig.maxTokensPerRequest();
    Optional<VramRegion> sequenceRegion = findAvailableRegion(spaceNeeded);
    // Could not find space to allocate
    if (sequenceRegion.isEmpty()) {
      return false;
    }
    allocatedRegions.put(sequenceRegion.get().startByte(), sequenceRegion.get());
    sequenceToRegion.put(sequenceId, sequenceRegion.get());
    sequenceRegion.get().useBytes(promptBytes);
    return true;
  }

  @Override
  public synchronized void onTokenGenerated(int sequenceId) {
    VramRegion vramRegion = getSequenceRegion(sequenceId);
    if (!vramRegion.useBytes(memoryConfig.bytesPerToken())) {
      throw new RuntimeException(
          String.format("Sequence %d exceeded available capacity for memory region", sequenceId));
    }
  }

  @Override
  public synchronized void free(int sequenceId) {
    VramRegion vramRegion = getSequenceRegion(sequenceId);
    allocatedRegions.remove(vramRegion.startByte());
    sequenceToRegion.remove(sequenceId);
  }

  @Override
  public long remainingCapacity(int sequenceId) {
    return getSequenceRegion(sequenceId).availableCapacity();
  }

  private Optional<VramRegion> findAvailableRegion(long minBytes) {
    if (allocatedRegions.isEmpty() && minBytes < memoryConfig.totalVramBytes()) {
      return Optional.of(new VramRegion(0, minBytes));
    }
    long prevEnd = 0;
    for (Map.Entry<Long, VramRegion> startAndRegion : allocatedRegions.entrySet()) {
      long openSpace = startAndRegion.getKey() - prevEnd;
      if (openSpace >= minBytes) {
        return Optional.of(new VramRegion(prevEnd, prevEnd + minBytes));
      }
      prevEnd = startAndRegion.getValue().endByte();
    }
    // Handle tail
    if (prevEnd + minBytes < memoryConfig.totalVramBytes()) {
      return Optional.of(new VramRegion(prevEnd, prevEnd + minBytes));
    }
    return Optional.empty();
  }

  private VramRegion getSequenceRegion(int sequenceId) {
    VramRegion vramRegion = sequenceToRegion.get(sequenceId);
    if (vramRegion == null) {
      throw new IllegalArgumentException(
          "Attempting to get region for sequence that has not been admitted");
    }
    return vramRegion;
  }
}
