package com.inferencelab.gpuworker.kv;

public class VramRegion {

  private final long startByte;
  private final long endByte;
  private long usedBytes;

  public VramRegion(long startByte, long endByte) {
    this.startByte = startByte;
    this.endByte = endByte;
    this.usedBytes = 0;
  }

  public boolean useBytes(long numBytes) {
    if (availableCapacity() < numBytes) {
      return false;
    }
    usedBytes += numBytes;
    return true;
  }

  public long startByte() {
    return startByte;
  }

  public long endByte() {
    return endByte;
  }

  public long usedBytes() {
    return usedBytes;
  }

  public long availableCapacity() {
    return endByte - (startByte + usedBytes);
  }
}
