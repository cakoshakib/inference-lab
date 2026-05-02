package com.inferencelab.gpuworker.kv;

public interface KvManager {

  boolean tryAdmit(int sequenceId, long promptBytes);

  void onTokenGenerated(int sequenceId);

  void free(int sequenceId);

  long remainingCapacity(int sequenceId);
}
