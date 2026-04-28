package com.inferencelab.gpuworker;

import com.inferencelab.gpuworker.api.InferenceController;
import com.inferencelab.gpuworker.gpu.config.GpuComputeConfig;
import com.inferencelab.gpuworker.gpu.config.GpuConfig;
import com.inferencelab.gpuworker.gpu.config.GpuMemoryConfig;
import io.javalin.Javalin;

public class GpuWorker {
  public static void main(String[] args) {
    GpuConfig gpuConfig =
        GpuConfig.builder()
            .memoryConfig(
                GpuMemoryConfig.builder()
                    .totalVramBytes(24 * 1024 * 1024 * 1024L) // 24 GB
                    .modelWeightBytes(10 * 1024 * 1024 * 1024L) // 10 GB
                    .bytesPerToken(10 * 1024 * 1024L) // 10 MB per token
                    .maxTokensPerRequest(2048)
                    .build())
            .computeConfig(
                GpuComputeConfig.builder()
                    .baseLatencyMs(50)
                    .latencyPerTokenMs(5)
                    .batchEfficiencyFactor(0.1)
                    .maxBatchSize(8)
                    .build())
            .build();

    // Start API
    InferenceController controller = new InferenceController();
    Javalin app = Javalin.create();

    controller.registerRoutes(app);
    app.start(8081);
  }
}
