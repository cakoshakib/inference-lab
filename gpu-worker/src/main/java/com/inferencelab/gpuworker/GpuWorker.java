package com.inferencelab.gpuworker;

import com.inferencelab.gpuworker.api.InferenceController;

import io.javalin.Javalin;

public class GpuWorker {
    public static void main(String[] args) {
        InferenceController controller = new InferenceController();
        Javalin app = Javalin.create();

        controller.registerRoutes(app);
        app.start(8081);
    }
}
