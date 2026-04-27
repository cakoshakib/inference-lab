package com.inferencelab.gpuworker.api;

import com.inferencelab.gpuworker.api.objects.InferenceRequest;
import com.inferencelab.gpuworker.api.objects.InferenceSubmitResponse;
import com.inferencelab.gpuworker.api.objects.StreamMetadata;
import io.javalin.Javalin;
import io.javalin.http.Context;
import io.javalin.http.sse.SseClient;
import java.util.Map;
import java.util.UUID;

public class InferenceController {

  public void registerRoutes(Javalin app) {
    app.post("/api/inference", this::handleSubmit);
    app.get("/api/inference/{requestId}", this::handleGetStatus);
    app.sse("/api/inference/{requestId}/stream", this::handleStream);
  }

  private void handleSubmit(Context ctx) {
    InferenceRequest request = ctx.bodyAsClass(InferenceRequest.class);
    String requestId = UUID.randomUUID().toString();

    // TODO: create job via service, persist to store

    InferenceSubmitResponse response =
        new InferenceSubmitResponse(requestId, "QUEUED", "/api/inference/" + requestId + "/stream");
    ctx.json(response);
  }

  private void handleGetStatus(Context ctx) {
    String requestId = ctx.pathParam("requestId");

    // TODO: look up job from store, return real status
    ctx.status(404).json(Map.of("error", "Not implemented", "request_id", requestId));
  }

  private void handleStream(SseClient client) {
    String requestId = client.ctx().pathParam("requestId");
    client.keepAlive();

    // TODO: look up job, start token generation, stream events
    client.sendEvent("meta", new StreamMetadata(requestId, "QUEUED"));
    client.close();
  }
}
