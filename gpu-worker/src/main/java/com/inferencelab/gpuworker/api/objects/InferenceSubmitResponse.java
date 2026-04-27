package com.inferencelab.gpuworker.api.objects;

import com.fasterxml.jackson.databind.PropertyNamingStrategies;
import com.fasterxml.jackson.databind.annotation.JsonNaming;

@JsonNaming(PropertyNamingStrategies.SnakeCaseStrategy.class)
public record InferenceSubmitResponse(
    String requestId,
    String status,
    String sseUrl
) {}
