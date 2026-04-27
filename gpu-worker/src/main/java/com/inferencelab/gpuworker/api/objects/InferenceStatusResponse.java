package com.inferencelab.gpuworker.api.objects;

import com.fasterxml.jackson.databind.PropertyNamingStrategies;
import com.fasterxml.jackson.databind.annotation.JsonNaming;

@JsonNaming(PropertyNamingStrategies.SnakeCaseStrategy.class)
public record InferenceStatusResponse(
    String requestId,
    String status,
    String prompt,
    int maxTokens,
    int priority,
    int tokensGenerated,
    String createdAt) {}
