package com.inferencelab.gpuworker.api.objects;

import com.fasterxml.jackson.databind.PropertyNamingStrategies;
import com.fasterxml.jackson.databind.annotation.JsonNaming;

@JsonNaming(PropertyNamingStrategies.SnakeCaseStrategy.class)
public record InferenceRequest(
    String prompt,
    int maxTokens,
    int priority
) {}
