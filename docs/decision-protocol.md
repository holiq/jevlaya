# Decision Protocol

## Primitive: choice

Use when exactly one label should be selected.

```json
{
  "type": "choice",
  "instructions": "Classify the ticket.",
  "criteria": {
    "billing": "payment or invoice issue",
    "support": "account or product issue",
    "technical": "software defect"
  }
}
```

Expected normalized output:

```json
{
  "type": "choice",
  "choice": "billing",
  "probabilities": {
    "billing": 0.91,
    "support": 0.06,
    "technical": 0.03
  },
  "confidence": 0.91
}
```

## Primitive: score

Use for an ordered rubric.

```json
{
  "type": "score",
  "instructions": "Rate urgency.",
  "criteria": ["low", "medium", "high"]
}
```

The probability array must preserve the exact order of `criteria`.

## Primitive: noul

Use for binary conditions.

```json
{
  "type": "noul",
  "instructions": "Should this require human review?"
}
```

Example:

```json
{
  "type": "noul",
  "noul": 0.83
}
```

Interpretation: model-estimated probability of yes, not an application-specific action threshold.

## Validation rules

- Question IDs must be unique.
- `choice.criteria` must contain at least two options.
- `score.criteria` must contain at least two ordered levels.
- `noul` does not require criteria.
- Provider output must be normalized before reaching application code.
- Unknown provider fields may be retained in metadata but must not alter canonical semantics.
