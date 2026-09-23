# Routing

Routing is a Jevlaya policy layer.

## Example

```yaml
routing:
  primary: laya
  fallback: jev

policy:
  min_confidence: 0.85
  on_uncertain: fallback
  on_provider_error: fallback
```

## Recommended decision flow

```text
request
  -> validate
  -> primary provider
  -> normalize
  -> inspect uncertainty
  -> accept OR fallback OR escalate
```

## Do not use one global threshold blindly

Thresholds should be evaluated per task and risk class.

For example:

- low-risk categorization may tolerate lower confidence;
- security-sensitive gating may require substantially higher coverage/precision;
- human escalation may be preferable to automatic action when uncertainty is high.

Threshold selection belongs to the application owner and should be backed by labeled evaluation data.

## Routing metadata

Record:

- initial provider
- fallback provider, if used
- reason for fallback
- confidence/probability
- policy version
- final provider
- latency added by fallback
