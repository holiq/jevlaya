"""Standard sample dataset for DecisionBench evaluation."""

from jevlaya.bench.models import BenchmarkDataset, BenchmarkExample
from jevlaya.protocol.models import (
    ChoiceQuestion,
    DecisionRequest,
    NoulQuestion,
    ScoreQuestion,
)


def get_sample_dataset() -> BenchmarkDataset:
    """Return a versioned 1.0.0 sample dataset covering triage, moderation, and scoring."""
    examples = [
        BenchmarkExample(
            id="triage-001",
            request=DecisionRequest(
                state={
                    "ticket_id": "T-101",
                    "subject": "Double charged on my subscription",
                    "body": "Hi, I was billed twice this month. Please issue a refund.",
                },
                questions={
                    "department": ChoiceQuestion(
                        instructions="Which department should handle this request?",
                        criteria={
                            "billing": "Invoices, refunds, and charges",
                            "tech": "Software bugs and service outages",
                            "sales": "Contracts and plan upgrades",
                        },
                    ),
                    "urgency": ScoreQuestion(
                        instructions="Rate the urgency of this ticket.",
                        criteria=["low", "medium", "high"],
                    ),
                    "churn_risk": NoulQuestion(
                        instructions="Does the customer indicate churn risk?",
                    ),
                },
            ),
            ground_truth={
                "department": "billing",
                "urgency": "medium",
                "churn_risk": False,
            },
        ),
        BenchmarkExample(
            id="triage-002",
            request=DecisionRequest(
                state={
                    "ticket_id": "T-102",
                    "subject": "System crash on payment processing",
                    "body": "Checkout service returns 500 error for all customers. Losing revenue!",
                },
                questions={
                    "department": ChoiceQuestion(
                        instructions="Which department should handle this request?",
                        criteria={
                            "billing": "Invoices, refunds, and charges",
                            "tech": "Software bugs and service outages",
                            "sales": "Contracts and plan upgrades",
                        },
                    ),
                    "urgency": ScoreQuestion(
                        instructions="Rate the urgency of this ticket.",
                        criteria=["low", "medium", "high"],
                    ),
                    "churn_risk": NoulQuestion(
                        instructions="Does the customer indicate churn risk?",
                    ),
                },
            ),
            ground_truth={
                "department": "tech",
                "urgency": "high",
                "churn_risk": True,
            },
        ),
        BenchmarkExample(
            id="triage-003",
            request=DecisionRequest(
                state={
                    "ticket_id": "T-103",
                    "subject": "Enterprise licensing quote inquiry",
                    "body": "We want to purchase 500 enterprise seats next quarter.",
                },
                questions={
                    "department": ChoiceQuestion(
                        instructions="Which department should handle this request?",
                        criteria={
                            "billing": "Invoices, refunds, and charges",
                            "tech": "Software bugs and service outages",
                            "sales": "Contracts and plan upgrades",
                        },
                    ),
                    "urgency": ScoreQuestion(
                        instructions="Rate the urgency of this ticket.",
                        criteria=["low", "medium", "high"],
                    ),
                    "churn_risk": NoulQuestion(
                        instructions="Does the customer indicate churn risk?",
                    ),
                },
            ),
            ground_truth={
                "department": "sales",
                "urgency": "low",
                "churn_risk": False,
            },
        ),
    ]

    return BenchmarkDataset(
        name="jevlaya-standard-triage",
        version="1.0.0",
        description="Standard evaluation suite for multi-primitive customer support decisions.",
        examples=examples,
    )
