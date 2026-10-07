"""Inference helpers."""
from __future__ import annotations

from dataclasses import dataclass

from sklearn.pipeline import Pipeline

from .preprocessing import preprocess_corpus

SAMPLE_MESSAGES = [
    "Congratulations! You've won a $1000 Walmart gift card. Go to http://bit.ly/1234 to claim now.",
    "Hey, are we still meeting up for lunch today?",
    "Urgent! Your account has been compromised. Verify your details here: www.fakebank.com/verify",
    "Reminder: Your appointment is scheduled for tomorrow at 10am.",
    "FREE entry in a weekly competition to win an iPad. Just text WIN to 80085 now!",
]


@dataclass(frozen=True)
class Prediction:
    message: str
    label: str
    spam_probability: float

    def __str__(self) -> str:
        return (f"Message: {self.message}\nPrediction: {self.label}\n"
                f"Spam Probability: {self.spam_probability:.2f}\n"
                f"Not-Spam Probability: {1 - self.spam_probability:.2f}\n" + "-" * 50)


def predict(model: Pipeline, messages: list[str]) -> list[Prediction]:
    processed = preprocess_corpus(messages)
    proba = model.predict_proba(processed)[:, 1]
    return [Prediction(m, "Spam" if p >= 0.5 else "Not-Spam", float(p))
            for m, p in zip(messages, proba)]
