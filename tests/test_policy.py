import unittest

import pandas as pd

from adaptive_llm_router.router.policy import cost_first_policy


class CostFirstPolicyTest(unittest.TestCase):
    def test_selects_the_cheapest_eligible_model(self):
        capability_table = pd.DataFrame([
            {
                "model": "premium-model",
                "task_type": "coding",
                "quality_medium": 0.95,
                "input_usd_per_1m": 2.0,
                "output_usd_per_1m": 10.0,
            },
            {
                "model": "budget-model",
                "task_type": "coding",
                "quality_medium": 0.85,
                "input_usd_per_1m": 0.5,
                "output_usd_per_1m": 2.0,
            },
        ])

        prediction = {
            "task_type": "coding",
            "difficulty_bucket": "medium",
            "required_quality": 0.80,
            "estimated_input_tokens": 500,
            "expected_output_tokens": 200,
        }

        selected_model = cost_first_policy(
            prediction=prediction,
            capability_table=capability_table,
        )

        self.assertEqual(selected_model, "budget-model")
