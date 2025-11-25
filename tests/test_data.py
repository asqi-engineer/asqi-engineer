MOCK_SCORE_CARD_CONFIG = {
    "score_card_name": "Mock Chatbot Scorecard",
    "indicators": [
        {
            "id": "config_easy",
            "type": "audit",
            "name": "Configuration Ease",
            "assessment": [
                {"outcome": "A", "description": "Very easy"},
                {"outcome": "B", "description": "Easy"},
                {"outcome": "C", "description": "Medium"},
                {"outcome": "D", "description": "Hard"},
                {"outcome": "E", "description": "Very hard"},
            ],
        },
        {
            "id": "config_v2",
            "type": "audit",
            "name": "Configuration Ease (v2)",
            "assessment": [
                {"outcome": "A", "description": "Very simple"},
                {"outcome": "B", "description": "Simple"},
                {"outcome": "C", "description": "Moderate"},
                {"outcome": "D", "description": "Advanced"},
                {"outcome": "E", "description": "Expert only"},
            ],
        },
    ],
}

MOCK_AUDIT_RESPONSES = {
    "responses": [
        {"indicator_id": "config_easy", "selected_outcome": "A", "notes": "ok"},
        {"indicator_id": "config_v2", "selected_outcome": "C", "notes": "ok"},
    ]
}
