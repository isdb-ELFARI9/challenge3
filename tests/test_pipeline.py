import pytest
from fastapi.testclient import TestClient
from api import app
import json

client = TestClient(app)

def test_enhance_standard_endpoint():
    # Test data
    test_prompt = "We are seeing more use of diminishing Musharaka in real estate funds, but FAS 4 does not seem to cover shirkah al-ʿaqd structures. How should we treat partner contributions and exits?"
    
    # Make request to the API
    response = client.post(
        "/enhance-standard",
        json={"prompt": test_prompt}
    )
    
    # Check if request was successful
    assert response.status_code == 200
    
    # Parse response
    result = response.json()
    
    # Verify response structure
    assert "document" in result
    assert "change_summary" in result
    assert "reasoning_trace" in result
    
    # Verify content types
    assert isinstance(result["document"], str)
    assert isinstance(result["change_summary"], str)
    assert isinstance(result["reasoning_trace"], str)
    
    # Verify content is not empty
    assert len(result["document"]) > 0
    assert len(result["change_summary"]) > 0
    assert len(result["reasoning_trace"]) > 0

def test_enhance_standard_error_handling():
    # Test with empty prompt
    response = client.post(
        "/enhance-standard",
        json={"prompt": ""}
    )
    assert response.status_code == 500
    
    # Test with invalid JSON
    response = client.post(
        "/enhance-standard",
        data="invalid json"
    )
    assert response.status_code == 422  # Validation error

if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 