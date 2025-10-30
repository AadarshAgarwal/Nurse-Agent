# Fever Proforma Nurse Agent

This system helps healthcare professionals collect patient fever data and generate provisional diagnoses using AI analysis.

## Features

- Interactive data collection based on the Indian OPD fever proforma
- OpenAI GPT-4 integration for medical analysis
- Structured provisional diagnosis and next steps generation
- Option to save assessments to text files

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Create a `.env` file in the project directory with your OpenAI API key:
```
API_KEY=your-openai-api-key-here
```

You can get an OpenAI API key from https://platform.openai.com/api-keys

## Testing

The system includes comprehensive tests to ensure reliability:

### Run Basic Tests
```bash
python run_tests.py
```

This will run:
- ✅ Import and initialization tests
- ✅ Patient data formatting tests
- ✅ Missing data handling tests  
- ✅ Special characters handling tests
- ✅ Integration tests

### Run Full Test Suite (with unittest)
```bash
python -m unittest test_nurse_agent.py -v
```

### Test Coverage
The tests cover:
- API key loading from environment variables
- Patient data collection and formatting
- Error handling for missing API keys
- Special characters and edge cases
- Mock OpenAI API integration
- Complete workflow testing

## Usage

Run the nurse agent:
```bash
python nurse_agent.py
```

The system will:
1. Automatically load your API key from the `.env` file
2. Guide you through collecting patient information
3. Generate a provisional diagnosis and next steps
4. Display the complete assessment
5. Optionally save the results to a file

Note: If the `.env` file is not found or doesn't contain the API key, the system will ask you to enter it manually.

## Input Data Collected

The system collects comprehensive patient data including:

### Patient Information
- Name, Age, Gender, Date, Occupation, Address

### Chief Complaint
- Fever presence, duration, onset

### History of Presenting Illness
- Fever characteristics (frequency, timing, temperature)
- Associated symptoms (chills, sweating, fatigue, etc.)
- Symptom progression

### Medical History
- Chronic illnesses, previous infections
- Medication history and allergies

### Social & Exposure History
- Smoking, alcohol, travel history
- Living conditions
- Contact with sick individuals
- Vector exposure
- Food/water contamination exposure

## Output

The system generates:
- Complete formatted patient data
- AI-generated provisional diagnosis
- Recommended next steps and investigations
- Treatment recommendations
- Follow-up care suggestions

## Important Notes

- This system is designed to assist healthcare professionals, not replace clinical judgment
- All diagnoses are provisional and require clinical validation
- Ensure patient privacy and follow HIPAA/local privacy regulations
- The system is optimized for common fever causes in Indian healthcare settings

## Example Usage

```python
# For programmatic use
from nurse_agent import FeverProformaNurseAgent

agent = FeverProformaNurseAgent(api_key="your-openai-key")
result = agent.run_assessment()
print(result)
```
