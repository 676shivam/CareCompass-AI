# CareCompass AI — HealthCare Symptom Guidance Project

A portfolio-oriented educational web application built with Python, Flask, scikit-learn, HTML, CSS and JavaScript.

## Features

- Responsive, professional web interface with mobile layouts.
- Symptom selection and free-text symptom matching against the supplied dataset vocabulary.
- Top-three model matches with a clear warning that model scores are not clinical probabilities.
- Conservative activity guidance and specialist suggestions.
- Evidence-aware Ayurveda context with safety cautions; no claim of a proven cure or personalized prescription.
- Condition-aware general food guidance (what to prefer/limit), clearly labeled as educational and not a diagnosis-based diet prescription.
- Optional Google Places API clinic search, with Google Maps search fallback when no key is configured.
- Hazaribagh local-care directory links.
- Emergency safety notice for India (112).

## Run locally

1. Install Python 3.10+.
2. From this directory, install requirements:

   ```bash
   pip install -r requirements.txt
   ```

3. Optional: set a Google Places API key as an environment variable. Enable Places API (New), restrict the key, and check billing/usage in Google Cloud.

   Linux/macOS:
   ```bash
   export GOOGLE_PLACES_API_KEY="YOUR_KEY"
   ```

   Windows PowerShell:
   ```powershell
   $env:GOOGLE_PLACES_API_KEY="YOUR_KEY"
   ```

4. Start the server:

   ```bash
   python app.py
   ```

5. Open `http://127.0.0.1:5000` in your browser.

## Run in Google Colab

Upload and extract the ZIP into `/content/healthcare_ai_project`, then run:

```python
%cd /content/healthcare_ai_project
!pip install -r requirements.txt
```

Optional API key:

```python
import os
from getpass import getpass
os.environ["GOOGLE_PLACES_API_KEY"] = getpass("Google Places API key (blank if unavailable): ").strip()
```

Start the app and preview it:

```python
import threading, time
from google.colab import output
from app import app
threading.Thread(target=lambda: app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False), daemon=True).start()
time.sleep(2)
output.serve_kernel_port_as_iframe(5000, height=1100)
```

If you update the API key after importing `app`, restart the runtime and run the cells in order.

## Project structure

- `app.py` — Flask routes, prediction logic, safety messaging, Ayurveda guidance, and clinic search.
- `templates/index.html` — responsive user interface.
- `model_bundle.joblib` — trained model and metadata.
- `data/` — supplied symptom dataset, disease descriptions, precautions, and symptom severity data.
- `requirements.txt` — Python dependencies.
- `training_metrics.json` — training metadata if available.

## Important limitations

- This is an educational portfolio prototype, not a medical device and not a diagnostic service.
- The source dataset may contain repeated/permuted symptom combinations and has not been clinically validated. Model outputs must not be presented as true disease probabilities.
- The safety keyword screen is incomplete and cannot rule out emergencies.
- Exercise guidance is conservative and is not a personalized rehabilitation prescription.
- Ayurveda content summarizes evidence limitations and safety cautions; it must not replace proven care. Herbal products can interact with medicines or contain contaminants.
- Clinic links are navigation aids, not endorsements. Verify providers, specialties, contact details, hours and availability independently.
- Never expose a Google API key in frontend HTML or a public repository.
